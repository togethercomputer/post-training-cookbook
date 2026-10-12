"""Drive any ``Env`` with a hosted chat-completions endpoint and score the rollout."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from together import AsyncTogether

from post_training_cookbook.envs.base import (
    CompletedEpisode,
    EnvFactory,
    Task,
    TruncationCause,
    classify_truncation,
    close_quietly,
)
from post_training_cookbook.serverless_eval.completions import complete_turn

logger = logging.getLogger(__name__)

DEFAULT_CONCURRENCY = 8


@dataclass(frozen=True)
class SamplingConfig:
    """How the hosted model is sampled on each turn of an eval rollout."""

    temperature: float = 0.0
    max_tokens: int | None = None
    enable_thinking: bool | None = None
    append_unparsed_tool_calls: bool = False


@dataclass(frozen=True)
class EvalSummary:
    """Aggregate scores over the episodes that completed, plus the episodes themselves.

    ``metric_means`` averages every numeric key found in the episodes' ``metrics``,
    surfacing env-specific scores (e.g. ``accuracy`` for single-turn datasets) that
    the fixed summary fields don't name. Envs only report metrics on the terminal
    step, so a truncated episode contributes to ``reward_mean`` but to none of an
    env's keys; the harness's own ``eval/*`` keys are reported for every episode.
    ``metric_counts`` carries each key's denominator so two means printed side by
    side are not read as sharing one.
    """

    reward_mean: float
    num_turns_mean: float
    terminated_ratio: float
    metric_means: dict[str, float]
    metric_counts: dict[str, int]
    num_failed: int
    episodes: list[CompletedEpisode]

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)


async def _run_episode(
    client: AsyncTogether,
    model: str,
    env_factory: EnvFactory,
    task: Task,
    max_turns: int,
    sampling: SamplingConfig,
) -> CompletedEpisode:
    """Run one multi-turn episode to completion against the chat endpoint.

    Owns the env lifecycle: guarantees ``env.close()`` runs even if a turn or
    the endpoint raises. The returned record carries no ``actions`` — the
    endpoint returns structured messages, not sampled tokens, so there is nothing
    to train on, and the only causes this path can report are ``MAX_TURNS`` and
    ``ENV``. A turn the endpoint cut off at ``max_tokens`` still counts as a normal
    turn; the ``eval/length_capped`` metric is how that shows up in the score.
    """
    env = env_factory(task)
    reward = 0.0
    terminated = False
    truncation_cause: TruncationCause | None = None
    num_turns = 0
    metrics: dict[str, Any] = {}
    length_capped = False

    try:
        messages = list(await env.reset(task))

        for _ in range(max_turns):
            turn = await complete_turn(
                client,
                model,
                messages,
                env.tool_specs,
                temperature=sampling.temperature,
                max_tokens=sampling.max_tokens,
                enable_thinking=sampling.enable_thinking,
                append_unparsed_tool_calls=sampling.append_unparsed_tool_calls,
            )
            num_turns += 1
            length_capped = length_capped or turn.finish_reason == "length"
            try:
                result = await env.step(turn.message)
            except Exception as exc:
                logger.warning("env.step raised %s: %s; scoring as ENV truncation", type(exc).__name__, exc)
                truncation_cause = TruncationCause.ENV
                break
            if result.done:
                terminated = result.terminated
                truncation_cause = classify_truncation(result)
                reward = result.reward
                messages = list(result.messages)
                metrics = dict(result.metrics)
                break
            messages = list(env.messages)
        else:
            truncation_cause = TruncationCause.MAX_TURNS
    finally:
        await close_quietly(env)

    return CompletedEpisode(
        task=task,
        messages=messages,
        reward=reward,
        terminated=terminated,
        truncation_cause=truncation_cause,
        num_turns=num_turns,
        metrics={**metrics, "eval/length_capped": float(length_capped)},
    )


def _numeric_metrics(episodes: list[CompletedEpisode]) -> dict[str, list[float]]:
    """Collect every numeric metric key's reported values, keyed and sorted by name."""
    values: dict[str, list[float]] = {}
    for ep in episodes:
        for key, value in ep.metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):  # bool is an int subclass
                values.setdefault(key, []).append(float(value))
    return dict(sorted(values.items()))


def _summarize(episodes: list[CompletedEpisode], num_failed: int) -> EvalSummary:
    """Aggregate completed episodes. Raises ``ValueError`` if there is nothing to average."""
    if not episodes:
        raise ValueError("cannot summarize zero episodes")

    values = _numeric_metrics(episodes)
    return EvalSummary(
        reward_mean=float(np.mean([ep.reward for ep in episodes])),
        num_turns_mean=float(np.mean([ep.num_turns for ep in episodes])),
        terminated_ratio=sum(1 for ep in episodes if ep.terminated) / len(episodes),
        metric_means={key: float(np.mean(vals)) for key, vals in values.items()},
        metric_counts={key: len(vals) for key, vals in values.items()},
        num_failed=num_failed,
        episodes=episodes,
    )


async def evaluate(
    *,
    client: AsyncTogether,
    model: str,
    tasks: list[Task],
    env_factory: EnvFactory,
    max_turns: int,
    sampling: SamplingConfig = SamplingConfig(),
    max_concurrency: int = DEFAULT_CONCURRENCY,
) -> EvalSummary:
    """Evaluate a hosted chat model on ``tasks`` via chat-completions rollouts.

    At most ``max_concurrency`` episodes in flight. Individual failures are
    logged and dropped (counted in ``num_failed``); if every episode fails
    — a bad key, an unserved model, an endpoint that is down — this raises
    rather than reporting a zero score indistinguishable from a bad model.
    """
    if max_turns < 1:
        raise ValueError(f"max_turns must be >= 1, got {max_turns}")
    if max_concurrency < 1:
        raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}")
    if not tasks:
        raise ValueError("tasks must not be empty")

    gate = asyncio.Semaphore(max_concurrency)

    async def run_one(task: Task) -> CompletedEpisode:
        async with gate:
            return await _run_episode(client, model, env_factory, task, max_turns, sampling)

    results = await asyncio.gather(*(run_one(task) for task in tasks), return_exceptions=True)

    episodes: list[CompletedEpisode] = []
    failed = 0
    for result in results:
        if isinstance(result, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
            raise result  # an interrupted run must not report a score over the survivors
        if isinstance(result, BaseException):
            logger.warning("episode failed (%s: %s); skipping", type(result).__name__, result)
            failed += 1
        else:
            episodes.append(result)

    if not episodes:
        raise RuntimeError(f"all {failed} episodes failed; see the logged warnings above")

    return _summarize(episodes, failed)


def run_and_report(
    *,
    model: str,
    tasks: list[Task],
    env_factory: EnvFactory,
    max_turns: int,
    sampling: SamplingConfig = SamplingConfig(),
    max_concurrency: int = DEFAULT_CONCURRENCY,
    label: str = "",
    out_dir: str = "runs",
    timeout: float = 7200.0,
) -> EvalSummary:
    """Evaluate ``model`` on ``tasks``, print a summary, and dump the episodes.

    The blocking entry point recipes call: owns the event loop and the Together
    client (credentials come from ``TOGETHER_API_KEY`` / ``TOGETHER_BASE_URL``).
    ``label`` is free-text appended to the config line (e.g. ``"domain=retail"``).
    Uses ``asyncio.run``; notebooks should ``await evaluate(...)`` with their own client.
    """
    load_dotenv(override=True)
    header = " | ".join(part for part in (f"Evaluating {model}", label) if part)
    print(
        f"{header} | tasks={len(tasks)} max_turns={max_turns} "
        f"temp={sampling.temperature} thinking={sampling.enable_thinking} "
        f"concurrency={max_concurrency}"
    )

    async def evaluate_with_client() -> EvalSummary:
        client = AsyncTogether(timeout=timeout)
        try:
            return await evaluate(
                client=client,
                model=model,
                tasks=tasks,
                env_factory=env_factory,
                max_turns=max_turns,
                sampling=sampling,
                max_concurrency=max_concurrency,
            )
        finally:
            await client.close()

    summary = asyncio.run(evaluate_with_client())

    print(
        f"\nEVAL reward={summary.reward_mean:.3f} "
        f"terminated={summary.terminated_ratio:.2%} turns={summary.num_turns_mean:.1f} "
        f"episodes={summary.num_episodes} failed={summary.num_failed}"
    )
    if summary.metric_means:
        scores = " ".join(
            f"{key}={value:.3f} ({summary.metric_counts[key]}/{summary.num_episodes})"
            for key, value in summary.metric_means.items()
        )
        print(f"  metrics: {scores}")

    out_path = _episodes_path(Path(out_dir), model, label)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([ep.to_dict() for ep in summary.episodes], indent=2))
    print(f"Wrote {len(summary.episodes)} episodes to {out_path}")

    return summary


def _episodes_path(out_dir: Path, model: str, label: str) -> Path:
    """One directory per eval run, mirroring the ``runs/<run_name>/`` training layout."""
    slug = "_".join(part for part in ("serverless_eval", label, model.split("/")[-1]) if part)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return out_dir / f"{re.sub(r'[^A-Za-z0-9._-]+', '-', slug)}_{timestamp}" / "episodes.json"
