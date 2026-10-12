"""Multi-turn episode collection: run an env to completion, classify how it ended.

`_collect_single_episode` owns the env lifecycle, the per-turn budget check, and
the truncation taxonomy (`TruncationCause`). `collect_episodes` runs N of these
concurrently per task. Both produce `envs.base.CompletedEpisode` records — the
boundary data type the rest of the multi-turn pipeline consumes.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import nullcontext
from typing import Any

from together.post_training import SamplingParams, SessionClient

from post_training_cookbook.codecs import MessageCodec
from post_training_cookbook.envs.base import (
    Action,
    CompletedEpisode,
    Env,
    EnvFactory,
    PolicySpan,
    StepResult,
    Task,
    TruncationCause,
    classify_truncation,
    close_quietly,
    step_env,
)
from post_training_cookbook.grpo.config import GRPOConfig
from post_training_cookbook.overlong import compute_overlong_penalty
from post_training_cookbook.samples import make_prompt

_LENGTH_TRUNCATION_CAUSES = frozenset({TruncationCause.CONTEXT_OVERFLOW, TruncationCause.MAX_TURNS})


async def cancel_and_drain(tasks: list[asyncio.Task[Any]]) -> None:
    """Cancel rollout tasks and wait for their environments to close.

    Tasks already carrying a cancellation are skipped: a second one lands inside
    their ``finally: await close_quietly(env)`` and aborts the cleanup this
    function exists to await, leaking the environments.
    """
    for task in tasks:
        if not task.cancelling():
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def _run_turn(
    session: SessionClient,
    codec: MessageCodec,
    env: Env,
    context_tokens: list[int],
    *,
    max_tokens: int,
    temperature: float,
    timeout: float,
    failed_parse_reward: float,
    routing_replay: bool,
) -> tuple[Action, StepResult, float, float]:
    """Sample one assistant turn and feed it through the env.

    Returns ``(action, step_result, sample_seconds, env_seconds)``.
    """
    t0 = time.monotonic()
    sample_result = await session.generator.sample_async(
        make_prompt(context_tokens),
        num_samples=1,
        sampling_params=SamplingParams(max_tokens=max_tokens, temperature=temperature),
        timeout=timeout,
        return_routed_experts=routing_replay,
    )
    sample_duration = time.monotonic() - t0

    seq = sample_result.sequences[0]
    key = seq.routed_experts_key if routing_replay else None
    if routing_replay and not key:
        raise RuntimeError("routing_replay is on but the generator returned no routed_experts_key")
    tokens = [int(token) for token in (seq.tokens or [])]
    logprobs = [float(lp) for lp in (seq.logprobs or [])]

    t0 = time.monotonic()
    result = await step_env(codec, env, tokens, failed_parse_reward=failed_parse_reward)
    env_duration = time.monotonic() - t0

    spans = tuple(PolicySpan(int(seg.start_token), int(seg.version)) for seg in sample_result.policy_segments)
    return Action(tokens, logprobs, spans, key), result, sample_duration, env_duration


async def _collect_single_episode(
    session: SessionClient,
    codec: MessageCodec,
    env_factory: EnvFactory,
    task: Task,
    config: GRPOConfig,
    temperature: float,
) -> CompletedEpisode:
    """Run one multi-turn episode to completion.

    Owns the env lifecycle: constructs via ``env_factory`` and guarantees
    ``env.close()`` runs even if ``reset``/``step`` raises or the episode
    is cut short (context overflow, max-turns).
    """
    env = env_factory(task)
    actions: list[Action] = []
    sample_durations: list[float] = []
    env_durations: list[float] = []
    reward = 0.0
    terminated = False
    truncation_cause: TruncationCause | None = None
    env_metrics: dict[str, Any] = {}

    try:
        t0 = time.monotonic()
        messages = list(await env.reset(task))
        reset_duration = time.monotonic() - t0

        tools = env.tool_specs
        num_context_assistants = sum(1 for m in messages if m["role"] == "assistant")
        context_tokens = codec.encode_messages(messages, tools=tools, add_generation_prompt=True)
        response_budget = config.max_model_length - len(context_tokens)

        for _ in range(config.max_turns):
            remaining = config.max_model_length - len(context_tokens)
            if remaining < config.min_response_tokens:
                truncation_cause = TruncationCause.CONTEXT_OVERFLOW
                break

            action, result, sample_duration, env_duration = await _run_turn(
                session,
                codec,
                env,
                context_tokens,
                max_tokens=min(config.max_sample_tokens, remaining),
                temperature=temperature,
                timeout=config.timeout,
                failed_parse_reward=config.failed_parse_reward,
                routing_replay=config.routing_replay,
            )
            actions.append(action)
            sample_durations.append(sample_duration)
            env_durations.append(env_duration)

            if result.done:
                terminated = result.terminated
                truncation_cause = classify_truncation(result)
                reward = result.reward
                messages = list(result.messages)
                env_metrics = dict(result.metrics)
                break

            messages = list(env.messages)
            context_tokens = codec.encode_messages(env.messages, tools=tools, add_generation_prompt=True)
        else:
            truncation_cause = TruncationCause.MAX_TURNS
    finally:
        await close_quietly(env)

    # Soft overlong reward shaping (DAPO §3.2): only length-budget
    # exhaustion gets the additive penalty; parse and env truncations do not.
    raw_reward = reward
    if truncation_cause in _LENGTH_TRUNCATION_CAUSES and config.overlong_penalty:
        reward += compute_overlong_penalty(
            response_length=sum(len(action.tokens) for action in actions),
            max_response_length=response_budget,
            overlong_buffer_len=config.overlong_buffer_len,
            penalty_factor=config.overlong_penalty_factor,
        )

    metrics: dict[str, Any] = {
        **env_metrics,
        "raw_reward": raw_reward,
        "timing/reset_s": reset_duration,
        "timing/sample_s": sum(sample_durations),
        "timing/env_s": sum(env_durations) + reset_duration,
        "timing/sample_durations": sample_durations,
        "timing/env_step_durations": env_durations,
    }

    return CompletedEpisode(
        task=task,
        messages=messages,
        actions=actions,
        tool_specs=tools,
        reward=reward,
        terminated=terminated,
        truncation_cause=truncation_cause,
        num_turns=len(actions),
        num_context_assistants=num_context_assistants,
        metrics=metrics,
    )


async def collect_episodes(
    session: SessionClient,
    codec: MessageCodec,
    tasks: list[Task],
    config: GRPOConfig,
    env_factory: EnvFactory,
    *,
    sequential: bool = False,
) -> list[list[CompletedEpisode]]:
    """Run multi-turn episodes grouped by task.

    Each task gets ``config.episodes_per_task`` independent episodes (each with
    a fresh env from ``env_factory``). Returns one group per task — the natural
    shape for GRPO advantage computation within each group.

    When ``sequential`` is True, episodes run one at a time instead of
    concurrently. This eliminates batching non-determinism at the cost
    of throughput — useful for deterministic validation.
    """
    if config.max_turns is None or config.max_turns < 1:
        raise ValueError("max_turns must be set and >= 1 for multi-turn collection")

    n_per_task = config.episodes_per_task
    gate = asyncio.Semaphore(1) if sequential else nullcontext()

    async def run_episode(task: Task) -> CompletedEpisode:
        async with gate:
            return await _collect_single_episode(session, codec, env_factory, task, config, config.temperature)

    pending = []
    for task in tasks:
        for _ in range(n_per_task):
            pending.append(asyncio.create_task(run_episode(task)))
    try:
        episodes = list(await asyncio.gather(*pending))
    except BaseException:
        # BaseException on purpose: a CancelledError from the async loop's worker
        # shutdown must reach the siblings too, or their envs never close. Kept over
        # TaskGroup so failures stay flat for every caller of collect_episodes.
        await cancel_and_drain(pending)
        raise
    return [episodes[i : i + n_per_task] for i in range(0, len(episodes), n_per_task)]
