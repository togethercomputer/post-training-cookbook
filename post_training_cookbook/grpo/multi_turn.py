"""Multi-turn step orchestration: flatten episodes, build samples, drive validation.

`flatten_episode` turns a completed episode into `TrainableSequence`s for
`build_policy_sample` (actions get mask=1; prompts, tool responses, and headers
get mask=0).

`make_multi_turn_collector` is the per-step `collect_samples` closure consumed
by `training_loop`. It pulls episode groups from `collect_episodes`, computes
group-relative advantages, and emits training samples + a W&B step log.
`run_multi_turn_validation` reuses `collect_episodes` with greedy decoding.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from typing import Any, NamedTuple, TypeVar

import numpy as np
from together.post_training import Sample, SessionClient

from post_training_cookbook.codecs import MessageCodec
from post_training_cookbook.envs.base import Action, CompletedEpisode, EnvFactory, Task, TruncationCause
from post_training_cookbook.grpo.batching import oversample_until_kept
from post_training_cookbook.grpo.config import GRPOConfig
from post_training_cookbook.grpo.data_progress import DataProgress
from post_training_cookbook.grpo.episode import collect_episodes
from post_training_cookbook.grpo.loop import CollectSamplesFn
from post_training_cookbook.grpo.samples import GroupAdvantages, build_policy_sample, compute_group_advantages
from post_training_cookbook.run_artifacts import Run, export_episodes, log_step
from post_training_cookbook.samples import is_trainable
from post_training_cookbook.sequence_split import Completion, Context, TrainableSequence, Turn, split_turns
from post_training_cookbook.types import Message, ToolSpec

logger = logging.getLogger(__name__)


def flatten_episode(
    codec: MessageCodec,
    messages: list[Message],
    actions: list[Action],
    tools: list[ToolSpec] | None = None,
    num_context_assistants: int = 0,
) -> list[TrainableSequence]:
    """Flatten a multi-turn episode into GRPO training sequences.

    Sampled assistant tokens from ``actions`` are the trained completions;
    prompts, history, and template scaffolding are context. ``split_turns``
    handles the prefix-stability split (see that module).

    ``num_context_assistants`` leading assistant messages came from the
    environment (e.g. ``env.reset()``), not the model; they are treated as
    context and not paired with actions.
    """
    assistant_idxs = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    sampled_assistant_idxs = assistant_idxs[num_context_assistants:]
    if len(sampled_assistant_idxs) != len(actions):
        raise ValueError(
            f"Number of sampled assistant messages ({len(sampled_assistant_idxs)}) "
            f"!= number of actions ({len(actions)}) "
            f"(total assistants={len(assistant_idxs)}, context={num_context_assistants})"
        )

    # Anything after the last assistant turn never becomes trainable here.
    messages = messages[: sampled_assistant_idxs[-1]]

    turns = [
        Turn(
            prompt=Context(codec.encode_messages(messages[:assistant_idx], tools=tools, add_generation_prompt=True)),
            completion=Completion(action.tokens, action.logprobs, action.routed_experts_key),
        )
        for action, assistant_idx in zip(actions, sampled_assistant_idxs, strict=True)
    ]
    return split_turns(turns)


def build_episode_group_samples(
    group: list[CompletedEpisode],
    advantages: list[float],
    codec: MessageCodec,
    *,
    overlong_masking: bool = False,
    routing_replay: bool = False,
) -> list[Sample]:
    """Build GRPO training samples for a group of episodes with pre-computed advantages.

    When ``overlong_masking`` is enabled, truncated episodes are dropped
    entirely — their rewards still affect the group baseline (advantages were
    already computed) but they produce no gradient (DAPO overlong filtering).
    When ``routing_replay`` is enabled, every trainable sequence must carry a
    generator routing key, which is forwarded unchanged to the trainer.
    """
    samples: list[Sample] = []
    untrainable = 0
    episodes_examined = 0
    for episode, advantage in zip(group, advantages, strict=True):
        if overlong_masking and episode.truncated:
            continue
        try:
            segments = list(
                flatten_episode(
                    codec, episode.messages, episode.actions, episode.tool_specs, episode.num_context_assistants
                )
            )
        except ValueError:
            roles = [msg.get("role", "?") for msg in episode.messages]
            decoded_actions = [codec.decode(action.tokens) for action in episode.actions]
            logger.warning(
                "Skipping episode (task=%s): flatten_episode failed — "
                "roles=%s, actions=%d, context_assistants=%d, terminated=%s, truncated=%s\n"
                "  decoded actions: %s",
                episode.task.metadata,
                roles,
                len(episode.actions),
                episode.num_context_assistants,
                episode.terminated,
                episode.truncated,
                decoded_actions,
            )
            continue
        episodes_examined += 1
        for segment in segments:
            if not is_trainable(segment.loss_mask):
                untrainable += 1  # an action emitted nothing, so no slot trains
                continue
            if routing_replay and not segment.routed_experts_key:
                raise ValueError("routing_replay is on but a training sequence carries no routed_experts_key")
            samples.append(
                build_policy_sample(
                    tokens=segment.tokens,
                    logprobs=segment.logprobs,
                    loss_mask=segment.loss_mask,
                    advantage=advantage,
                    routed_experts_key=segment.routed_experts_key if routing_replay else None,
                )
            )

    # Aggregated: a policy collapsed into empty actions would otherwise warn
    # once per segment per episode per step.
    if untrainable:
        logger.warning(
            "Dropped %d segments that train no slot across %d episodes examined", untrainable, episodes_examined
        )
    return samples


def _distribution_stats(values: list[float], prefix: str) -> dict[str, float]:
    """Compute mean, p95, and max for a list of values."""
    if not values:
        return {f"{prefix}_mean": 0.0, f"{prefix}_p95": 0.0, f"{prefix}_max": 0.0}
    arr = np.array(values)
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_p95": float(np.percentile(arr, 95)),
        f"{prefix}_max": float(np.max(arr)),
    }


@dataclass
class _StepMetrics:
    """Scalar accumulator for a step's episodes; ``finalize`` produces the W&B step_log."""

    rewards: list[float] = field(default_factory=list)
    raw_rewards: list[float] = field(default_factory=list)
    num_turns: list[int] = field(default_factory=list)
    sample_times: list[float] = field(default_factory=list)
    env_times: list[float] = field(default_factory=list)
    reset_times: list[float] = field(default_factory=list)
    sample_durations: list[float] = field(default_factory=list)
    env_step_durations: list[float] = field(default_factory=list)
    terminated: int = 0
    truncated: int = 0
    parse_failed: int = 0
    context_overflow: int = 0

    def add(self, ep: CompletedEpisode) -> None:
        self.rewards.append(ep.reward)
        self.raw_rewards.append(ep.metrics.get("raw_reward", ep.reward))
        self.num_turns.append(ep.num_turns)
        self.sample_times.append(ep.metrics.get("timing/sample_s", 0.0))
        self.env_times.append(ep.metrics.get("timing/env_s", 0.0))
        self.reset_times.append(ep.metrics.get("timing/reset_s", 0.0))
        self.sample_durations.extend(ep.metrics.get("timing/sample_durations", []))
        self.env_step_durations.extend(ep.metrics.get("timing/env_step_durations", []))
        self.terminated += int(ep.terminated)
        self.truncated += int(ep.truncated)
        self.parse_failed += int(ep.truncation_cause == TruncationCause.PARSE_FAILURE)
        self.context_overflow += int(ep.truncation_cause == TruncationCause.CONTEXT_OVERFLOW)

    def finalize(
        self,
        *,
        num_training_samples: int,
        num_groups: int,
        zero_variance_skipped_groups: int,
    ) -> dict[str, Any]:
        count = len(self.rewards)
        total_times = [s + e for s, e in zip(self.sample_times, self.env_times)]
        sample_fraction = (
            float(np.mean([s / t for s, t in zip(self.sample_times, total_times) if t > 0]))
            if any(t > 0 for t in total_times)
            else 0.0
        )

        return {
            "reward_mean": float(np.mean(self.rewards)),
            "reward_std": float(np.std(self.rewards)),
            "raw_reward_mean": float(np.mean(self.raw_rewards)),
            "num_turns_mean": float(np.mean(self.num_turns)),
            "num_turns_max": max(self.num_turns, default=0),
            "terminated_ratio": self.terminated / max(count, 1),
            "truncated_ratio": self.truncated / max(count, 1),
            "parse_failure_ratio": self.parse_failed / max(count, 1),
            "context_overflow_ratio": self.context_overflow / max(count, 1),
            "num_episodes": count,
            "num_training_samples": num_training_samples,
            "zero_variance_skip_ratio": zero_variance_skipped_groups / max(num_groups, 1),
            # Per-operation distribution
            **_distribution_stats(self.sample_durations, "timing/op_sample_s"),
            **_distribution_stats(self.env_step_durations, "timing/op_env_step_s"),
            # Per-episode distribution
            **_distribution_stats(self.sample_times, "timing/episode_sample_s"),
            **_distribution_stats(self.env_times, "timing/episode_env_s"),
            "timing/episode_reset_mean_s": float(np.mean(self.reset_times)) if self.reset_times else 0.0,
            "timing/sample_fraction": sample_fraction,
        }


_T = TypeVar("_T")


def _drain(items: list[_T]) -> Iterator[_T]:
    """Yield each item, releasing the list's reference to it after the consumer is done."""
    for i in range(len(items)):
        item = items[i]
        items[i] = None  # type: ignore[call-overload]
        yield item


class TaskOutcome(NamedTuple):
    """One task's collected episode group plus its computed group advantages.

    ``advantages.values is None`` when the group has zero reward variance —
    the group still counts toward attempt accounting but produces no training
    samples.
    """

    group: list[CompletedEpisode]
    advantages: GroupAdvantages


def task_outcome(group: list[CompletedEpisode]) -> TaskOutcome:
    """Pair a collected episode group with its group-relative advantages."""
    return TaskOutcome(group, compute_group_advantages([episode.reward for episode in group]))


def build_step_samples(
    outcomes: list[TaskOutcome],
    *,
    stale: list[TaskOutcome],
    codec: MessageCodec,
    config: GRPOConfig,
    run: Run,
    step: int,
) -> tuple[list[Sample], dict[str, Any]]:
    """Consume episode outcomes, building samples, metrics, and episode exports.

    Both lists are drained: each group's episodes are released as they are consumed,
    so the entries are gone on return (the lists keep their length). Read anything
    else you need off them first.

    Args:
        outcomes: Groups considered for training. Those with zero reward variance
            build no samples but still count toward ``zero_variance_skip_ratio``,
            exactly as in the lockstep loop.
        stale: Groups the async loop excluded for exceeding its staleness budget.
            They contribute reward metrics and the episode export but sit outside
            the ``zero_variance_skip_ratio`` accounting entirely, so that ratio
            describes the same training cohort it does in the lockstep loop.
        codec: Codec used to render each episode into trainable sequences.
        config: Supplies the overlong-masking policy.
        run: Destination for the episode export.
        step: Zero-indexed step the export is filed under.
    """
    kept_groups = sum(1 for outcome in outcomes if outcome.advantages.values is not None)
    zero_variance_groups = len(outcomes) - kept_groups
    num_groups = len(outcomes)

    samples: list[Sample] = []
    serialized_groups: list[list[dict[str, Any]]] = []
    metrics = _StepMetrics()

    for outcome in _drain(outcomes):
        if outcome.advantages.values is not None:
            samples.extend(
                build_episode_group_samples(
                    outcome.group,
                    outcome.advantages.values,
                    codec,
                    overlong_masking=config.overlong_masking,
                    routing_replay=config.routing_replay,
                )
            )
        serialized_groups.append([episode.to_dict() for episode in outcome.group])
        for episode in outcome.group:
            metrics.add(episode)

    for outcome in _drain(stale):
        serialized_groups.append([episode.to_dict() for episode in outcome.group])
        for episode in outcome.group:
            metrics.add(episode)

    step_log = metrics.finalize(
        num_training_samples=len(samples),
        num_groups=num_groups,
        zero_variance_skipped_groups=zero_variance_groups,
    )
    export_episodes(run, step, serialized_groups)

    return samples, step_log


def make_multi_turn_collector(
    config: GRPOConfig,
    codec: MessageCodec,
    tasks: list[Task],
    env_factory: EnvFactory,
    run: Run,
    *,
    data_progress: DataProgress,
) -> CollectSamplesFn:
    """Build the per-step ``collect_samples`` closure for multi-turn training.

    When ``config.max_sampling_attempts > 1``, each step keeps drawing batches
    of tasks until ``config.batch_size`` groups have non-trivial advantages or
    the cap is reached (DAPO §3.2 dynamic sampling).
    """

    def collect_samples(step: int, session: SessionClient) -> tuple[list[Sample], dict[str, Any]]:
        def draw_batch() -> list[TaskOutcome]:
            batch_tasks = [tasks[i] for i in data_progress.draw_batch()]
            groups = session.run(collect_episodes(session, codec, batch_tasks, config, env_factory))
            return [task_outcome(group) for group in groups]

        outcomes, attempts = oversample_until_kept(
            draw_batch,
            is_kept=lambda outcome: outcome.advantages.values is not None,
            target=config.batch_size,
            max_attempts=config.max_sampling_attempts,
        )

        num_groups = len(outcomes)
        kept_groups = sum(1 for outcome in outcomes if outcome.advantages.values is not None)
        samples, step_log = build_step_samples(outcomes, stale=[], codec=codec, config=config, run=run, step=step)

        if config.max_sampling_attempts > 1:
            step_log["dynamic_sampling/attempts"] = attempts
            step_log["dynamic_sampling/total_groups_sampled"] = num_groups
            step_log["dynamic_sampling/kept_groups"] = kept_groups

        print(
            f"[{session.session_id[:8]}] [step {step}] reward={step_log['reward_mean']:.3f}"
            f" turns={step_log['num_turns_mean']:.1f} (max={step_log['num_turns_max']})"
            f" terminated={step_log['terminated_ratio']:.2%}"
            f" samples={len(samples)} attempts={attempts}",
            flush=True,
        )

        return samples, step_log

    return collect_samples


def run_multi_turn_validation(
    step: int,
    *,
    config: GRPOConfig,
    codec: MessageCodec,
    val_tasks: list[Task],
    env_factory: EnvFactory,
    session: SessionClient,
    run: Run,
) -> None:
    """Run validation rollouts for ``config.validation_iterations`` and log val/* metrics.

    Reuses ``collect_episodes`` with ``episodes_per_task=1`` and (by default) greedy
    decoding so validation behaves like the training collector but without learning.
    """
    session.run(
        run_multi_turn_validation_async(
            step,
            config=config,
            codec=codec,
            val_tasks=val_tasks,
            env_factory=env_factory,
            session=session,
            run=run,
        )
    )


async def run_multi_turn_validation_async(
    step: int,
    *,
    config: GRPOConfig,
    codec: MessageCodec,
    val_tasks: list[Task],
    env_factory: EnvFactory,
    session: SessionClient,
    run: Run,
) -> None:
    """Run and log multi-turn validation without re-entering the session event loop."""
    val_max_turns = config.val_max_turns if config.val_max_turns is not None else config.max_turns
    val_temperature = config.val_temperature if config.val_temperature is not None else 0.0
    val_config = replace(
        config, episodes_per_task=1, temperature=val_temperature, max_turns=val_max_turns, routing_replay=False
    )
    iterations = max(config.validation_iterations, 1)

    async def run_iter() -> list[list[CompletedEpisode]]:
        return await collect_episodes(
            session, codec, val_tasks, val_config, env_factory, sequential=config.sequential_validation
        )

    async def run_all() -> tuple[list[list[list[CompletedEpisode]]], int]:
        if config.sequential_validation:
            awaitables = [run_iter() for _ in range(iterations)]
        else:
            awaitables = [asyncio.create_task(run_iter()) for _ in range(iterations)]

        successes: list[list[list[CompletedEpisode]]] = []
        failed = 0
        for awaitable in awaitables:
            try:
                successes.append(await awaitable)
            except Exception as exc:
                logger.warning("validation iteration failed (%s: %s); skipping", type(exc).__name__, exc)
                failed += 1
        return successes, failed

    iter_groups, failed_iterations = await run_all()

    all_groups: list[list[CompletedEpisode]] = []
    iter_reward_means: list[float] = []
    for groups in iter_groups:
        all_groups.extend(groups)
        iter_episodes = [ep for group in groups for ep in group]
        if iter_episodes:
            iter_reward_means.append(float(np.mean([ep.reward for ep in iter_episodes])))

    episodes = [ep for group in all_groups for ep in group]
    if not episodes:
        logger.warning("validation step %d: all %d iterations failed", step, iterations)
        log_step(run, step, {"val/all_iterations_failed": 1.0})
        return

    export_episodes(
        run,
        step,
        [[ep.to_dict() for ep in group] for group in all_groups],
        prefix="val",
    )

    reward_mean = float(np.mean([ep.reward for ep in episodes]))
    raw_reward_mean = float(np.mean([ep.metrics.get("raw_reward", ep.reward) for ep in episodes]))
    num_turns_mean = float(np.mean([ep.num_turns for ep in episodes]))
    terminated_ratio = sum(1 for ep in episodes if ep.terminated) / max(len(episodes), 1)
    reward_iter_std = float(np.std(iter_reward_means, ddof=1)) if len(iter_reward_means) > 1 else 0.0

    log_step(
        run,
        step,
        {
            "val/reward_mean": reward_mean,
            "val/reward_iter_std": reward_iter_std,
            "val/raw_reward_mean": raw_reward_mean,
            "val/num_turns_mean": num_turns_mean,
            "val/terminated_ratio": terminated_ratio,
            "val/iter_fail_ratio": failed_iterations / max(iterations, 1),
        },
    )
    print(
        f"[{session.session_id[:8]}] [step {step}] VALIDATION reward={reward_mean:.3f}"
        f" (iter_std={reward_iter_std:.3f}, n_iter={iterations - failed_iterations}/{iterations})"
        f" turns={num_turns_mean:.1f} terminated={terminated_ratio:.2%}",
        flush=True,
    )
