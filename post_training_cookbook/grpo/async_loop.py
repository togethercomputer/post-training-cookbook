"""Bounded multi-turn rollouts overlapping GRPO optimizer steps."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable
from statistics import fmean
from typing import Any, NamedTuple

from together.post_training import SessionClient

from post_training_cookbook.codecs import MessageCodec
from post_training_cookbook.envs.base import EnvFactory, Task
from post_training_cookbook.grpo.config import GRPOConfig
from post_training_cookbook.grpo.data_progress import DataProgress
from post_training_cookbook.grpo.episode import cancel_and_drain, collect_episodes
from post_training_cookbook.grpo.loop import apply_train_step
from post_training_cookbook.grpo.multi_turn import TaskOutcome, build_step_samples, task_outcome
from post_training_cookbook.run_artifacts import Run

AsyncValidateStepFn = Callable[[int], Awaitable[None]]


class _QueuedGroup(NamedTuple):
    """A collected group and the policy version current when its rollout started."""

    outcome: TaskOutcome
    scheduled_version: int
    task_index: int


def _drift_ratio(queued_groups: list[_QueuedGroup]) -> float:
    """Fraction of generated tokens produced by a policy other than the scheduled one.

    Actions with no segment data are excluded from both sides of the ratio, so
    "unreported" cannot read as "no drift". See https://docs.together.ai/docs/post-training/monitoring for the two
    sources of drift this covers.
    """
    drifted = total = 0
    for queued in queued_groups:
        for episode in queued.outcome.group:
            for action in episode.actions:
                segments = sorted(action.policy_segments)
                if not segments:
                    continue
                # Span-relative, so numerator and denominator cover the same range.
                ends = [segment.start_token for segment in segments[1:]] + [len(action.tokens)]
                total += len(action.tokens) - segments[0].start_token
                for segment, end in zip(segments, ends, strict=True):
                    if segment.version != queued.scheduled_version:
                        drifted += end - segment.start_token
    return drifted / total if total else 0.0


def _async_metrics(
    accepted: list[_QueuedGroup],
    stale: list[TaskOutcome],
    degenerate: list[TaskOutcome],
    *,
    policy_version: int,
    queue_depth: int,
    train_wait: float,
) -> dict[str, Any]:
    """Build the ``async/*`` step log for one admission round."""
    metrics: dict[str, Any] = {
        "async/accepted_groups": len(accepted),
        "async/dropped_stale_groups": len(stale),
        "async/dropped_zero_variance_groups": len(degenerate),
        "async/queue_depth": queue_depth,
        "async/train_wait_s": train_wait,
    }
    if not accepted:
        # Omitted, not zeroed: "no data" must not read as "perfectly on-policy".
        # Also keeps fmean() off an empty sequence, which raises.
        return metrics
    return metrics | {
        "async/staleness_mean": fmean(policy_version - queued.scheduled_version for queued in accepted),
        "async/staleness_max": max(policy_version - queued.scheduled_version for queued in accepted),
        "async/mid_rollout_drift_ratio": _drift_ratio(accepted),
    }


async def async_training_loop(
    total_steps: int,
    session: SessionClient,
    config: GRPOConfig,
    codec: MessageCodec,
    tasks: list[Task],
    env_factory: EnvFactory,
    run: Run,
    validate: AsyncValidateStepFn | None = None,
    *,
    data_progress: DataProgress,
    start_step: int = 0,
) -> None:
    """Train on fresh, nonzero-variance groups from a continuously replenished pool.

    ``groups_in_flight`` workers each collect one task's group and hand it over
    through a bounded queue. Groups carry the policy version current when their
    rollout *started*; each step admits those within ``config.max_steps_off_policy``.

    Zero-variance groups are capped at ``batch_size * max_sampling_attempts`` per
    step, so a reward function that separates nothing yields a short step instead of
    refilling forever. Stale groups are exempt: ``policy_version`` is frozen across a
    step's admission, so they cannot spin. Validation cancels the workers and awaits
    their environments; queued groups carry over and cancelled tasks are reissued.
    See https://docs.together.ai/docs/post-training/grpo#async-training.
    """
    if not tasks:
        raise ValueError("async training requires at least one task")
    if config.max_steps_off_policy is None:
        raise ValueError("async training requires max_steps_off_policy to be set")
    if config.episodes_per_task < 2:
        raise ValueError("async training requires episodes_per_task >= 2 for nonzero advantages")

    groups_in_flight = int(config.groups_in_flight)
    max_steps_off_policy = config.max_steps_off_policy
    zero_variance_budget = config.batch_size * config.max_sampling_attempts

    group_queue: asyncio.Queue[_QueuedGroup] = asyncio.Queue(maxsize=groups_in_flight)
    carried: deque[_QueuedGroup] = deque()
    # Server-side policy versions are session-scoped and start at 0 (they reset on
    # resume and on manager re-attach), so the client counter starts there too and
    # shares numbering with the versions the sampler reports per token span.
    policy_version = 0
    overall_start = time.monotonic()

    async def worker() -> None:
        while True:
            task_index = data_progress.claim()
            task = tasks[task_index % len(tasks)]
            scheduled_version = policy_version
            try:
                (group,) = await collect_episodes(session, codec, [task], config, env_factory)
                await group_queue.put(_QueuedGroup(task_outcome(group), scheduled_version, task_index))
            except asyncio.CancelledError:
                data_progress.retry(task_index)
                raise

    if validate is not None:
        await validate(start_step)

    try:
        async with asyncio.TaskGroup() as pool:
            workers = [pool.create_task(worker()) for _ in range(groups_in_flight)]
            try:
                for step in range(start_step, total_steps):
                    collect_start = time.monotonic()
                    queue_depth = group_queue.qsize() + len(carried)
                    accepted: list[_QueuedGroup] = []
                    stale: list[TaskOutcome] = []
                    degenerate: list[TaskOutcome] = []
                    train_wait = 0.0
                    while len(accepted) < config.batch_size and len(degenerate) < zero_variance_budget:
                        wait_start = time.monotonic()
                        queued = carried.popleft() if carried else await group_queue.get()
                        train_wait += time.monotonic() - wait_start
                        data_progress.consume(queued.task_index)
                        if policy_version - queued.scheduled_version > max_steps_off_policy:
                            stale.append(queued.outcome)
                        elif queued.outcome.advantages.values is None:
                            degenerate.append(queued.outcome)
                        else:
                            accepted.append(queued)
                    del queued  # Release the last group's episodes along with the lists below.

                    async_log = _async_metrics(
                        accepted,
                        stale,
                        degenerate,
                        policy_version=policy_version,
                        queue_depth=queue_depth,
                        train_wait=train_wait,
                    )
                    # Degenerate groups ride along so zero_variance_skip_ratio keeps
                    # the meaning it has in the lockstep loop.
                    outcomes = [queued.outcome for queued in accepted] + degenerate
                    accepted.clear()  # build_step_samples drains `outcomes`; aliases go first.
                    degenerate.clear()
                    samples, step_log = build_step_samples(
                        outcomes, stale=stale, codec=codec, config=config, run=run, step=step
                    )
                    print(
                        f"[{session.session_id[:8]}] [step {step + 1}]"
                        f" reward={step_log['reward_mean']:.3f}"
                        f" turns={step_log['num_turns_mean']:.1f}"
                        f" groups={async_log['async/accepted_groups']}"
                        f" (stale={async_log['async/dropped_stale_groups']}"
                        f" degenerate={async_log['async/dropped_zero_variance_groups']})"
                        f" samples={len(samples)} wait={train_wait:.1f}s",
                        flush=True,
                    )
                    trained = await apply_train_step(
                        session,
                        samples,
                        step,
                        total_steps,
                        config,
                        run,
                        {**step_log, **async_log},
                        time.monotonic() - collect_start,
                        overall_start=overall_start,
                    )
                    if trained:
                        policy_version += 1

                    if validate is not None and (step + 1) % config.validation_interval == 0:
                        # Keep completed groups; cancelled task leases are reissued by new workers.
                        while not group_queue.empty():
                            carried.append(group_queue.get_nowait())
                        await cancel_and_drain(workers)
                        await validate(step + 1)
                        if step + 1 < total_steps:
                            workers = [pool.create_task(worker()) for _ in range(groups_in_flight)]
            finally:
                await cancel_and_drain(workers)
    except ExceptionGroup as group:
        # TaskGroup wraps everything, including step-body failures. Surface a lone
        # error flat, as the lockstep loop does; concurrent ones stay grouped.
        if len(group.exceptions) == 1:
            raise group.exceptions[0] from None
        raise
