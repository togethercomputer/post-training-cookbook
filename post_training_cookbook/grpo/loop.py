"""Shared GRPO training loop and learning-rate schedule."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable
from typing import Any

from together.post_training import (
    Sample,
    SessionClient,
)

from post_training_cookbook.config import optim_step_params
from post_training_cookbook.grpo.config import GRPOConfig, policy_loss
from post_training_cookbook.run_artifacts import Run, checkpoint_if_due_async, log_step

CollectSamplesFn = Callable[[int, SessionClient], tuple[list[Sample], dict[str, Any]]]
"""Per-step callback that collects training samples and returns metrics."""

ValidateStepFn = Callable[[int], None]
"""Per-step callback that runs validation and logs results."""

_PUBLISH_POLL_INTERVAL_S = 0.5
# Far below config.timeout: a publish stuck for 5 min must not hide in the step budget.
_PUBLISH_WAIT_TIMEOUT_S = 300.0


def get_lr_cosine(
    step: int,
    total_steps: int,
    base_lr: float,
    warmup_ratio: float = 0.03,
    min_lr_ratio: float = 0.1,
) -> float:
    warmup_steps = int(total_steps * warmup_ratio)
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return base_lr * (min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


async def _await_pending_publish(session: SessionClient) -> None:
    """Block until no weight publish is outstanding for this session.

    Deferred publish modes hand the broadcast to a server-side worker that holds
    the same per-session trainer lock as forward/backward, so entering backward
    with one outstanding just blocks inside the operation instead. Waiting here
    keeps the stall attributable and bounded.
    """
    # Held outside the try: the timeout can fire before the first retrieve returns
    # (a hung session read is exactly what this bounds), and the handler must still
    # be able to say what it last saw rather than raising UnboundLocalError.
    seen = "no session snapshot retrieved"
    try:
        async with asyncio.timeout(_PUBLISH_WAIT_TIMEOUT_S) as publish_wait:
            while True:
                state = (await session.retrieve_async()).policy_state
                seen = f"applied={state.applied_weights_version} target={state.target_weights_version}"
                applied_current = int(state.applied_weights_version) >= int(state.target_weights_version)
                if not state.pending_publish and applied_current:
                    return
                await asyncio.sleep(_PUBLISH_POLL_INTERVAL_S)
    except TimeoutError as exc:
        # socket.timeout aliases TimeoutError, so only rewrite when *our* deadline
        # expired — otherwise a hung read reads as a stuck publish.
        if not publish_wait.expired():
            raise
        raise TimeoutError(f"weight publish did not land within {_PUBLISH_WAIT_TIMEOUT_S:.0f}s ({seen})") from exc


async def apply_train_step(
    session: SessionClient,
    samples: list[Sample],
    step: int,
    total_steps: int,
    config: GRPOConfig,
    run: Run,
    step_log: dict[str, Any],
    collect_time: float,
    *,
    overall_start: float,
) -> bool:
    """Apply a zero-indexed step and record its one-indexed metrics/checkpoints.

    Returns whether an optimizer step was issued.
    """
    if not samples:
        # Nothing to train on. Still log the collection metrics — they carry the
        # reason the step was empty (exhausted sampling attempts, every group
        # dropped) — and still advance the checkpoint schedule.
        log_step(run, step + 1, {"timing/step_collect_s": collect_time, **step_log})
        await checkpoint_if_due_async(
            session,
            run,
            step=step + 1,
            total_steps=total_steps,
            training_interval=config.training_checkpoint_interval,
            inference_interval=config.inference_checkpoint_interval,
        )
        return False

    if config.weight_sync_type != "WEIGHT_SYNC_TYPE_SYNCHRONOUS":
        await _await_pending_publish(session)

    t0 = time.monotonic()
    fb_result = await session.trainer.forward_backward_async(
        loss=policy_loss(config),
        samples=samples,
        timeout=config.timeout,
    )
    fb_time = time.monotonic() - t0

    current_lr = get_lr_cosine(
        step,
        total_steps,
        config.learning_rate,
        config.warmup_ratio,
        config.min_lr_ratio,
    )

    t0 = time.monotonic()
    await session.trainer.optim_step_async(
        **optim_step_params(config, current_lr),
        timeout=config.timeout,
    )
    # Publish the new weights. SYNCHRONOUS lands before this returns, so the next
    # rollout samples the policy just trained; deferred modes return immediately and
    # the next step's _await_pending_publish absorbs whatever is still outstanding.
    await session.trainer.weights_sync_async(
        weight_sync_type=config.weight_sync_type,
        timeout=config.timeout,
    )
    optim_time = time.monotonic() - t0
    step_total = collect_time + fb_time + optim_time

    logged_step = step + 1
    print(
        f"[{session.session_id[:8]}] [step {logged_step}] loss={fb_result.loss:.6f} lr={current_lr:.2e} "
        f"elapsed={time.monotonic() - overall_start:.1f}s",
        flush=True,
    )
    print(
        f"[{session.session_id[:8]}] [step {logged_step}] timing: collect={collect_time:.1f}s "
        f"fb={fb_time:.1f}s optim={optim_time:.1f}s total={step_total:.1f}s",
        flush=True,
    )
    log_step(
        run,
        logged_step,
        {
            "loss": fb_result.loss,
            "learning_rate": current_lr,
            "elapsed": time.monotonic() - overall_start,
            "timing/step_collect_s": collect_time,
            "timing/step_forward_backward_s": fb_time,
            "timing/step_optim_s": optim_time,
            "timing/step_total_s": step_total,
            **{f"fb/{k}": v for k, v in (fb_result.metrics or {}).items()},
            **step_log,
        },
    )

    await checkpoint_if_due_async(
        session,
        run,
        step=logged_step,
        total_steps=total_steps,
        training_interval=config.training_checkpoint_interval,
        inference_interval=config.inference_checkpoint_interval,
    )
    return True


def training_loop(
    total_steps: int,
    collect_samples: CollectSamplesFn,
    session: SessionClient,
    config: GRPOConfig,
    run: Run,
    validate: ValidateStepFn | None = None,
    *,
    start_step: int = 0,
) -> None:
    overall_start = time.monotonic()

    if validate is not None:
        validate(start_step)

    step = start_step
    while step < total_steps:
        t0 = time.monotonic()
        samples, step_log = collect_samples(step, session)
        collect_time = time.monotonic() - t0

        session.run(
            apply_train_step(
                session,
                samples,
                step,
                total_steps,
                config,
                run,
                step_log,
                collect_time,
                overall_start=overall_start,
            )
        )
        step += 1

        if validate is not None and step % config.validation_interval == 0:
            validate(step)
