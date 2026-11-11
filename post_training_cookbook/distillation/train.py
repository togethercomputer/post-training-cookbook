"""Off-policy distillation training: SFT on teacher rows, scored on the generator as it trains.

The student trains with cross-entropy on chat rows from :func:`post_training_cookbook.distillation.data.sft_rows`,
masking the prompt and training the assistant turn (``TRAIN_ON_LAST``). The resource keeps a generator
replica, so each eval set is scored at step 0 (the untouched base model), every ``eval_interval`` steps
and at the last step, on the same session, with the same sampling. No separate endpoint is needed.
"""

from __future__ import annotations

import itertools
import json
import logging
import math
import os
import random
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from together.post_training import AdamParams, LossConfig, Sample

from post_training_cookbook.codecs import MessageCodec, get_codec
from post_training_cookbook.config import lora_config_from, session_metadata_from
from post_training_cookbook.distillation.config import DistillConfig
from post_training_cookbook.distillation.evaluate import EvalSet, student_outputs
from post_training_cookbook.grpo.loop import get_lr_cosine
from post_training_cookbook.multi_lora import resolve_base_model, training_session
from post_training_cookbook.run_artifacts import Run, checkpoint_if_due, latest_checkpoint_ref, log_step, track_run
from post_training_cookbook.samples import trained_slots
from post_training_cookbook.sft.train import TRAIN_ON_LAST, build_conversation_samples

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DistillResult:
    run_dir: Path
    # The last inference checkpoint saved (its registration holds the deployable adapter or model).
    inference_checkpoint: str | None
    # The most recent metrics for every eval set, keyed by set name.
    eval_metrics: dict[str, dict[str, float]]


def render_rows(rows: Sequence[dict[str, Any]], codec: MessageCodec, max_length: int) -> tuple[list[Sample], int]:
    """Render chat rows into cross-entropy samples; return the samples and how many rows were unrenderable."""
    samples: list[Sample] = []
    dropped = 0
    for row in rows:
        try:
            built = build_conversation_samples(
                row["messages"], codec=codec, train_on_what=TRAIN_ON_LAST, max_length=max_length
            )
        except ValueError:
            built = []
        if built:
            samples.extend(built)
        else:
            dropped += 1
    return samples, dropped


def _batches(n: int, batch_size: int, seed: int) -> Iterator[list[int]]:
    """Index batches forever, reshuffling each epoch; a trailing partial batch is dropped."""
    if n < batch_size:
        raise ValueError(f"{n} samples cannot fill a batch of {batch_size}")
    for epoch in itertools.count():
        order = list(range(n))
        random.Random(seed + epoch).shuffle(order)
        for start in range(0, n - batch_size + 1, batch_size):
            yield order[start : start + batch_size]


def _adam_params(config: DistillConfig, learning_rate: float) -> AdamParams:
    """Adam parameters for one step; unset betas and eps are left to the server's defaults."""
    params = AdamParams(learning_rate=learning_rate, weight_decay=config.weight_decay)
    for key, value in (("beta1", config.adam_beta1), ("beta2", config.adam_beta2), ("eps", config.adam_eps)):
        if value is not None:
            params[key] = value
    return params


def _codec(config: DistillConfig) -> MessageCodec:
    return get_codec(
        resolve_base_model(config),
        hf_token=os.environ.get("HF_TOKEN"),
        enable_thinking=config.enable_thinking,
        preserve_thinking=config.preserve_thinking,
        thinking_effort=config.thinking_effort,
    )


def _session(config: DistillConfig):
    return training_session(
        model=config.model,
        model_resources_id=config.model_resources_id,
        display_name=config.wandb_run_name,
        metadata=session_metadata_from(config),
        resume_from_checkpoint_id=config.resume_from_checkpoint_id,
        load_optimizer=config.load_optimizer,
        lora_config=lora_config_from(config),
        lora_enabled=config.use_lora,
        num_generator_replicas=config.num_generator_replicas,
    )


def _require_generator(config: DistillConfig, eval_sets: Sequence[EvalSet]) -> None:
    if eval_sets and config.model_resources_id is None and config.num_generator_replicas < 1:
        raise ValueError("eval sets are sampled on the generator; set num_generator_replicas >= 1")


def run_evals(
    session: Any,
    codec: MessageCodec,
    config: DistillConfig,
    run: Run,
    step: int,
    eval_sets: Sequence[EvalSet],
) -> dict[str, dict[str, float]]:
    """Score each eval set on the session's generator, log ``eval/<set>/<metric>``, and save the outputs."""
    results: dict[str, dict[str, float]] = {}
    for eval_set in eval_sets:
        t0 = time.monotonic()
        texts, truncated = student_outputs(
            session,
            codec,
            eval_set.items,
            max_tokens=config.eval_max_tokens,
            temperature=config.eval_temperature,
            timeout=config.timeout,
        )
        metrics = dict(eval_set.metrics(eval_set.items, texts))
        metrics["truncated_rate"] = sum(truncated) / max(len(truncated), 1)
        results[eval_set.name] = metrics
        log_step(run, step, {f"eval/{eval_set.name}/{k}": v for k, v in metrics.items()})
        dump = [
            {"id": item["id"], "output": text, "truncated": trunc}
            for item, text, trunc in zip(eval_set.items, texts, truncated, strict=True)
        ]
        (run.dir / f"eval_{eval_set.name}_step{step:04d}.json").write_text(json.dumps(dump, ensure_ascii=False))
        shown = " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items())
        print(f"[step {step}] eval {eval_set.name}: {shown} ({time.monotonic() - t0:.0f}s)", flush=True)
    return results


def train(
    config: DistillConfig,
    rows: Sequence[dict[str, Any]],
    eval_sets: Sequence[EvalSet] = (),
) -> DistillResult:
    """Fine-tune the student on teacher ``rows``, scoring ``eval_sets`` before, during and after training.

    Rendering happens before the resource is provisioned, so an unrenderable dataset fails before
    anything bills. The resource and session are stopped on exit, even if a step raises.
    """
    load_dotenv(override=True)
    eval_sets = list(eval_sets)
    _require_generator(config, eval_sets)
    codec = _codec(config)
    samples, dropped = render_rows(rows, codec, config.max_length)
    if not samples:
        raise ValueError(
            f"none of {len(rows)} rows rendered into a trainable sample; check enable_thinking against the "
            "teacher data and max_length"
        )
    if dropped:
        logger.warning("Dropped %d of %d rows the chat template could not render trainably", dropped, len(rows))
    steps_per_epoch = len(samples) // config.batch_size
    if steps_per_epoch == 0:
        raise ValueError(f"{len(samples)} samples cannot fill a batch of {config.batch_size}")
    total_steps = max(1, int(steps_per_epoch * config.num_epochs))
    if config.max_steps is not None:
        total_steps = min(total_steps, config.max_steps)
    print(
        f"{len(rows)} rows -> {len(samples)} samples ({dropped} dropped); {total_steps} steps "
        f"({steps_per_epoch}/epoch, batch {config.batch_size}); eval sets: {[e.name for e in eval_sets]}",
        flush=True,
    )

    with (
        _session(config) as session,
        track_run(config, kind="distill", session_id=session.session_id, total_steps=total_steps) as run,
    ):
        latest = run_evals(session, codec, config, run, 0, eval_sets) if eval_sets else {}
        overall_start = time.monotonic()
        for step, indices in zip(range(total_steps), _batches(len(samples), config.batch_size, config.seed)):
            batch = [samples[i] for i in indices]
            lr = get_lr_cosine(step, total_steps, config.learning_rate, config.warmup_ratio, config.min_lr_ratio)
            t0 = time.monotonic()
            result = session.trainer.forward_backward(
                samples=batch, loss=LossConfig(type="LOSS_TYPE_CROSS_ENTROPY"), timeout=config.timeout
            )
            session.trainer.optim_step(adam_params=_adam_params(config, lr), timeout=config.timeout)
            slots = trained_slots(batch)
            step_num = step + 1
            loss = result.loss / slots if slots else math.nan
            log_step(
                run,
                step_num,
                {
                    "loss": loss,
                    "loss_sum": result.loss,
                    "trained_tokens": slots,
                    "learning_rate": lr,
                    "elapsed_s": time.monotonic() - overall_start,
                    "timing/step_s": time.monotonic() - t0,
                },
            )
            if step_num == 1 or step_num % 10 == 0:
                print(f"[step {step_num}/{total_steps}] loss={loss:.4f} lr={lr:.2e}", flush=True)

            last = step_num == total_steps
            if eval_sets and (step_num % config.eval_interval == 0 or last):
                session.trainer.weights_sync(timeout=config.timeout)
                due = [e for e in eval_sets if e.every_eval or last]
                latest |= run_evals(session, codec, config, run, step_num, due)

            checkpoint_if_due(
                session,
                run,
                step=step_num,
                total_steps=total_steps,
                training_interval=config.training_checkpoint_interval,
                inference_interval=config.inference_checkpoint_interval,
            )
        inference_checkpoint = latest_checkpoint_ref(run, kind="inference")
        run_dir = run.dir
    return DistillResult(run_dir=run_dir, inference_checkpoint=inference_checkpoint, eval_metrics=latest)


def evaluate_student(config: DistillConfig, eval_sets: Sequence[EvalSet]) -> dict[str, dict[str, float]]:
    """Score the untrained student on ``eval_sets`` (a fresh LoRA adapter is the base model), then release the GPUs."""
    load_dotenv(override=True)
    eval_sets = list(eval_sets)
    if not eval_sets:
        raise ValueError("evaluate_student needs at least one eval set")
    _require_generator(config, eval_sets)
    codec = _codec(config)
    with (
        _session(config) as session,
        track_run(config, kind="distill-eval", session_id=session.session_id, total_steps=0) as run,
    ):
        return run_evals(session, codec, config, run, 0, eval_sets)
