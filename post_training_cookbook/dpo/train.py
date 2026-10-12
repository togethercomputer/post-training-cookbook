"""DPO training orchestration using the two-pass custom-loss helper."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import datasets
import torch
from dotenv import load_dotenv
from together.post_training import AdamParams, Sample, SessionClient

from post_training_cookbook.codecs import MessageCodec, get_codec
from post_training_cookbook.config import lora_config_from, session_metadata_from
from post_training_cookbook.custom_loss import compute_custom_forward_backward, compute_logprobs
from post_training_cookbook.dpo.config import DPOConfig
from post_training_cookbook.dpo.loss import make_dpo_loss, masked_logp_sum
from post_training_cookbook.dpo.samples import build_preference_pair
from post_training_cookbook.grpo.loop import get_lr_cosine
from post_training_cookbook.multi_lora import training_session
from post_training_cookbook.run_artifacts import checkpoint_if_due, log_step, track_run
from post_training_cookbook.samples import extract_loss_mask
from post_training_cookbook.types import Message

# Maps a dataset row to (chosen_messages, rejected_messages), or None to skip the row.
RowToPair = Callable[[Mapping[str, Any]], tuple[list[Message], list[Message]] | None]


def _load_dataset(config: DPOConfig):
    return _load_split(
        config,
        split=config.dataset_split,
        subsample=config.dataset_subsample,
    )


def _load_val_dataset(config: DPOConfig):
    if config.val_interval <= 0 or config.val_subsample == 0:
        return None
    return _load_split(
        config,
        split=config.val_split,
        subsample=config.val_subsample,
    )


def _load_split(config: DPOConfig, *, split: str, subsample: int | None):
    if config.dataset_config:
        dataset = datasets.load_dataset(config.dataset_name, config.dataset_config)
    else:
        dataset = datasets.load_dataset(config.dataset_name)
    dataset = dataset[split].shuffle(seed=0)
    if subsample is not None:
        n = min(subsample, len(dataset))
        dataset = dataset.select(range(n))
    return dataset


def helpsteer3_pair(row: Mapping[str, Any]) -> tuple[list[Message], list[Message]] | None:
    """Map a HelpSteer3 row to (chosen_messages, rejected_messages).

    ``overall_preference`` is in [-3, 3]; negative means response1 wins, positive
    means response2 wins, zero is a tie (skipped).
    """
    pref = row["overall_preference"]
    if pref == 0:
        return None
    chosen_text, rejected_text = (
        (row["response1"], row["response2"]) if pref < 0 else (row["response2"], row["response1"])
    )
    context = list(row["context"])
    chosen = [*context, {"role": "assistant", "content": chosen_text}]
    rejected = [*context, {"role": "assistant", "content": rejected_text}]
    return chosen, rejected


def _build_flat_samples(
    dataset: Iterable[Mapping[str, Any]],
    to_pair: RowToPair,
    codec: MessageCodec,
    max_length: int,
) -> list[Sample]:
    """Build the interleaved [c0, r0, c1, r1, …] sample list.

    Pairs are dropped when ``to_pair`` skips the row (e.g. a HelpSteer3 tie) or
    when either side has no response tokens left after truncating the
    (prompt + response) sequence to ``max_length`` — such a pair would
    contribute a zero-gradient ``-log σ(0)`` term to the DPO objective.
    """
    flat_samples: list[Sample] = []
    skipped_rows = 0
    skipped_truncated = 0
    for row in dataset:
        pair = to_pair(row)
        if pair is None:
            skipped_rows += 1
            continue
        chosen_msgs, rejected_msgs = pair
        built = build_preference_pair(
            chosen_messages=chosen_msgs,
            rejected_messages=rejected_msgs,
            codec=codec,
            max_length=max_length,
        )
        if built is None:
            skipped_truncated += 1
            continue
        flat_samples.extend(built)
    if skipped_rows:
        print(f"Skipped {skipped_rows} rows to_pair rejected")
    if skipped_truncated:
        print(f"Skipped {skipped_truncated} pairs with no response tokens after max_length={max_length} truncation")
    return flat_samples


def _compute_masked_logp_sums(
    session: SessionClient,
    samples: list[Sample],
    *,
    chunk_size: int,
    timeout: float,
    progress_label: str | None = None,
) -> list[float]:
    """Score every sample with a forward-only pass and return the mask-weighted logp sum."""
    sums: list[float] = []
    n = len(samples)
    started = time.monotonic()
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk_logprobs = compute_logprobs(session, samples[start:end], timeout=timeout)
        for data, sample in zip(chunk_logprobs, samples[start:end], strict=True):
            # float64 explicitly: the default dtype is float32, which both drifts
            # from a plain Python sum and leaves the precision hostage to
            # torch.set_default_dtype.
            sums.append(float(masked_logp_sum(torch.tensor(data, dtype=torch.float64), extract_loss_mask(sample))))
        if progress_label:
            print(f"[{progress_label}] {end}/{n} samples ({time.monotonic() - started:.0f}s elapsed)")
    return sums


def _eval_preference_accuracy(
    session: SessionClient,
    samples: list[Sample],
    ref_sums: list[float],
    *,
    chunk_size: int,
    timeout: float,
) -> dict[str, float]:
    """DPO 'implicit reward accuracy' on held-out preference pairs.

    A pair counts as correct when the trained policy has shifted relative to
    the reference such that ``(logπ_θ(y_w) - logπ_ref(y_w)) > (logπ_θ(y_l) - logπ_ref(y_l))``
    — i.e. the policy now prefers the chosen response over rejected. Less
    noisy than per-batch train loss.
    """
    trained_sums = _compute_masked_logp_sums(session, samples, chunk_size=chunk_size, timeout=timeout)
    if len(trained_sums) != len(ref_sums):
        raise RuntimeError(f"eval mismatch: {len(trained_sums)} trained sums vs {len(ref_sums)} ref sums")
    n_pairs = len(trained_sums) // 2
    correct = 0
    chosen_deltas: list[float] = []
    rejected_deltas: list[float] = []
    for k in range(n_pairs):
        chosen_idx, rejected_idx = 2 * k, 2 * k + 1
        chosen_delta = trained_sums[chosen_idx] - ref_sums[chosen_idx]
        rejected_delta = trained_sums[rejected_idx] - ref_sums[rejected_idx]
        chosen_deltas.append(chosen_delta)
        rejected_deltas.append(rejected_delta)
        if chosen_delta > rejected_delta:
            correct += 1
    return {
        "preference_accuracy": correct / n_pairs,
        "mean_chosen_delta": sum(chosen_deltas) / n_pairs,
        "mean_rejected_delta": sum(rejected_deltas) / n_pairs,
        "mean_margin": sum(c - r for c, r in zip(chosen_deltas, rejected_deltas)) / n_pairs,
    }


def train(config: DPOConfig, *, to_pair: RowToPair = helpsteer3_pair) -> None:
    """Run DPO on the preference pairs ``to_pair`` extracts from each dataset row."""
    load_dotenv(override=True)
    codec = get_codec(
        config.model,
        hf_token=os.environ.get("HF_TOKEN"),
        enable_thinking=config.enable_thinking,
    )

    dataset = _load_dataset(config)
    print(f"Loaded {len(dataset)} preference pairs")
    val_dataset = _load_val_dataset(config)
    if val_dataset is not None:
        print(f"Loaded {len(val_dataset)} validation preference pairs")

    tokenize_start = time.monotonic()
    flat_samples = _build_flat_samples(dataset, to_pair, codec, config.max_length)
    n_samples = len(flat_samples)
    n_pairs = n_samples // 2

    val_samples: list[Sample] = []
    if val_dataset is not None:
        val_samples = _build_flat_samples(val_dataset, to_pair, codec, config.max_length)
    print(f"Built {n_pairs} preference pairs ({n_samples} flat samples) in {time.monotonic() - tokenize_start:.1f}s")
    if n_pairs == 0:
        raise ValueError(
            f"Nothing to train: every pair in split {config.dataset_split!r} was rejected by to_pair or "
            f"left with no response tokens at max_length={config.max_length}"
        )

    with (
        training_session(
            model=config.model,
            display_name=config.wandb_run_name,
            metadata=session_metadata_from(config),
            resume_from_checkpoint_id=config.resume_from_checkpoint_id,
            load_optimizer=config.load_optimizer,
            lora_enabled=config.use_lora,
            lora_config=lora_config_from(config),
            num_generator_replicas=0,
        ) as session,
        track_run(config, kind="dpo", session_id=session.session_id, total_steps=config.max_steps) as run,
    ):
        overall_start = time.monotonic()
        pair_batch = config.batch_size
        ref_sums = _compute_masked_logp_sums(
            session,
            flat_samples,
            chunk_size=config.precompute_ref_batch_size,
            timeout=config.timeout,
            progress_label="ref-precompute",
        )
        print(
            f"Reference logp sums cached: {n_pairs} pairs "
            f"(mean={sum(ref_sums) / len(ref_sums):.3f}) "
            f"in {time.monotonic() - overall_start:.1f}s"
        )

        val_ref_sums: list[float] = []
        if val_dataset is not None:
            val_ref_sums = _compute_masked_logp_sums(
                session,
                val_samples,
                chunk_size=config.precompute_ref_batch_size,
                timeout=config.timeout,
                progress_label="val-ref-precompute",
            )
            print(f"Val ref sums cached: {len(val_ref_sums) // 2} pairs")

        def maybe_eval(step: int) -> None:
            if not val_samples or config.val_interval <= 0:
                return
            if step % config.val_interval != 0:
                return
            metrics = _eval_preference_accuracy(
                session,
                val_samples,
                val_ref_sums,
                chunk_size=config.precompute_ref_batch_size,
                timeout=config.timeout,
            )
            print(
                f"[step {step}] val preference_accuracy={metrics['preference_accuracy']:.3f} "
                f"margin={metrics['mean_margin']:+.3f} "
                f"chosen_Δ={metrics['mean_chosen_delta']:+.3f} "
                f"rejected_Δ={metrics['mean_rejected_delta']:+.3f}"
            )
            log_step(run, step, {f"val/{k}": v for k, v in metrics.items()})

        for step in range(config.max_steps):
            step_num = step + 1
            maybe_eval(step)
            pair_start = (step * pair_batch) % n_pairs
            pair_indices = [(pair_start + i) % n_pairs for i in range(pair_batch)]
            flat_indices = [idx for p in pair_indices for idx in (2 * p, 2 * p + 1)]

            batch_samples = [flat_samples[i] for i in flat_indices]
            batch_ref_sums = [ref_sums[i] for i in flat_indices]

            current_lr = get_lr_cosine(
                step,
                config.max_steps,
                config.learning_rate,
                warmup_ratio=config.warmup_ratio,
                min_lr_ratio=config.min_lr_ratio,
            )

            loss_fn = make_dpo_loss(
                batch_ref_sums,
                [extract_loss_mask(sample) for sample in batch_samples],
                beta=config.beta,
                label_smoothing=config.label_smoothing,
            )
            result = compute_custom_forward_backward(
                session,
                batch_samples,
                loss_fn,
                timeout=config.timeout,
            )

            session.trainer.optim_step(
                adam_params=AdamParams(
                    beta1=config.adam_beta1,
                    beta2=config.adam_beta2,
                    eps=config.adam_eps,
                    weight_decay=config.weight_decay,
                    learning_rate=current_lr,
                ),
                timeout=config.timeout,
            )

            elapsed = time.monotonic() - overall_start
            log_step(
                run,
                step_num,
                {
                    "loss": result.loss,
                    "learning_rate": current_lr,
                    "pairs": pair_batch,
                    "elapsed_s": elapsed,
                    **{f"cfb/{k}": v for k, v in result.metrics.items()},
                },
            )
            print(
                f"[step {step_num}] loss={result.loss:.4f} lr={current_lr:.2e} "
                f"pairs={pair_batch} elapsed={elapsed:.1f}s"
            )
            checkpoint_if_due(
                session,
                run,
                step=step_num,
                total_steps=config.max_steps,
                training_interval=config.training_checkpoint_interval,
                inference_interval=config.inference_checkpoint_interval,
            )

        # Final eval after the last optim_step.
        maybe_eval(config.max_steps)
    print(f"DPO training completed in {time.monotonic() - overall_start:.1f}s")
