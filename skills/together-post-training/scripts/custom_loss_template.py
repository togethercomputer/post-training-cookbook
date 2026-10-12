#!/usr/bin/env python3
"""Custom client-side loss training loop (DPO worked example).

Use this template when the built-in cross-entropy / GRPO losses don't fit and
the loss must be computed in Python. The pattern is two-pass:

  1. ``forward`` runs a forward-only pass and returns one log-probability per
     trained slot in ``loss_fn_outputs`` (no gradients, no optimizer change).
     ``forward_backward`` can return the same outputs via ``return_loss_fn_outputs``,
     but it still accumulates gradients that would be added to the ones step 3
     submits, so scoring passes belong on ``forward``.
  2. Treat those logprobs as autograd leaves, evaluate your loss, ``.backward()``.
  3. ``custom_forward_backward`` submits the per-slot gradients server-side.
  4. ``optim_step`` applies the update.

The cookbook wraps steps 1-3 in ``post_training_cookbook.custom_loss.compute_logprobs`` and
``compute_custom_forward_backward`` — this template uses those helpers so you only
write the loss. DPO is the worked example; swap ``dpo_loss`` for any objective
that's a function of the model's log-probabilities (the only TODO that matters).

Usage:
    export TOGETHER_API_KEY="..."
    export HF_TOKEN="..."            # for gated models / the tokenizer
    python custom_loss_template.py

This template depends on the post_training_cookbook package (for the codec + helpers).
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Any

import torch
from together.post_training import (
    AdamParams,
    ComputeConfig,
    ModelResourcesClient,
    Sample,
)

from post_training_cookbook.codecs import MessageCodec, get_codec
from post_training_cookbook.custom_loss import CustomLossFn, compute_custom_forward_backward, compute_logprobs
from post_training_cookbook.samples import build_cross_entropy_sample, extract_loss_mask, is_trainable

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class Config:
    base_model: str = "Qwen/Qwen3.8-27B"
    total_steps: int = 500
    pairs_per_step: int = 8  # preference pairs per optim step
    max_length: int = 2048

    # DPO
    beta: float = 0.1

    # AdamW + cosine LR
    lr: float = 5e-6
    warmup_ratio: float = 0.03
    min_lr_ratio: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    weight_decay: float = 0.1

    op_timeout: float = 7200.0


def cosine_lr(step: int, cfg: Config) -> float:
    warmup = int(cfg.total_steps * cfg.warmup_ratio)
    if step < warmup:
        return cfg.lr * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(cfg.total_steps - warmup, 1)
    return cfg.lr * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


# ---------------------------------------------------------------------------
# TODO: Replace with your preference data
# ---------------------------------------------------------------------------


def load_preference_pairs() -> list[tuple[list[dict], list[dict]]]:
    """Return (chosen_messages, rejected_messages) pairs.

    Each side is a full conversation ending in the assistant response being
    scored. Replace this stub with your real data source.
    """
    shared = [{"role": "user", "content": "Explain why the sky is blue."}]
    return [
        (
            shared
            + [{"role": "assistant", "content": "Rayleigh scattering makes short (blue) wavelengths scatter most."}],
            shared + [{"role": "assistant", "content": "Because it just is."}],
        )
    ] * 64


# ---------------------------------------------------------------------------
# Sample construction (SFT-style: full sequence, mask on the response)
# ---------------------------------------------------------------------------


def build_side_sample(messages: list[dict[str, Any]], codec: MessageCodec, max_length: int) -> Sample | None:
    """Build one preference side's Sample.

    Score it with `extract_loss_mask(sample)`, never with the token-space mask
    that went in: the latter is one entry longer.

    Returns:
        The Sample, or `None` when `max_length` clipping leaves no trained
        token — a prompt that fills the budget on its own.
    """
    prompt_tokens = codec.encode_messages(messages[:-1], add_generation_prompt=True)
    full_tokens = codec.encode_messages(messages, add_generation_prompt=False)
    response_len = len(full_tokens) - len(prompt_tokens)

    tokens = full_tokens[:max_length]
    loss_mask = ([0.0] * len(prompt_tokens) + [1.0] * response_len)[:max_length]
    if not is_trainable(loss_mask):
        return None

    # The weights are both the cross-entropy mask and the selector for which
    # slots `forward` returns a real logprob at: zero-weight slots read back as
    # exactly 0.0.
    return build_cross_entropy_sample(tokens, loss_mask)


def masked_sum(logprobs: list[float], mask: list[float]) -> float:
    """Sum the logprobs of the slots the mask trains.

    Both arrays are target-aligned and the same length: the mask came off the
    built sample and `forward` returns one logprob per slot.

    Raises:
        ValueError: If the two lengths disagree, which means one of them is
            still in token space.
    """
    if len(logprobs) != len(mask):
        raise ValueError(
            f"logprobs holds {len(logprobs)} slots but mask holds {len(mask)}; "
            "if the service clipped the sample, lower max_length"
        )
    return sum(logprob * weight for logprob, weight in zip(logprobs, mask))


# ---------------------------------------------------------------------------
# TODO: Replace with your loss. This one is DPO.
# ---------------------------------------------------------------------------


def make_dpo_loss(ref_sums: list[float], masks: list[list[float]], beta: float) -> CustomLossFn:
    """Return a CustomLossFn over a flat [chosen0, rejected0, chosen1, ...] batch.

    The closure receives one (S_i,) tensor of per-slot logprobs per sample and
    must return a scalar tensor. Here we reduce each sample to its masked
    logprob sum, form the chosen-vs-rejected margin relative to the reference,
    and return the DPO objective -log sigma(beta * margin).

    Args:
        ref_sums: Reference masked-logprob sums, flat-interleaved.
        masks: Each sample's target-aligned mask, from `extract_loss_mask`.
        beta: The DPO temperature on the margin.
    """

    def loss_fn(logprobs: list[torch.Tensor]) -> torch.Tensor:
        terms = []
        for k in range(len(logprobs) // 2):
            chosen_idx, rejected_idx = 2 * k, 2 * k + 1
            chosen_mask = torch.tensor(masks[chosen_idx], dtype=torch.float32)
            rejected_mask = torch.tensor(masks[rejected_idx], dtype=torch.float32)
            policy_chosen = (logprobs[chosen_idx] * chosen_mask).sum()
            policy_rejected = (logprobs[rejected_idx] * rejected_mask).sum()
            margin = (policy_chosen - ref_sums[chosen_idx]) - (policy_rejected - ref_sums[rejected_idx])
            terms.append(-torch.nn.functional.logsigmoid(beta * margin))
        return torch.stack(terms).mean()

    return loss_fn


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = Config()
    # Preference data has no reasoning traces
    codec = get_codec(cfg.base_model, hf_token=os.environ.get("HF_TOKEN"), enable_thinking=False)
    pairs = load_preference_pairs()

    # Flatten to [chosen0, rejected0, chosen1, rejected1, ...].
    flat_samples: list[Sample] = []
    skipped = 0
    for chosen_msgs, rejected_msgs in pairs:
        chosen = build_side_sample(chosen_msgs, codec, cfg.max_length)
        rejected = build_side_sample(rejected_msgs, codec, cfg.max_length)
        if chosen is None or rejected is None:
            skipped += 1  # DPO needs both halves, so drop the whole pair
            continue
        flat_samples += [chosen, rejected]
    if skipped:
        print(f"Skipped {skipped} pairs with no response tokens after max_length={cfg.max_length} truncation")
    n_pairs = len(flat_samples) // 2
    if n_pairs == 0:
        raise ValueError(
            f"Nothing to train: every pair was left with no response tokens at max_length={cfg.max_length}"
        )

    # Custom losses never sample, so provision a trainer-only resource (no generator).
    resources = ModelResourcesClient.create(
        base_model=cfg.base_model,
        api_key=os.environ.get("TOGETHER_API_KEY"),
        base_url=os.environ.get("TOGETHER_BASE_URL"),
        compute_config=ComputeConfig(num_generator_replicas=0),
    )
    session = resources.create_session()
    print(f"Session: {session.session_id}  |  {n_pairs} pairs, {cfg.total_steps} steps")

    try:
        # Snapshot the reference policy ONCE (the model before any updates).
        ref_logprobs = compute_logprobs(session, flat_samples, timeout=cfg.op_timeout)
        ref_sums = [
            masked_sum(logprobs, extract_loss_mask(sample)) for logprobs, sample in zip(ref_logprobs, flat_samples)
        ]
        print(f"Cached reference logprob sums for {len(ref_sums)} samples")

        adam = AdamParams(
            beta1=cfg.adam_beta1,
            beta2=cfg.adam_beta2,
            eps=cfg.adam_eps,
            weight_decay=cfg.weight_decay,
        )
        start = time.monotonic()
        for step in range(cfg.total_steps):
            p0 = (step * cfg.pairs_per_step) % n_pairs
            pair_idx = [(p0 + i) % n_pairs for i in range(cfg.pairs_per_step)]
            flat_idx = [j for p in pair_idx for j in (2 * p, 2 * p + 1)]

            batch = [flat_samples[i] for i in flat_idx]
            batch_masks = [extract_loss_mask(flat_samples[i]) for i in flat_idx]
            batch_ref = [ref_sums[i] for i in flat_idx]

            loss_fn = make_dpo_loss(batch_ref, batch_masks, cfg.beta)
            result = compute_custom_forward_backward(session, batch, loss_fn, timeout=cfg.op_timeout)

            lr = cosine_lr(step, cfg)
            adam["learning_rate"] = lr
            session.trainer.optim_step(
                adam_params=adam,
                timeout=cfg.op_timeout,
            )
            print(f"[{step}] loss={result.loss:.4f} lr={lr:.2e} elapsed={time.monotonic() - start:.1f}s")

        ckpt = session.create_inference_checkpoint()
        print(f"\nDone! Inference checkpoint {ckpt.id}, registered as {ckpt.registration}")
    finally:
        resources.stop()
        print("Session stopped.")


if __name__ == "__main__":
    main()
