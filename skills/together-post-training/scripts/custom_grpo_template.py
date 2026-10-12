#!/usr/bin/env python3
"""Custom GRPO training loop template using the Together Post Training SDK directly.

This is a complete, runnable template. Customize the sections marked with
TODO to plug in your own data, tokenizer, and reward function.

Features:
  - Pre-tokenized prompt input (bring your own data pipeline)
  - Inline reward function (replace with your domain logic)
  - Cosine LR decay with linear warmup
  - Periodic checkpointing (training + inference)
  - Zero-variance group skipping
  - Graceful shutdown on KeyboardInterrupt

Usage:
    export TOGETHER_API_KEY="..."
    python custom_grpo_template.py

Requires: together SDK (pip install together)
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
from together.post_training import (
    AdamParams,
    EncodedTextChunk,
    GrpoLossParams,
    LossConfig,
    ModelInput,
    ModelInputChunk,
    ModelResourcesClient,
    Sample,
    SamplingParams,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class Config:
    """All hyperparameters in one place."""

    base_model: str = "Qwen/Qwen3.8-27B"

    # Training budget
    total_steps: int = 500
    batch_size: int = 8
    group_size: int = 8

    # Sampling
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512

    # Absolute ratio bounds (not offsets from 1.0). None → omit.
    # Server defaults are 0.8 / 1.2 for PPO/GRPO, 0.0/4.0 for CISPO.
    clip_low_threshold: float | None = None
    clip_high_threshold: float | None = None
    agg_type: str = "GRPO_LOSS_AGGREGATION_TYPE_FIXED_HORIZON"

    # AdamW optimizer
    lr: float = 1e-5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    weight_decay: float = 0.1

    # LR schedule
    warmup_ratio: float = 0.03
    min_lr_ratio: float = 0.1

    # Checkpointing
    checkpoint_every: int = 50

    # Timeouts (seconds)
    session_timeout: float = 3600.0
    op_timeout: float = 300.0


# ---------------------------------------------------------------------------
# Cosine LR schedule with linear warmup
# ---------------------------------------------------------------------------


def cosine_lr(step: int, cfg: Config) -> float:
    warmup_steps = int(cfg.total_steps * cfg.warmup_ratio)
    if step < warmup_steps:
        return cfg.lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(cfg.total_steps - warmup_steps, 1)
    return cfg.lr * (cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))


# ---------------------------------------------------------------------------
# TODO: Replace with your data pipeline
# ---------------------------------------------------------------------------


def load_prompts() -> list[list[int]]:
    """Return pre-tokenized prompts as lists of token IDs.

    Replace this stub with your real data source, e.g.:
        import json
        with open("prompts.jsonl") as f:
            return [json.loads(line)["tokens"] for line in f]
    """
    return [[101, 102, 103, 104, 105]] * 500  # placeholder


def decode(token_ids: list[int]) -> str:
    """Decode token IDs to text.

    Replace with your tokenizer, e.g.:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.8-27B")
        return tok.decode(token_ids, skip_special_tokens=True)
    """
    return " ".join(str(t) for t in token_ids)


# ---------------------------------------------------------------------------
# TODO: Replace with your reward logic
# ---------------------------------------------------------------------------


def reward_fn(response_text: str, prompt_tokens: list[int]) -> float:
    """Score a single completion. Higher = better.

    Replace the body with your domain-specific scoring. Examples:
      - Exact match: return 1.0 if extract_answer(text) == target else 0.0
      - Code execution: run the code and check stdout
      - LLM-as-judge: call an evaluator model
    """
    text = response_text.strip()
    if not text:
        return 0.0
    if "\\boxed{" in text:
        return 1.0
    if any(ch.isdigit() for ch in text):
        return 0.5
    return 0.0


# ---------------------------------------------------------------------------
# Sample construction
# ---------------------------------------------------------------------------


def build_grpo_sample_by_hand(
    prompt_tokens: list[int],
    response_tokens: list[int],
    response_logprobs: list[float],
    advantage: float,
) -> Sample:
    """Build one GRPO training sample, target-aligned.

    Slot `i` scores the prediction made at input position `i` against
    `target_tokens[i]`, so N tokens train N-1 slots: `model_input` drops the last
    token (it is a target that is never an input) and every loss array drops its
    first entry (slot 0 already predicts `tokens[1]`).

    If you depend on the cookbook, call `post_training_cookbook.grpo.samples.build_policy_sample`
    instead -- it owns this trim and validates it. This template stays on the SDK
    alone, so it does the arithmetic here.
    """
    prompt_len, response_len = len(prompt_tokens), len(response_tokens)
    tokens = prompt_tokens + response_tokens

    # Token space: one entry per token of `tokens`. The prompt
    # boundary shows up only as the run of zeros at the head of `loss_mask`.
    loss_mask = [0.0] * prompt_len + [1.0] * response_len
    advantages = [0.0] * prompt_len + [advantage] * response_len
    logprobs = [0.0] * prompt_len + [float(logprob) for logprob in response_logprobs]

    # Checked on the assembled arrays, so editing the padding above cannot slip
    # a length past this. The mask check covers a run with no trainable slot,
    # including a one-token run with no prompt.
    if len(logprobs) != len(tokens):
        raise ValueError(f"logprobs holds {len(logprobs)} entries but the run holds {len(tokens)} tokens")
    if not any(loss_mask[1:]):
        raise ValueError(f"a {len(tokens)}-token run trains no slot; skip the completion instead of building it")

    # One trim, applied to every array at once. Trimming the targets and
    # forgetting a sibling is the classic failure; `references/sdk-reference.md`
    # ("If you build by hand") tabulates what each one costs.
    loss_fn_inputs = {
        "target_tokens": {"data": tokens[1:], "dtype": "int64"},
        "weights": {"data": loss_mask[1:], "dtype": "float32"},
        "advantages": {"data": advantages[1:], "dtype": "float32"},
        "logprobs": {"data": logprobs[1:], "dtype": "float32"},
    }
    model_input = ModelInput(chunks=[ModelInputChunk(encoded_text=EncodedTextChunk(tokens=tokens[:-1]))])

    return Sample(model_input=model_input, loss_fn_inputs=loss_fn_inputs)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = Config()
    prompts = load_prompts()

    resources = ModelResourcesClient.create(
        base_model=cfg.base_model,
        api_key=os.environ.get("TOGETHER_API_KEY"),
        base_url=os.environ.get("TOGETHER_BASE_URL"),
        timeout=cfg.session_timeout,
    )
    session = resources.create_session()
    print(f"Session: {session.session_id}  |  {cfg.total_steps} steps")

    clips = {
        k: v
        for k, v in (
            ("clip_low_threshold", cfg.clip_low_threshold),
            ("clip_high_threshold", cfg.clip_high_threshold),
        )
        if v is not None
    }
    loss = LossConfig(
        type="LOSS_TYPE_GRPO",
        grpo_params=GrpoLossParams(agg_type=cfg.agg_type, **clips),
    )
    adam = AdamParams(
        beta1=cfg.adam_beta1,
        beta2=cfg.adam_beta2,
        eps=cfg.adam_eps,
        weight_decay=cfg.weight_decay,
    )

    try:
        for step in range(cfg.total_steps):
            # Select batch
            start = (step * cfg.batch_size) % len(prompts)
            batch = prompts[start : start + cfg.batch_size]
            if len(batch) < cfg.batch_size:
                batch += prompts[: cfg.batch_size - len(batch)]

            # Sample completions
            model_inputs = [
                ModelInput(chunks=[ModelInputChunk(encoded_text=EncodedTextChunk(tokens=p))]) for p in batch
            ]
            results = session.generator.sample_batch(
                model_inputs,
                num_samples=cfg.group_size,
                sampling_params=SamplingParams(
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    max_tokens=cfg.max_tokens,
                ),
                timeout=cfg.op_timeout,
            )

            # Score and build samples
            samples: list[Sample] = []
            all_rewards: list[float] = []
            skipped = 0
            degenerate = 0

            for prompt_tokens, result in zip(batch, results):
                rewards, tokens_list, logprobs_list = [], [], []
                for seq in result.sequences:
                    response_tokens = [int(t) for t in (seq.tokens or [])]
                    response_logprobs = [float(v) for v in (seq.logprobs or [])]
                    rewards.append(reward_fn(decode(response_tokens), prompt_tokens))
                    tokens_list.append(response_tokens)
                    logprobs_list.append(response_logprobs)

                all_rewards.extend(rewards)
                arr = np.array(rewards)
                if np.std(arr) < 1e-8:
                    skipped += 1
                    continue

                mean_reward = float(np.mean(arr))
                for response_tokens, response_logprobs, reward in zip(tokens_list, logprobs_list, rewards):
                    # The baseline above already counted this reward, and that
                    # reward is real signal -- only the payload is dropped. A
                    # tokens/logprobs length disagreement is not routine, so it
                    # is left to raise in the builder.
                    if not response_tokens:
                        degenerate += 1
                        continue
                    samples.append(
                        build_grpo_sample_by_hand(
                            prompt_tokens, response_tokens, response_logprobs, reward - mean_reward
                        )
                    )

            if len(samples) < cfg.group_size:
                print(f"[{step}] skipped (only {len(samples)} samples)")
                continue

            # Forward-backward + optim step
            fb = session.trainer.forward_backward(samples=samples, loss=loss, timeout=cfg.op_timeout)
            lr = cosine_lr(step, cfg)
            adam["learning_rate"] = lr
            session.trainer.optim_step(
                adam_params=adam,
                timeout=cfg.op_timeout,
            )
            session.trainer.weights_sync(
                weight_sync_type="WEIGHT_SYNC_TYPE_SYNCHRONOUS",
                timeout=cfg.op_timeout,
            )

            mr = float(np.mean(all_rewards)) if all_rewards else 0.0
            print(
                f"[{step}] loss={fb.loss:.4f} reward={mr:.3f} samples={len(samples)} "
                f"skipped={skipped} degenerate={degenerate} lr={lr:.2e}"
            )

            # Checkpoint
            if (step + 1) % cfg.checkpoint_every == 0:
                tc = session.create_training_checkpoint()
                ic = session.create_inference_checkpoint()
                print(f"  checkpoint: train={tc.id} infer={ic.id}")

        # Final checkpoint
        ic = session.create_inference_checkpoint()
        print(f"\nDone! Inference checkpoint {ic.id}, registered as {ic.registration}")

    except KeyboardInterrupt:
        print("\nInterrupted — saving emergency checkpoint...")
        session.create_training_checkpoint()
    finally:
        resources.stop()
        print("Session stopped.")


if __name__ == "__main__":
    main()
