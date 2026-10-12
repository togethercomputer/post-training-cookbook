#!/usr/bin/env python3
"""Multi-LoRA hyperparameter sweep on one shared model resource.

Together Post Training separates the *model resource* (GPU pods + base-model weights)
from the *sessions* that train on it. With `lora_enabled=True`, each attached
session is an independent LoRA adapter over the same frozen base weights — so N
sweep arms cost roughly one base model's worth of GPUs instead of N.

This template provisions one resource, trains every sweep arm on it concurrently
(each arm is its own adapter session), and stops the resource exactly once when
the sweep finishes. Compare arms by their `val/accuracy` curves in the shared
W&B group — per-arm validation already is the model selection.

Edit two things: `base` (the shared config) and `SWEEP` (the grid). Every key in
a SWEEP dict must be a field on the config you're sweeping.

Usage:
    export TOGETHER_API_KEY="..."
    export HF_TOKEN="..."
    export WANDB_API_KEY="..."          # optional but recommended for comparison
    python multi_lora_sweep_template.py
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from together.post_training import ModelResourcesClient

from post_training_cookbook.grpo.config import GRPOConfig
from post_training_cookbook.grpo.train import train_single_turn

# ---------------------------------------------------------------------------
# TODO 1: the shared base config (everything the arms have in common).
# `model` here provisions the shared resource; each arm drops it and attaches
# via model_resources_id (a config sets exactly one of the two, never both).
# ---------------------------------------------------------------------------
base = GRPOConfig(
    model="Qwen/Qwen3.8-27B",
    dataset="gsm8k",
    num_examples=2000,
    num_val_examples=200,
    batch_size=16,
    group_size=8,
    num_steps=100,
    max_model_length=1536,
    max_prompt_length=768,
    max_sample_tokens=768,
    validation_interval=25,
    wandb_project="grpo-gsm8k",
    wandb_group="gsm8k-sweep",  # one group so the arms line up in W&B
)

# ---------------------------------------------------------------------------
# TODO 2: the sweep grid. Each dict overrides fields on `base`.
# ---------------------------------------------------------------------------
SWEEP = [{"learning_rate": lr, "lora_rank": rank} for lr in (1e-5, 5e-6) for rank in (8, 16)]


def arm_name(hp: dict) -> str:
    """A stable, readable run name per arm (also its W&B run name)."""
    return "-".join(f"{k}{v}" for k, v in hp.items())


def main() -> None:
    # Provision the base-model GPUs once. The context manager stops the
    # resource exactly once on exit — even if an arm raises.
    with ModelResourcesClient.create(base_model=base.model, lora_enabled=True) as resource:
        configs = [
            replace(
                base,
                model=None,  # the resource defines the base model now
                model_resources_id=resource.model_resources_id,  # attach, don't provision
                wandb_run_name=arm_name(hp),
                **hp,
            )
            for hp in SWEEP
        ]

        # train_single_turn blocks, so each arm runs in its own thread; each
        # SessionClient drives its own event loop, so the sessions don't interfere.
        with ThreadPoolExecutor(max_workers=len(configs)) as pool:
            futures = {cfg.wandb_run_name: pool.submit(train_single_turn, cfg) for cfg in configs}

        # One arm failing must not kill the others — collect exceptions after the pool drains.
        failed = {name: exc for name, f in futures.items() if (exc := f.exception()) is not None}

    for name, exc in failed.items():
        print(f"[{name}] arm failed: {exc!r}")
    if failed:
        raise SystemExit(f"{len(failed)}/{len(configs)} sweep arms failed")
    print(f"All {len(configs)} arms finished. Compare them in the '{base.wandb_group}' W&B group.")


if __name__ == "__main__":
    main()
