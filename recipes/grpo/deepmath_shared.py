"""GRPO on full DeepMath, attached to a model resource someone else provisioned.

This script never creates or deletes the resource — it is a pure tenant. The
session it makes is its own LoRA adapter; the GPUs belong to whoever stood up
the resource, and they keep running after this script exits.

Hyperparameters mirror the shipped `deepmath.py` recipe (Qwen3.6-35B-A3B, full
DeepMath, thinking-on, 16k/15k budget, LR 1e-4, LoRA defaults left to the
platform) — the only difference is this attaches instead of provisioning.

List the resource IDs you can attach to with ``python scripts/multi_lora/fetch.py``,
and free their GPUs when done with ``python scripts/multi_lora/clean_up.py``.
"""

import os

from post_training_cookbook.grpo.config import GRPOConfig
from post_training_cookbook.grpo.train import train_single_turn

# A teammate provisioned this and shared the ID (Slack, a ticket, env var).
RESOURCE_ID = os.environ["MODEL_RESOURCES_ID"]

config = GRPOConfig(
    model_resources_id=RESOURCE_ID,  # attach instead of provisioning; the resource defines the base model
    dataset="deepmath",
    num_examples=7000,
    num_val_examples=200,
    batch_size=16,
    group_size=8,
    num_steps=40,
    learning_rate=1e-4,
    max_model_length=16384,
    max_prompt_length=1024,
    max_sample_tokens=15360,
    max_sampling_attempts=4,
    validation_interval=5,
    wandb_project="grpo-deepmath",
)

if __name__ == "__main__":
    train_single_turn(config)  # the resource is left running for others
