"""SFT, then GRPO from those weights with a fresh optimizer.

SFT saves a training checkpoint; GRPO loads it with ``load_optimizer=False`` so
the adapter continues but Adam moments and the step counter do not. The two
phases use separate sessions (SFT is trainer-only; GRPO needs a generator).

GSM8K is the vehicle: SFT trains on the reference solutions, posed in GRPO's
question format, so the model learns the ``####`` answer format GRPO then
scores. The two W&B runs share a group and a short handoff id in their names.

    uv run python recipes/sft_then_grpo.py
"""

from dataclasses import replace
from uuid import uuid4

from post_training_cookbook.datasets.registry import DATASETS
from post_training_cookbook.grpo.config import GRPOConfig
from post_training_cookbook.grpo.train import train_single_turn
from post_training_cookbook.sft.config import SFTConfig
from post_training_cookbook.sft.train import train as sft_train

MODEL = "Qwen/Qwen3.8-27B"
HANDOFF_ID = uuid4().hex[:8]
WANDB_PROJECT = "sft-then-grpo"
WANDB_GROUP = f"sft-then-grpo-{HANDOFF_ID}"


def gsm8k_messages(row):
    return [
        {"role": "user", "content": DATASETS["gsm8k"].format_question(row)},
        {"role": "assistant", "content": row["answer"]},
    ]


sft_config = SFTConfig(
    model=MODEL,
    dataset_name="openai/gsm8k",
    dataset_config="main",
    dataset_split="train",
    batch_size=8,
    learning_rate=1e-4,
    max_length=1536,
    max_steps=20,
    enable_thinking=False,
    lora_rank=16,
    lora_alpha=16,
    training_checkpoint_interval=20,
    wandb_project=WANDB_PROJECT,
    wandb_group=WANDB_GROUP,
    wandb_run_name=f"sft-handoff-{HANDOFF_ID}",
)

grpo_config = GRPOConfig(
    model=MODEL,
    dataset="gsm8k",
    enable_thinking=False,
    num_examples=4000,
    num_val_examples=200,
    batch_size=16,
    group_size=8,
    num_steps=20,
    learning_rate=1e-5,
    max_model_length=1536,
    max_prompt_length=768,
    max_sample_tokens=768,
    lora_rank=16,
    lora_alpha=16,
    wandb_project=WANDB_PROJECT,
    wandb_group=WANDB_GROUP,
    wandb_run_name=f"grpo-resume-{HANDOFF_ID}",
)

if __name__ == "__main__":
    checkpoint_id = sft_train(sft_config, to_messages=gsm8k_messages)
    if checkpoint_id is None:
        raise SystemExit("SFT did not write a training checkpoint; set training_checkpoint_interval")
    train_single_turn(replace(grpo_config, resume_from_checkpoint_id=checkpoint_id, load_optimizer=False))
