"""SFT on No Robots with full-weight fine-tuning (no LoRA adapters).

Demonstrates the ``use_lora=False`` toggle. Provisions the model resource with
``lora_enabled=False``, so the trainer updates full weights instead of LoRA adapters.
"""

import os

from dotenv import load_dotenv

from post_training_cookbook.sft.config import SFTConfig
from post_training_cookbook.sft.train import TRAIN_ON_LAST, train

load_dotenv(override=True)

config = SFTConfig(
    model="Qwen/Qwen3.8-27B",
    # Qwen3.8-27B's template enables thinking by default, which leaves No Robots'
    # thinking-free conversations unrenderable and skips every row.
    enable_thinking=False,
    dataset_name="HuggingFaceH4/no_robots",
    dataset_split="train",
    batch_size=4,
    learning_rate=2e-5,
    max_length=4096,
    max_steps=50,
    train_on_what=TRAIN_ON_LAST,
    use_lora=False,
    wandb_project=os.environ.get("WANDB_PROJECT"),
)

if __name__ == "__main__":
    train(config)
