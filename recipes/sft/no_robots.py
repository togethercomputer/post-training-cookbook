"""SFT on No Robots — edit config values below and run."""

from post_training_cookbook.sft.config import SFTConfig
from post_training_cookbook.sft.train import TRAIN_ON_LAST, train

config = SFTConfig(
    model="Qwen/Qwen3.8-27B",
    # Qwen3.8-27B's template enables thinking by default, which leaves No Robots'
    # thinking-free conversations unrenderable and skips every row.
    enable_thinking=False,
    dataset_name="HuggingFaceH4/no_robots",
    dataset_split="train",
    batch_size=8,
    learning_rate=1e-4,
    max_length=4096,
    max_steps=50,
    train_on_what=TRAIN_ON_LAST,
    training_checkpoint_interval=50,
    wandb_project="sft-no-robots",
)

if __name__ == "__main__":
    train(config)
