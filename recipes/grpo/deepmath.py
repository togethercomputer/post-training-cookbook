"""GRPO on full DeepMath with Qwen3.6-35B-A3B."""

from post_training_cookbook.grpo.config import GRPOConfig
from post_training_cookbook.grpo.train import train_single_turn

config = GRPOConfig(
    model="Qwen/Qwen3.6-35B-A3B",
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
    train_single_turn(config)
