"""GRPO on Game24 — edit config values below and run."""

from post_training_cookbook.grpo.config import GRPOConfig
from post_training_cookbook.grpo.train import train_single_turn

config = GRPOConfig(
    model="Qwen/Qwen3.6-35B-A3B",
    dataset="game24",
    num_examples=1200,
    num_val_examples=100,
    batch_size=24,
    group_size=4,
    max_sampling_attempts=3,
    num_epochs=4,
    learning_rate=1e-5,
    max_model_length=2048,
    max_prompt_length=512,
    max_sample_tokens=1536,
    overlong_buffer_len=48,
    warmup_ratio=0.03,
    min_lr_ratio=0.3,
    validation_interval=50,
    wandb_project="grpo-game24",
)

if __name__ == "__main__":
    train_single_turn(config)
