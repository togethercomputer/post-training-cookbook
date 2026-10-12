"""GRPO on DeepMath level-5 with Qwen3.8-27B (thinking on)."""

from post_training_cookbook.grpo.config import GRPOConfig
from post_training_cookbook.grpo.train import train_single_turn

config = GRPOConfig(
    model="Qwen/Qwen3.8-27B",
    dataset="deepmath-lv5",
    enable_thinking=True,
    num_examples=5000,
    num_val_examples=200,
    batch_size=8,
    group_size=8,
    num_steps=40,
    learning_rate=2e-4,
    max_model_length=16384,
    max_prompt_length=1024,
    max_sample_tokens=15360,
    overlong_penalty_factor=1.0,
    overlong_buffer_len=6144,
    max_sampling_attempts=4,
    validation_interval=5,
    wandb_project="grpo-deepmath-lv5",
)

if __name__ == "__main__":
    train_single_turn(config)
