"""GRPO on GSM8K with the Muon optimizer — edit config values below and run.

Muon (``optimizer="muon"``) requires full-weight training (``use_lora=False``).
See https://docs.together.ai/docs/post-training/cookbook#optimizer for its settings.
"""

from post_training_cookbook.grpo.config import GRPOConfig
from post_training_cookbook.grpo.train import train_single_turn

config = GRPOConfig(
    model="Qwen/Qwen3.8-27B",
    dataset="gsm8k",
    enable_thinking=False,
    num_examples=4000,
    num_val_examples=200,
    batch_size=16,
    group_size=8,
    max_sampling_attempts=3,
    num_steps=200,
    learning_rate=1e-5,
    max_model_length=1536,
    max_prompt_length=768,
    max_sample_tokens=768,
    training_checkpoint_interval=20,
    inference_checkpoint_interval=20,
    validation_interval=100,
    wandb_project="grpo-gsm8k-muon",
    optimizer="muon",
    use_lora=False,
)

if __name__ == "__main__":
    train_single_turn(config)
