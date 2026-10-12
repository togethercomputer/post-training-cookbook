"""DPO on nvidia/HelpSteer3 — edit values below and run."""

from post_training_cookbook.dpo.config import DPOConfig
from post_training_cookbook.dpo.train import train

config = DPOConfig(
    model="Qwen/Qwen3.8-27B",
    enable_thinking=False,  # preference data has no reasoning traces
    dataset_name="nvidia/HelpSteer3",
    dataset_split="train",
    # The loop visits at most max_steps * batch_size unique pairs
    dataset_subsample=1600,
    batch_size=8,
    learning_rate=5e-6,
    beta=0.1,
    max_length=4096,
    max_steps=200,
    precompute_ref_batch_size=512,
)

if __name__ == "__main__":
    train(config)
