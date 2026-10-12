"""GRPO learning-rate sweep on full DeepMath — four arms, one model resource.

The owner half of the multi-LoRA lifecycle: provision a single base model's
worth of GPUs and train all four LR arms on it concurrently, each as an
independent LoRA adapter session. Compare arms by reading their validation
curves side by side in the `deepmath-sweep` W&B group — per-arm validation
already is the model selection, no extra eval code. An arm that raises doesn't
stop its siblings, and the context manager guarantees the resource is torn down
exactly once when the sweep finishes.

This mirrors the shipped `deepmath.py` recipe (Qwen3.6-35B-A3B, full DeepMath,
thinking-on, 16k/15k budget, LoRA defaults left to the platform) and sweeps the
learning rate around the recipe's default to compare validation performance.
"""

from concurrent.futures import ThreadPoolExecutor

from together.post_training import ModelResourcesClient

from post_training_cookbook.grpo.config import GRPOConfig
from post_training_cookbook.grpo.train import train_single_turn

MODEL = "Qwen/Qwen3.6-35B-A3B"


def build_config(model_resources_id: str, lr: float) -> GRPOConfig:
    return GRPOConfig(
        model_resources_id=model_resources_id,
        learning_rate=lr,
        wandb_run_name=f"lr{lr:.0e}",
        dataset="deepmath",
        num_examples=2000,
        num_val_examples=128,
        batch_size=16,
        group_size=8,
        num_steps=20,
        max_model_length=16384,
        max_prompt_length=1024,
        max_sample_tokens=15360,
        max_sampling_attempts=4,
        validation_interval=5,
        wandb_project="grpo-deepmath",
        wandb_group="deepmath-sweep",
    )


LEARNING_RATES = (5e-5, 1e-4, 2e-4, 4e-4)

if __name__ == "__main__":
    # Provision the base-model GPUs once; every arm attaches its own adapter session.
    with ModelResourcesClient.create(base_model=MODEL, lora_enabled=True) as resource:
        configs = [build_config(resource.model_resources_id, lr) for lr in LEARNING_RATES]
        with ThreadPoolExecutor(max_workers=len(configs)) as pool:
            arms = {cfg.wandb_run_name: pool.submit(train_single_turn, cfg) for cfg in configs}
        failed = {name: exc for name, future in arms.items() if (exc := future.exception()) is not None}

    for name, exc in failed.items():
        print(f"[{name}] arm failed: {exc!r}")
    if failed:
        raise SystemExit(f"{len(failed)}/{len(configs)} sweep arms failed")
