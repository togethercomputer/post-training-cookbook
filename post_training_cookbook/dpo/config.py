"""DPO training configuration."""

from __future__ import annotations

from dataclasses import dataclass, fields

import yaml

from post_training_cookbook.config import validate_lora_config, validate_resume


@dataclass
class DPOConfig:
    model: str = "Qwen/Qwen3.8-27B"
    dataset_name: str = "nvidia/HelpSteer3"
    dataset_config: str | None = None
    dataset_split: str = "train"
    dataset_subsample: int | None = 2000
    batch_size: int = 8
    learning_rate: float = 5e-6
    max_length: int = 2048
    # Qwen models in the catalog default to thinking on, which leaves thinking-free
    # preference data (like HelpSteer3) unrenderable. Set False for those.
    enable_thinking: bool | None = None
    max_steps: int = 1000
    beta: float = 0.1
    label_smoothing: float = 0.0
    warmup_ratio: float = 0.03
    min_lr_ratio: float = 0.1
    precompute_ref_batch_size: int = 128
    val_split: str = "validation"
    val_subsample: int | None = 100
    val_interval: int = 100
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_group: str | None = None
    timeout: float = 7200.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    weight_decay: float = 0.1

    use_lora: bool = True
    lora_rank: int | None = None
    lora_alpha: int | None = None
    lora_dropout: float | None = None

    training_checkpoint_interval: int = 0
    inference_checkpoint_interval: int = 0

    # Training-checkpoint ID to start from. Omit ``load_optimizer`` (or True)
    # to restore optimizer and step. ``load_optimizer=False`` loads weights
    # only (fresh optimizer, step 0).
    resume_from_checkpoint_id: str | None = None
    load_optimizer: bool | None = None

    def __post_init__(self) -> None:
        """Validate config fields."""
        if self.training_checkpoint_interval < 0 or self.inference_checkpoint_interval < 0:
            raise ValueError("checkpoint intervals must be >= 0 (0 disables)")
        validate_lora_config(self)
        validate_resume(self)

    def run_name_suffix(self) -> str:
        """Hyperparameter slug embedded in the run-artifact directory name."""
        return f"{self.learning_rate:.2e}_{self.batch_size}_b{self.beta}"

    @classmethod
    def from_yaml(cls, path: str) -> DPOConfig:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        valid = {f.name for f in fields(cls)}
        unknown = set(data) - valid
        if unknown:
            raise ValueError(f"Unknown config keys: {unknown}")
        return cls(**data)
