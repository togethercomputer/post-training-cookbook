"""SFT training configuration."""

from __future__ import annotations

from dataclasses import dataclass, fields

import yaml

from post_training_cookbook.config import validate_lora_config, validate_resume

TRAIN_ON_ALL = "all_assistant_messages"
TRAIN_ON_LAST = "last_assistant_message"


@dataclass
class SFTConfig:
    # Name exactly one model source: `model` provisions a fresh resource, or
    # `model_resources_id` attaches to an existing shared one (which then defines
    # the base model — leave `model` unset in that case).
    model: str | None = None
    model_resources_id: str | None = None
    # A Hugging Face Hub repo ID, or a local directory of .jsonl files.
    dataset_name: str = "HuggingFaceH4/no_robots"
    dataset_config: str | None = None
    dataset_split: str = "train"
    batch_size: int = 8
    learning_rate: float = 1e-4
    max_length: int = 4096
    num_epochs: int = 1
    # Optional cap on optimizer steps; None trains every full batch of every epoch.
    max_steps: int | None = None
    train_on_what: str = TRAIN_ON_ALL
    # See post_training_cookbook.codecs.base for the preserve_thinking vs default-split trade-off.
    enable_thinking: bool | None = None
    preserve_thinking: bool | None = None
    # Family-specific reasoning budget (Kimi-K3, GLM-5: low/high/max; Muse Spark: low/medium/high/xhigh);
    # None = template default.
    thinking_effort: str | None = None
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_group: str | None = None
    timeout: float = 7200.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    weight_decay: float = 0.1

    # LoRA on by default; unset rank/alpha → platform default. False → full weights.
    use_lora: bool = True
    lora_rank: int | None = None
    lora_alpha: int | None = None
    lora_dropout: float | None = None

    training_checkpoint_interval: int = 0
    inference_checkpoint_interval: int = 0

    # Training-checkpoint ID to start from. Omit ``load_optimizer`` (or True)
    # to restore optimizer and step. ``load_optimizer=False`` loads weights
    # only (fresh optimizer, step 0) — e.g. SFT→RL.
    resume_from_checkpoint_id: str | None = None
    load_optimizer: bool | None = None

    def __post_init__(self) -> None:
        """Validate and set derived defaults."""
        if self.training_checkpoint_interval < 0 or self.inference_checkpoint_interval < 0:
            raise ValueError("checkpoint intervals must be >= 0 (0 disables)")
        if self.train_on_what not in (TRAIN_ON_ALL, TRAIN_ON_LAST):
            raise ValueError(f"train_on_what must be {TRAIN_ON_ALL!r} or {TRAIN_ON_LAST!r}: {self.train_on_what!r}")
        validate_lora_config(self)
        validate_resume(self)

    def run_name_suffix(self) -> str:
        """Hyperparameter slug embedded in the run-artifact directory name."""
        return f"{self.learning_rate:.2e}_{self.batch_size}"

    @classmethod
    def from_yaml(cls, path: str) -> SFTConfig:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        valid = {f.name for f in fields(cls)}
        unknown = set(data) - valid
        if unknown:
            raise ValueError(f"Unknown config keys: {unknown}")
        return cls(**data)
