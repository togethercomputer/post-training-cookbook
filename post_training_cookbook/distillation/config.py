"""Student-side configuration for off-policy distillation."""

from __future__ import annotations

from dataclasses import dataclass, fields

import yaml

from post_training_cookbook.config import validate_lora_config, validate_resume


@dataclass
class DistillConfig:
    # Name exactly one model source: `model` provisions a fresh resource, or
    # `model_resources_id` attaches to an existing shared one (which then defines
    # the base model — leave `model` unset in that case).
    model: str | None = None
    model_resources_id: str | None = None
    # Evaluation samples on the resource's generator, so keep at least one replica.
    # Provisioning only; an attached resource keeps its own replica count.
    num_generator_replicas: int = 1

    batch_size: int = 16
    # Epochs over the rendered samples; fractional values train part of an epoch.
    num_epochs: float = 2.0
    # Optional cap on optimizer steps.
    max_steps: int | None = None
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.05
    min_lr_ratio: float = 0.1
    max_length: int = 2048
    seed: int = 0

    # See post_training_cookbook.codecs.base. Match the teacher data: thinking off when the
    # teacher rows carry no reasoning, on when they carry reasoning_content.
    enable_thinking: bool | None = None
    preserve_thinking: bool | None = None
    thinking_effort: str | None = None

    # Eval sets are scored at step 0, every `eval_interval` steps, and at the last step.
    eval_interval: int = 25
    eval_max_tokens: int = 1024
    # 0.0 samples greedily.
    eval_temperature: float = 0.0

    # None leaves the server's default for that Adam parameter.
    adam_beta1: float | None = None
    adam_beta2: float | None = None
    adam_eps: float | None = None
    weight_decay: float = 0.0

    # LoRA on by default; unset rank/alpha → platform default. False → full weights.
    use_lora: bool = True
    lora_rank: int | None = None
    lora_alpha: int | None = None
    lora_dropout: float | None = None

    training_checkpoint_interval: int = 0
    inference_checkpoint_interval: int = 0
    resume_from_checkpoint_id: str | None = None
    load_optimizer: bool | None = None

    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_group: str | None = None
    timeout: float = 3600.0

    def __post_init__(self) -> None:
        """Validate the config."""
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be > 0")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be >= 1 when set")
        if self.eval_interval < 1:
            raise ValueError("eval_interval must be >= 1")
        if self.num_generator_replicas < 0:
            raise ValueError("num_generator_replicas must be >= 0")
        if self.training_checkpoint_interval < 0 or self.inference_checkpoint_interval < 0:
            raise ValueError("checkpoint intervals must be >= 0 (0 disables)")
        validate_lora_config(self)
        validate_resume(self)

    def run_name_suffix(self) -> str:
        """Hyperparameter slug embedded in the run-artifact directory name."""
        return f"{self.learning_rate:.2e}_{self.batch_size}"

    @classmethod
    def from_yaml(cls, path: str) -> DistillConfig:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        valid = {f.name for f in fields(cls)}
        unknown = set(data) - valid
        if unknown:
            raise ValueError(f"Unknown config keys: {unknown}")
        return cls(**data)
