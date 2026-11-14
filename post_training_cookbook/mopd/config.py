"""Configuration for the MOPD fusion stage."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Literal, get_args

import yaml

from post_training_cookbook.config import validate_resume
from post_training_cookbook.mopd.distill import Balance

__all__ = ["MOPDConfig", "TeacherScoring"]

# "generator" scores with prompt_logprobs on the teacher's generator (same engine as the student's
# sampling logprobs); "trainer" uses the teacher's forward-only trainer pass.
TeacherScoring = Literal["generator", "trainer"]


@dataclass
class MOPDConfig:
    """Fusion-stage settings: one student distilled from one teacher per domain.

    Name exactly one model source: ``model`` provisions a fresh resource for the run, or
    ``model_resources_id`` attaches to an existing one (the usual MOPD setup, since the experts
    trained there). Teachers are given as ``teacher_checkpoints``, a mapping from domain name to
    the expert's *training* checkpoint ID.

    Every prompt is answered ``samples_per_prompt`` times, so a step trains on
    ``len(domains) * prompts_per_domain * samples_per_prompt`` completions.
    """

    model: str | None = None
    model_resources_id: str | None = None
    num_generator_replicas: int = 1
    teacher_checkpoints: dict[str, str] = field(default_factory=dict)

    # Training-checkpoint ID to start the student from (an SFT model, say); unset starts from the
    # base model. ``load_optimizer=False`` loads weights only (fresh optimizer, step 0); omit or
    # True to also restore optimizer state. The client step count always restarts at zero.
    resume_from_checkpoint_id: str | None = None
    load_optimizer: bool | None = None

    # Student LoRA; unset rank/alpha/dropout → platform default. Every session on a resource is a
    # LoRA adapter, so MOPD has no full-weight mode.
    lora_rank: int | None = None
    lora_alpha: int | None = None
    lora_dropout: float | None = None
    # Teacher LoRA. Must equal the experts' settings: a session resumed from a checkpoint does not
    # inherit its rank and alpha. Unset matches experts trained with the platform defaults.
    teacher_lora_rank: int | None = None
    teacher_lora_alpha: int | None = None

    num_steps: int = 50
    prompts_per_domain: int = 16
    samples_per_prompt: int = 4
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.05
    min_lr_ratio: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    weight_decay: float = 0.0

    # Distillation signal: advantage = kl_coef * (teacher logprob - student logprob), shared
    # between domains per ``balance``.
    kl_coef: float = 1.0
    balance: Balance = "token"
    teacher_scoring: TeacherScoring = "generator"

    # Sampling. Temperature 1.0 keeps the student's samples, and so the reverse-KL estimate, on its
    # actual distribution.
    temperature: float = 1.0
    max_model_length: int = 4096
    max_prompt_length: int = 1024
    max_sample_tokens: int = 2048
    enable_thinking: bool | None = False

    # Held-out monitoring: greedy decoding on the first ``eval_examples_per_domain`` test rows of
    # each domain, every ``eval_interval`` steps (and at step 0 and the last step).
    eval_interval: int = 10
    eval_examples_per_domain: int = 100

    training_checkpoint_interval: int = 0
    export_samples: bool = True
    timeout: float = 7200.0

    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_group: str | None = None

    def __post_init__(self) -> None:
        """Reject inconsistent settings before anything is provisioned."""
        if (self.model is None) == (self.model_resources_id is None):
            raise ValueError("Set exactly one of model (provision a resource) or model_resources_id (attach).")
        if not self.teacher_checkpoints:
            raise ValueError("teacher_checkpoints must map every domain name to an expert training checkpoint")
        for name in ("num_steps", "prompts_per_domain", "samples_per_prompt", "eval_examples_per_domain"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.eval_interval < 0 or self.training_checkpoint_interval < 0:
            raise ValueError("eval_interval and training_checkpoint_interval must be >= 0 (0 disables)")
        if self.balance not in get_args(Balance):
            raise ValueError(f"balance must be one of {get_args(Balance)}, got {self.balance!r}")
        if self.teacher_scoring not in get_args(TeacherScoring):
            choices = get_args(TeacherScoring)
            raise ValueError(f"teacher_scoring must be one of {choices}, got {self.teacher_scoring!r}")
        if self.max_prompt_length + self.max_sample_tokens > self.max_model_length:
            raise ValueError(
                f"max_prompt_length ({self.max_prompt_length}) + max_sample_tokens ({self.max_sample_tokens}) "
                f"exceeds max_model_length ({self.max_model_length})"
            )
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0: the reverse-KL estimate needs sampled completions")
        validate_resume(self)

    def run_name_suffix(self) -> str:
        return f"{self.learning_rate:.2e}_{self.prompts_per_domain}x{self.samples_per_prompt}_{self.balance}"

    @classmethod
    def from_yaml(cls, path: str) -> MOPDConfig:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown config keys: {unknown}")
        return cls(**data)
