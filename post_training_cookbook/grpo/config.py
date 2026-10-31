"""GRPO training configuration."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Literal, get_args

import yaml
from together.post_training import (
    CispoLossParams,
    DroLossParams,
    GrpoLossParams,
    LossConfig,
    MuonScalingStrategy,
    PpoLossParams,
    WeightSyncType,
)

from post_training_cookbook.config import validate_lora_config, validate_optimizer, validate_resume

PolicyLoss = Literal["ppo", "grpo", "cispo", "dppo", "dro", "importance_sampling"]

_DEFAULT_AGG = "GRPO_LOSS_AGGREGATION_TYPE_FIXED_HORIZON"
_DEFAULT_RATIO = "GRPO_LOSS_RATIO_TYPE_TOKEN"
_CLIP_LOSSES = frozenset({"ppo", "grpo", "cispo"})


def _require_int(name: str, value: object, minimum: int = 1) -> None:
    """Reject invalid integer values loaded from YAML."""
    # bool is an int subclass, hence the explicit exclusion.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")


def _clip_kwargs(config: GRPOConfig) -> dict[str, float]:
    """Only include clip fields the caller set; omitted → server defaults."""
    out: dict[str, float] = {}
    if config.clip_low_threshold is not None:
        out["clip_low_threshold"] = config.clip_low_threshold
    if config.clip_high_threshold is not None:
        out["clip_high_threshold"] = config.clip_high_threshold
    return out


def policy_loss(config: GRPOConfig) -> LossConfig:
    """Translate ``GRPOConfig`` loss knobs into the SDK ``LossConfig`` payload."""
    clips = _clip_kwargs(config)
    match config.loss:
        case "ppo":
            if not clips:
                return LossConfig(type="LOSS_TYPE_PPO")
            return LossConfig(type="LOSS_TYPE_PPO", ppo_params=PpoLossParams(**clips))
        case "grpo":
            return LossConfig(
                type="LOSS_TYPE_GRPO",
                grpo_params=GrpoLossParams(
                    agg_type=config.loss_agg_type,
                    ratio_type=config.ratio_type,
                    **clips,
                ),
            )
        case "cispo":
            if not clips:
                return LossConfig(type="LOSS_TYPE_CISPO")
            return LossConfig(type="LOSS_TYPE_CISPO", cispo_params=CispoLossParams(**clips))
        case "dppo":
            return LossConfig(type="LOSS_TYPE_DPPO")
        case "dro":
            assert config.dro_beta is not None  # validated in __post_init__
            return LossConfig(
                type="LOSS_TYPE_DRO",
                dro_params=DroLossParams(beta=config.dro_beta),
            )
        case "importance_sampling":
            return LossConfig(type="LOSS_TYPE_IMPORTANCE_SAMPLING")
        case _:
            raise ValueError(f"unsupported loss: {config.loss!r}")


@dataclass
class GRPOConfig:
    """Configuration for GRPO training.

    Length parameters: only ``max_model_length`` is required.  The others
    derive sensible defaults from it when left as ``None``:

    * ``max_sample_tokens`` — caps tokens generated per sample.
      Defaults to ``max_model_length`` (fill the context).
    * ``max_prompt_length`` — hard filter for single-turn prompt length.
      Defaults to ``max_model_length - max_sample_tokens``.
    * ``overlong_buffer_len`` — token buffer before the overlong penalty
      ramps up.  Defaults to ``max_sample_tokens // 8``.

    Overlong handling is controlled by ``overlong_penalty_factor`` alone:
    set it to ``0`` to disable.  The ``overlong_penalty`` property exposes
    the on/off state for convenience.

    ``loss`` names the server objective over the same group-relative rollouts.
    Clip knobs are optional for ``ppo`` / ``grpo`` / ``cispo``; ``None`` omits
    them so the server defaults apply (PPO/GRPO ``0.8``/``1.2``, CISPO ``0.0``/``4.0``).
    ``loss_agg_type`` / ``ratio_type`` are ``grpo``-only (sequence ratio = GSPO);
    ``dro_beta`` is required for ``dro``. ``dppo`` and ``importance_sampling``
    send no client params.
    """

    # Shared. Name exactly one model source: `model` provisions a fresh resource,
    # or `model_resources_id` attaches to an existing shared one (which then
    # defines the base model — leave `model` unset in that case).
    model: str | None = None
    model_resources_id: str | None = None
    # Provisioning only; attached resources retain their existing replica count.
    num_generator_replicas: int = 1
    batch_size: int = 16
    num_steps: int | None = None
    learning_rate: float = 1e-5
    warmup_ratio: float = 0.05
    min_lr_ratio: float = 0.1

    # Policy objective over the GRPO loop's group-relative advantages.
    loss: PolicyLoss = "grpo"
    # Absolute ratio bounds (not offsets from 1.0). None → omit; server defaults
    # are loss-specific (ppo/grpo 0.8/1.2, cispo 0.0/4.0).
    clip_low_threshold: float | None = None
    clip_high_threshold: float | None = None
    # GRPO-only: how per-token losses reduce to a scalar, and whether the ratio
    # is token-level or sequence-level (GSPO).
    loss_agg_type: str = _DEFAULT_AGG
    ratio_type: str = _DEFAULT_RATIO
    # Required when loss="dro"; rejected otherwise.
    dro_beta: float | None = None

    # How the trainer's new weights reach the generator after each optimizer step.
    # SYNCHRONOUS publishes inline; the deferred modes require async collection
    # (max_steps_off_policy) to bound and measure the drift they allow. PIPELINE
    # additionally lets one action span two policies. See https://docs.together.ai/docs/post-training/cookbook.
    weight_sync_type: WeightSyncType = "WEIGHT_SYNC_TYPE_SYNCHRONOUS"

    max_model_length: int = 2048
    max_sample_tokens: int | None = None
    min_response_tokens: int = 1
    # See post_training_cookbook.codecs.base for the preserve_thinking vs default-split trade-off.
    enable_thinking: bool | None = None
    preserve_thinking: bool | None = None
    # Family-specific reasoning budget (Kimi-K3, GLM-5: low/high/max; Muse Spark: low/medium/high/xhigh);
    # None = template default.
    thinking_effort: str | None = None

    overlong_penalty_factor: float = 1.0
    overlong_masking: bool = False
    overlong_buffer_len: int | None = None
    temperature: float = 0.7

    # Validation. ``val_temperature=None`` means greedy (0.0); ``val_max_turns=None``
    # reuses ``max_turns``. ``val_max_turns`` and ``sequential_validation`` are multi-turn only.
    validation_interval: int = 50
    validation_iterations: int = 1
    val_temperature: float | None = None
    val_max_turns: int | None = None
    sequential_validation: bool = False

    training_checkpoint_interval: int = 0
    inference_checkpoint_interval: int = 0

    # Training-checkpoint ID to start from. Omit ``load_optimizer`` (or True)
    # to restore optimizer and step. ``load_optimizer=False`` loads weights
    # only (fresh optimizer, step 0) — e.g. SFT→RL.
    resume_from_checkpoint_id: str | None = None
    load_optimizer: bool | None = None
    # Source run directory whose ``checkpoints.jsonl`` holds the client step and
    # data position saved with ``resume_from_checkpoint_id``. Required to continue.
    resume_run_dir: str | None = None

    # DAPO dynamic sampling cap. ``1`` (default) draws one batch per step. When
    # set higher, each step keeps drawing batches and discarding zero-variance
    # groups until ``batch_size`` groups with non-trivial advantages have been
    # collected, or this many batches have been drawn — whichever first. The cap
    # protects against pathological cases where almost every group collapses to
    # identical rewards.
    max_sampling_attempts: int = 1

    wandb_project: str = "post-training-cookbook"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_group: str | None = None
    export_samples: bool = True

    # LoRA on by default; unset rank/alpha → platform default. False → full weights.
    use_lora: bool = True
    lora_rank: int | None = None
    lora_alpha: int | None = None
    lora_dropout: float | None = None

    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    weight_decay: float = 0.01

    # Muon requires full-weight training (use_lora=False); unset knobs defer to
    # the server. See https://docs.together.ai/docs/post-training/cookbook#optimizer for what Muon tunes.
    optimizer: Literal["adam", "muon"] = "adam"
    muon_scaling_strategy: MuonScalingStrategy | None = None
    muon_momentum: float | None = None
    muon_newton_schulz_steps: int | None = None

    timeout: float = 7200.0

    # Single-turn only (ignored for multi-turn)
    dataset: str = "gsm8k"
    num_examples: int = 7000
    num_val_examples: int = 200
    group_size: int = 8
    num_epochs: int = 1
    max_prompt_length: int | None = None

    # Capture generator MoE routing and forward its keys with training samples.
    routing_replay: bool = False

    # Multi-turn only (ignored for single-turn; max_turns=None means single-turn mode)
    max_turns: int | None = None
    episodes_per_task: int = 1
    failed_parse_reward: float = -0.1
    # None keeps lockstep collection; 0 enables async collection with no stale groups.
    # ``async_training_loop`` validates the rest of the async preconditions, so that
    # ``replace()``-derived configs (validation) are not held to them.
    max_steps_off_policy: int | None = None
    groups_in_flight: int | None = None

    def __post_init__(self) -> None:
        """Validate and set derived defaults."""
        if self.max_sample_tokens is None:
            self.max_sample_tokens = self.max_model_length
        if self.max_prompt_length is None:
            self.max_prompt_length = self.max_model_length - self.max_sample_tokens
        if self.overlong_buffer_len is None:
            self.overlong_buffer_len = self.max_sample_tokens // 8
        if self.groups_in_flight is None:
            self.groups_in_flight = self.batch_size

        # from_yaml does not enforce annotations, so the integer knobs a malformed
        # config would otherwise break deep inside a request are checked here.
        _require_int("num_generator_replicas", self.num_generator_replicas)
        _require_int("batch_size", self.batch_size)
        _require_int("groups_in_flight", self.groups_in_flight)
        _require_int("max_sampling_attempts", self.max_sampling_attempts)

        if self.max_steps_off_policy is not None:
            _require_int("max_steps_off_policy", self.max_steps_off_policy, minimum=0)

        if self.loss not in get_args(PolicyLoss):
            raise ValueError(f"unsupported loss: {self.loss!r}")

        if self.loss == "dro":
            if self.dro_beta is None:
                raise ValueError("dro_beta is required when loss='dro'")
        elif self.dro_beta is not None:
            raise ValueError("dro_beta is only valid when loss='dro'")

        if self.loss != "grpo":
            if self.loss_agg_type != _DEFAULT_AGG:
                raise ValueError("loss_agg_type is only valid when loss='grpo'")
            if self.ratio_type != _DEFAULT_RATIO:
                raise ValueError("ratio_type is only valid when loss='grpo'")

        if self.loss not in _CLIP_LOSSES and (
            self.clip_low_threshold is not None or self.clip_high_threshold is not None
        ):
            raise ValueError("clip thresholds are only valid for loss in {'ppo', 'grpo', 'cispo'}")

        if self.weight_sync_type not in get_args(WeightSyncType):
            raise ValueError(f"unsupported weight_sync_type: {self.weight_sync_type}")
        # A deferred publish only makes sense where something accounts for the
        # resulting off-policy tokens. Lockstep collection has no staleness budget
        # and logs no async/* metrics, so the drift would be silent and unbounded.
        if self.weight_sync_type != "WEIGHT_SYNC_TYPE_SYNCHRONOUS" and self.max_steps_off_policy is None:
            raise ValueError(
                f"{self.weight_sync_type} requires async collection; set max_steps_off_policy "
                "(or use WEIGHT_SYNC_TYPE_SYNCHRONOUS)"
            )
        # The server rejects PIPELINE on a LoRA session; fail before provisioning.
        if self.weight_sync_type == "WEIGHT_SYNC_TYPE_PIPELINE" and self.use_lora:
            raise ValueError("WEIGHT_SYNC_TYPE_PIPELINE requires use_lora=False")

        if self.max_sample_tokens > self.max_model_length:
            raise ValueError(
                f"max_sample_tokens ({self.max_sample_tokens}) exceeds max_model_length ({self.max_model_length})"
            )
        if self.max_prompt_length + self.min_response_tokens > self.max_model_length:
            raise ValueError(
                f"max_prompt_length ({self.max_prompt_length}) + "
                f"min_response_tokens ({self.min_response_tokens}) "
                f"exceeds max_model_length ({self.max_model_length})"
            )

        if self.training_checkpoint_interval < 0 or self.inference_checkpoint_interval < 0:
            raise ValueError("checkpoint intervals must be >= 0 (0 disables)")

        validate_lora_config(self)
        validate_optimizer(self)
        validate_resume(self)
        if self.continues_run and self.resume_run_dir is None:
            raise ValueError(
                "Continuing from a training checkpoint requires resume_run_dir (the source run directory); "
                "use load_optimizer=False to start a new run from the weights"
            )
        if not self.continues_run and self.resume_run_dir is not None:
            raise ValueError("resume_run_dir requires resume_from_checkpoint_id without load_optimizer=False")

    @property
    def continues_run(self) -> bool:
        """Whether this run continues a checkpointed run rather than starting at step 0."""
        return self.resume_from_checkpoint_id is not None and self.load_optimizer is not False

    @property
    def overlong_penalty(self) -> bool:
        return self.overlong_penalty_factor > 0

    def run_name_suffix(self) -> str:
        """Hyperparameter slug embedded in the run-artifact directory name."""
        group = self.episodes_per_task if self.max_turns is not None else self.group_size
        return f"{self.learning_rate:.2e}_{self.batch_size}x{group}"

    @classmethod
    def from_yaml(cls, path: str) -> GRPOConfig:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        valid = {f.name for f in fields(cls)}
        unknown = set(data) - valid
        if unknown:
            raise ValueError(f"Unknown config keys: {unknown}")
        return cls(**data)
