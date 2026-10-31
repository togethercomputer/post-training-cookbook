"""Shared config helpers for the training configs (GRPO, SFT, DPO).

LoRA ``rank``/``alpha``/``dropout`` are left unset (``None``) by default so the
platform picks them. The cookbook deliberately does not mirror the server's
defaults: a hardcoded client-side copy would silently pin an old value if the
platform ever changed its default. The one LoRA rule the cookbook enforces
locally is the one the server can't surface -- under full-weight fine-tuning
(``use_lora=False``) the server *silently drops* any LoRA fields, so
``validate_lora_config`` rejects them up front.

The optimizer selection (``optimizer_config_from``) follows the same
philosophy: an unset Muon scaling strategy is left to the server, and the one
rule enforced locally -- Muon needs full-weight training -- is validated up
front so the run fails before a resource is provisioned rather than after.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from together.post_training import (
    AdamParams,
    LoraConfig,
    MuonConfig,
    MuonParams,
    OptimizerConfig,
    SessionMetadata,
    WandbMetadata,
)

if TYPE_CHECKING:
    from post_training_cookbook.grpo.config import GRPOConfig


def session_metadata_from(config: Any) -> SessionMetadata | None:
    values = {
        "entity": getattr(config, "wandb_entity", None),
        "group": getattr(config, "wandb_group", None),
        "project": getattr(config, "wandb_project", None),
        "run_name": getattr(config, "wandb_run_name", None),
    }
    values = {key: value for key, value in values.items() if value is not None}
    return SessionMetadata(wandb=WandbMetadata(**values)) if values else None


def validate_resume(config: Any) -> None:
    """Reject weights-only resume without a training checkpoint to load."""
    if getattr(config, "load_optimizer", None) is False and getattr(config, "resume_from_checkpoint_id", None) is None:
        raise ValueError("load_optimizer=False requires resume_from_checkpoint_id")


def validate_lora_config(config: Any) -> None:
    """Reject LoRA fields set under full-weight fine-tuning.

    Called from each config's ``__post_init__``. When ``use_lora=False``, any of
    ``lora_rank``/``lora_alpha``/``lora_dropout`` set to a non-``None`` value
    raises ``ValueError`` -- otherwise ``lora_config_from`` would drop them and
    the user would never see the fields ignored.
    """
    if config.use_lora:
        return

    incompatible_with_full_ft = [
        f for f in ("lora_rank", "lora_alpha", "lora_dropout") if getattr(config, f) is not None
    ]
    if incompatible_with_full_ft:
        raise ValueError(
            f"{', '.join(incompatible_with_full_ft)} cannot be set when use_lora=False (full-weight training). "
            "Remove these fields, or set use_lora=True to keep them."
        )


def lora_config_from(config: Any) -> LoraConfig | None:
    """Translate a config's LoRA fields into the SDK's ``LoraConfig``.

    Only explicitly set fields are passed on; the SDK sends just those, so the
    server keeps deciding the rest.
    """
    if not config.use_lora:
        return None

    lora: dict[str, Any] = {}
    if config.lora_rank is not None:
        lora["rank"] = config.lora_rank
    if config.lora_alpha is not None:
        lora["alpha"] = config.lora_alpha
    if config.lora_dropout is not None:
        lora["dropout"] = config.lora_dropout
    return LoraConfig(**lora) if lora else None


def validate_optimizer(config: GRPOConfig) -> None:
    """Reject unsupported optimizer selections.

    Called from ``__post_init__``. Muon tunes full weight matrices and the
    platform rejects a Muon session that also trains LoRA adapters, so require
    ``use_lora=False`` up front rather than failing at resource creation.
    """
    # The value set guards the from_yaml path, where the field's Literal type does not apply.
    if config.optimizer not in ("adam", "muon"):
        raise ValueError(f"optimizer must be 'adam' or 'muon' (got {config.optimizer!r})")
    if config.optimizer == "muon" and config.use_lora:
        raise ValueError("optimizer='muon' requires full-weight training; set use_lora=False.")


def optimizer_config_from(config: GRPOConfig) -> OptimizerConfig | None:
    """The resource-level optimizer selection, or ``None`` for the server default (AdamW).

    Only Muon needs an explicit selection, and an unset ``muon_scaling_strategy``
    is left to the server rather than pinned client-side.
    """
    if config.optimizer != "muon":
        return None
    muon: dict[str, Any] = {}
    if config.muon_scaling_strategy is not None:
        muon["scaling_strategy"] = config.muon_scaling_strategy
    return OptimizerConfig(muon=MuonConfig(**muon))


def optim_step_params(config: GRPOConfig, learning_rate: float) -> dict[str, Any]:
    """The optimizer-specific ``optim_step`` keyword for this step.

    Returns ``{"adam_params": ...}``, or for Muon ``{"muon_params": ...}`` with a
    nested AdamW group for the parameters Muon does not tune; both groups share the
    step's ``learning_rate``. See https://docs.together.ai/docs/post-training/cookbook#optimizer.
    """
    adam = AdamParams(
        beta1=config.adam_beta1,
        beta2=config.adam_beta2,
        eps=config.adam_eps,
        weight_decay=config.weight_decay,
        learning_rate=learning_rate,
    )
    if config.optimizer != "muon":
        return {"adam_params": adam}

    muon = MuonParams(
        learning_rate=learning_rate,
        weight_decay=config.weight_decay,
        adam=adam,
    )
    if config.muon_momentum is not None:
        muon["momentum"] = config.muon_momentum
    if config.muon_newton_schulz_steps is not None:
        muon["newton_schulz_steps"] = config.muon_newton_schulz_steps
    return {"muon_params": muon}
