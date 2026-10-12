"""Tests for cookbook LoRA validation and the SDK-shaped lora_config builder."""

from __future__ import annotations

import pytest

from post_training_cookbook.config import lora_config_from, session_metadata_from
from post_training_cookbook.dpo.config import DPOConfig
from post_training_cookbook.grpo.config import GRPOConfig
from post_training_cookbook.sft.config import SFTConfig


@pytest.fixture(params=["sft", "grpo", "dpo"], ids=["sft", "grpo", "dpo"])
def config_factory(request):
    """Build an SFTConfig, GRPOConfig, or DPOConfig with extra kwargs.

    Lets a single test exercise all configs, since they share the
    LoRA validator + serializer.
    """
    if request.param == "sft":
        return lambda **kw: SFTConfig(**kw)
    if request.param == "grpo":
        return lambda **kw: GRPOConfig(model="test-model", **kw)
    return lambda **kw: DPOConfig(**kw)


class TestLoraDefaults:
    def test_unset_lora_stays_none(self, config_factory):
        """Unset rank/alpha are left to the server, not pinned client-side."""
        c = config_factory()
        assert c.use_lora is True
        assert (c.lora_rank, c.lora_alpha, c.lora_dropout) == (None, None, None)

    def test_explicit_lora_values_preserved(self, config_factory):
        c = config_factory(lora_rank=32, lora_alpha=64, lora_dropout=0.1)
        assert (c.lora_rank, c.lora_alpha, c.lora_dropout) == (32, 64, 0.1)

    def test_full_weight_keeps_lora_fields_none(self, config_factory):
        c = config_factory(use_lora=False)
        assert (c.lora_rank, c.lora_alpha, c.lora_dropout) == (None, None, None)


class TestLoraValidation:
    @pytest.mark.parametrize(("field", "value"), [("lora_rank", 16), ("lora_alpha", 16), ("lora_dropout", 0.05)])
    def test_full_weight_rejects_lora_fields(self, config_factory, field, value):
        with pytest.raises(ValueError, match=f"{field} cannot be set when use_lora=False"):
            config_factory(use_lora=False, **{field: value})

    def test_full_weight_rejects_multiple_lora_fields(self, config_factory):
        with pytest.raises(ValueError, match="lora_rank, lora_alpha, lora_dropout"):
            config_factory(use_lora=False, lora_rank=16, lora_alpha=16, lora_dropout=0.05)


class TestLoraConfigFrom:
    def test_unset_rank_defers_to_server(self, config_factory):
        assert lora_config_from(config_factory()) is None

    def test_explicit_fields_are_sent(self, config_factory):
        c = config_factory(lora_rank=16, lora_alpha=32, lora_dropout=0.05)
        config = lora_config_from(c)
        assert config == {"rank": 16, "alpha": 32, "dropout": 0.05}

    def test_only_set_fields_are_sent(self, config_factory):
        """Unset fields stay off the wire so the server keeps deciding them."""
        config = lora_config_from(config_factory(lora_rank=16))
        assert config == {"rank": 16}

    def test_full_weight_omits_session_lora_config(self, config_factory):
        assert lora_config_from(config_factory(use_lora=False)) is None


def test_session_metadata_uses_configured_wandb_fields():
    config = GRPOConfig(
        model="test-model",
        wandb_project="project",
        wandb_entity="entity",
        wandb_group="group",
        wandb_run_name="run-name",
    )

    metadata = session_metadata_from(config)

    assert metadata == {
        "wandb": {
            "entity": "entity",
            "group": "group",
            "project": "project",
            "run_name": "run-name",
        }
    }
