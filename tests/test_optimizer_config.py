"""Tests for cookbook optimizer validation and the SDK-shaped optimizer builders."""

from __future__ import annotations

import inspect

import pytest
from together.post_training import Trainer

from post_training_cookbook.config import optim_step_params, optimizer_config_from
from post_training_cookbook.grpo.config import GRPOConfig


def _grpo(**kw) -> GRPOConfig:
    return GRPOConfig(model="test-model", **kw)


class TestOptimizerValidation:
    def test_muon_requires_full_weight(self):
        with pytest.raises(ValueError, match="use_lora=False"):
            _grpo(optimizer="muon")

    def test_muon_with_full_weight_is_allowed(self):
        assert _grpo(optimizer="muon", use_lora=False).optimizer == "muon"

    def test_unknown_optimizer_rejected(self):
        with pytest.raises(ValueError, match="must be 'adam' or 'muon'"):
            _grpo(optimizer="lion")


class TestOptimizerConfigFrom:
    def test_adam_defers_to_server(self):
        assert optimizer_config_from(_grpo()) is None

    def test_muon_without_strategy_selects_muon(self):
        config = optimizer_config_from(_grpo(optimizer="muon", use_lora=False))
        assert config == {"muon": {}}

    def test_muon_with_strategy_emits_it(self):
        c = _grpo(optimizer="muon", use_lora=False, muon_scaling_strategy="MUON_SCALING_STRATEGY_MATCH_ADAM")
        config = optimizer_config_from(c)
        assert config == {"muon": {"scaling_strategy": "MUON_SCALING_STRATEGY_MATCH_ADAM"}}


class TestOptimStepParams:
    def test_adam_sends_adam_params(self):
        params = optim_step_params(_grpo(), 1e-4)
        assert set(params) == {"adam_params"}
        assert params["adam_params"]["learning_rate"] == 1e-4

    def test_muon_sends_muon_params_with_nested_adam(self):
        """Muon tunes the matrices; the nested AdamW group tunes the rest, sharing the schedule."""
        params = optim_step_params(_grpo(optimizer="muon", use_lora=False, muon_momentum=0.95), 2e-2)
        assert set(params) == {"muon_params"}
        muon = params["muon_params"]
        assert muon["learning_rate"] == 2e-2
        assert muon["momentum"] == 0.95
        assert muon["adam"]["learning_rate"] == 2e-2

    def test_unset_muon_knobs_are_omitted(self):
        """Unset momentum / newton_schulz_steps defer to the server rather than pin a value."""
        muon = optim_step_params(_grpo(optimizer="muon", use_lora=False), 2e-2)["muon_params"]
        assert "momentum" not in muon
        assert "newton_schulz_steps" not in muon

    @pytest.mark.parametrize("config", [_grpo(), _grpo(optimizer="muon", use_lora=False)])
    def test_keys_are_real_optim_step_kwargs(self, config):
        """These become **kwargs on the SDK call, so a stale name is a runtime TypeError."""
        accepted = inspect.signature(Trainer.optim_step).parameters
        assert set(optim_step_params(config, 1e-4)) <= set(accepted)
