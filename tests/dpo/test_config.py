"""Tests for DPO config validation."""

import pytest

from post_training_cookbook.dpo.config import DPOConfig


def test_load_optimizer_false_requires_checkpoint():
    with pytest.raises(ValueError, match="load_optimizer=False requires resume_from_checkpoint_id"):
        DPOConfig(load_optimizer=False)


def test_weights_only_resume_accepted():
    config = DPOConfig(resume_from_checkpoint_id="tckpt-1", load_optimizer=False)
    assert config.resume_from_checkpoint_id == "tckpt-1"
    assert config.load_optimizer is False


def test_accepts_provisioning_timeouts():
    config = DPOConfig(ready_timeout_s=60.0, inactive_timeout_s=30)
    assert config.ready_timeout_s == 60.0
    assert config.inactive_timeout_s == 30


def test_model_source_and_run_dir_default_to_none():
    config = DPOConfig()
    assert (config.model, config.model_resources_id, config.run_dir) == (None, None, None)


@pytest.mark.parametrize("field", ["ready_timeout_s", "inactive_timeout_s"])
def test_rejects_timeouts_when_attaching(field):
    with pytest.raises(ValueError, match="only apply when provisioning"):
        DPOConfig(model_resources_id="mr-1", **{field: 30})
