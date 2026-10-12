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
