"""Tests for the GRPO training loop's LR schedule and per-step backend calls."""

from __future__ import annotations

import asyncio
import itertools
import json
import time

import pytest

from post_training_cookbook.grpo import loop
from post_training_cookbook.grpo.config import GRPOConfig, policy_loss
from post_training_cookbook.grpo.loop import apply_train_step, get_lr_cosine, training_loop
from post_training_cookbook.run_artifacts import Run

from ._fakes import MockSession, _base_config


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {},
            {
                "type": "LOSS_TYPE_GRPO",
                "grpo_params": {
                    "agg_type": "GRPO_LOSS_AGGREGATION_TYPE_FIXED_HORIZON",
                    "ratio_type": "GRPO_LOSS_RATIO_TYPE_TOKEN",
                },
            },
        ),
        ({"loss": "ppo"}, {"type": "LOSS_TYPE_PPO"}),
        (
            {"loss": "grpo", "ratio_type": "GRPO_LOSS_RATIO_TYPE_SEQUENCE"},
            {
                "type": "LOSS_TYPE_GRPO",
                "grpo_params": {
                    "agg_type": "GRPO_LOSS_AGGREGATION_TYPE_FIXED_HORIZON",
                    "ratio_type": "GRPO_LOSS_RATIO_TYPE_SEQUENCE",
                },
            },
        ),
        ({"loss": "cispo"}, {"type": "LOSS_TYPE_CISPO"}),
        (
            {"loss": "cispo", "clip_high_threshold": 1.5},
            {
                "type": "LOSS_TYPE_CISPO",
                "cispo_params": {"clip_high_threshold": 1.5},
            },
        ),
        (
            {"loss": "ppo", "clip_low_threshold": 0.8, "clip_high_threshold": 1.28},
            {
                "type": "LOSS_TYPE_PPO",
                "ppo_params": {"clip_low_threshold": 0.8, "clip_high_threshold": 1.28},
            },
        ),
        ({"loss": "dppo"}, {"type": "LOSS_TYPE_DPPO"}),
        (
            {"loss": "dro", "dro_beta": 0.1},
            {"type": "LOSS_TYPE_DRO", "dro_params": {"beta": 0.1}},
        ),
        ({"loss": "importance_sampling"}, {"type": "LOSS_TYPE_IMPORTANCE_SAMPLING"}),
    ],
)
def test_policy_loss_variants(kwargs, expected):
    assert policy_loss(GRPOConfig(**kwargs)) == expected


class TestGetLrCosine:
    def test_warmup_then_decay(self):
        """LR ramps up during warmup, reaches base_lr, then decays monotonically."""
        base_lr = 1e-4
        lrs = [get_lr_cosine(s, 100, base_lr, warmup_ratio=0.1) for s in range(100)]
        # Starts near zero, not at base_lr
        assert lrs[0] < base_lr
        # Reaches base_lr at end of warmup
        assert abs(lrs[9] - base_lr) < 1e-8
        # Decays monotonically after warmup
        for i in range(10, len(lrs) - 1):
            assert lrs[i] >= lrs[i + 1] - 1e-12

    def test_final_lr(self):
        """LR approaches min_lr_ratio * base_lr at the end."""
        lr = get_lr_cosine(99, 100, base_lr=1e-4, warmup_ratio=0.1, min_lr_ratio=0.1)
        assert abs(lr - 1e-5) < 1e-6


@pytest.fixture
def run(tmp_path) -> Run:
    return Run(dir=tmp_path, wandb=None, rollouts_enabled=False)


def _samples(n: int = 1) -> list[dict]:
    return [{"id": i} for i in range(n)]


class TestTrainingLoop:
    """The loop's contract with the backend: what it submits, and through which sub-client."""

    def test_each_step_submits_backward_then_optim_step(self, run):
        session = MockSession()
        config = _base_config(learning_rate=1e-4, timeout=90.0)

        training_loop(3, lambda step, s: (_samples(), {}), session, config, run)

        assert len(session.trainer.forward_backward_calls) == 3
        assert len(session.trainer.optim_step_calls) == 3
        assert len(session.trainer.weights_sync_calls) == 3
        assert [c["timeout"] for c in session.trainer.forward_backward_calls] == [90.0] * 3

    def test_backward_forwards_policy_loss(self, run):
        session = MockSession()
        config = _base_config(loss="ppo", clip_high_threshold=1.2)

        training_loop(1, lambda step, s: (_samples(2), {}), session, config, run)

        call = session.trainer.forward_backward_calls[0]
        assert call["loss"] == policy_loss(config)
        assert len(call["samples"]) == 2

    def test_optim_step_publishes_weights_synchronously(self, run):
        """Cookbook loops sample right after the update, so the sync must be inline."""
        session = MockSession()

        training_loop(1, lambda step, s: (_samples(), {}), session, _base_config(), run)

        assert session.trainer.weights_sync_calls == [
            {
                "weight_sync_type": "WEIGHT_SYNC_TYPE_SYNCHRONOUS",
                "timeout": _base_config().timeout,
            }
        ]

    def test_empty_batch_skips_the_backend_entirely(self, run):
        """A step that collected nothing must not submit an empty gradient update."""
        session = MockSession()

        training_loop(2, lambda step, s: ([], {}), session, _base_config(), run)

        assert session.trainer.forward_backward_calls == []
        assert session.trainer.optim_step_calls == []
        assert session.trainer.weights_sync_calls == []

    def test_learning_rate_follows_the_schedule(self, run):
        session = MockSession()
        config = _base_config(learning_rate=1e-4, warmup_ratio=0.0, min_lr_ratio=0.1)

        training_loop(4, lambda step, s: (_samples(), {}), session, config, run)

        lrs = [c["adam_params"]["learning_rate"] for c in session.trainer.optim_step_calls]
        assert lrs == sorted(lrs, reverse=True), "cosine schedule should decay across steps"
        assert lrs[0] == pytest.approx(1e-4)


class TestApplyTrainStep:
    def _apply(self, session, config, run, samples=None):
        samples = _samples() if samples is None else samples
        return session.run(
            apply_train_step(session, samples, 0, 1, config, run, {}, 0.0, overall_start=time.monotonic())
        )

    @pytest.mark.parametrize(
        "weight_sync_type",
        ["WEIGHT_SYNC_TYPE_BACKGROUND_PUBLISH", "WEIGHT_SYNC_TYPE_PIPELINE"],
    )
    def test_deferred_publish_waits_until_previous_publish_is_applied(self, run, weight_sync_type):
        # Both deferred modes hand the publish to a server worker that holds the same
        # trainer lock as backward, so both must wait for it to land.
        session = MockSession()
        config = _base_config(
            weight_sync_type=weight_sync_type, use_lora=False, episodes_per_task=2, max_steps_off_policy=1
        )
        session.policy_states = iter([(True, 0, 1), (False, 0, 1), (False, 1, 1)])

        self._apply(session, config, run)

        assert session.retrieve_calls == [(True, 0), (False, 0), (False, 1)]
        assert len(session.trainer.forward_backward_calls) == 1
        assert session.trainer.weights_sync_calls[0]["weight_sync_type"] == weight_sync_type

    def test_synchronous_publish_does_not_poll(self, run):
        session = MockSession()
        self._apply(session, _base_config(), run)
        assert session.retrieve_calls == []

    def test_deferred_wait_times_out_before_backward(self, run, monkeypatch):
        session = MockSession()
        config = _base_config(
            weight_sync_type="WEIGHT_SYNC_TYPE_BACKGROUND_PUBLISH", episodes_per_task=2, max_steps_off_policy=1
        )
        monkeypatch.setattr(loop, "_PUBLISH_WAIT_TIMEOUT_S", 0.0)
        session.policy_states = itertools.repeat((True, 0, 1))

        with pytest.raises(TimeoutError, match="applied=0 target=1"):
            self._apply(session, config, run)
        assert session.trainer.forward_backward_calls == []

    def test_timeout_before_any_snapshot_still_reports(self, run, monkeypatch):
        """The bounded wait must survive a session read that never returns."""
        session = MockSession()
        config = _base_config(
            weight_sync_type="WEIGHT_SYNC_TYPE_BACKGROUND_PUBLISH", episodes_per_task=2, max_steps_off_policy=1
        )
        monkeypatch.setattr(loop, "_PUBLISH_WAIT_TIMEOUT_S", 0.01)

        async def never_returns():
            await asyncio.Event().wait()

        monkeypatch.setattr(session, "retrieve_async", never_returns)
        with pytest.raises(TimeoutError, match="no session snapshot retrieved"):
            self._apply(session, config, run)
        assert session.trainer.forward_backward_calls == []

    def test_empty_step_still_advances_checkpoints(self, run):
        session = MockSession()
        config = _base_config(inference_checkpoint_interval=1)
        self._apply(session, config, run, samples=[])
        assert session.trainer.forward_backward_calls == []
        assert session.checkpoints == ["inference"]

    def test_checkpoint_uses_async_session_method(self, run, monkeypatch):
        session = MockSession()
        monkeypatch.setattr(
            session,
            "create_inference_checkpoint",
            lambda: pytest.fail("sync checkpoint method re-enters the running session loop"),
        )
        self._apply(session, _base_config(), run)
        assert "model-1" in (run.dir / "checkpoints.jsonl").read_text()


def test_resume_matches_uninterrupted_schedule(tmp_path):
    full_dir, resumed_dir = tmp_path / "full", tmp_path / "resumed"
    full_dir.mkdir()
    resumed_dir.mkdir()
    full, resumed = MockSession(), MockSession()
    full_validation, resumed_validation, collected = [], [], []
    options = {
        "learning_rate": 0.01,
        "warmup_ratio": 0.5,
        "validation_interval": 2,
        "training_checkpoint_interval": 2,
        "inference_checkpoint_interval": 3,
    }

    def collect(step, session):
        if session is resumed:
            collected.append(step)
        return ([] if step in {1, 4} else _samples()), {}

    training_loop(6, collect, full, _base_config(**options), Run(full_dir), full_validation.append)
    training_loop(
        6, collect, resumed, _base_config(**options), Run(resumed_dir), resumed_validation.append, start_step=3
    )

    assert collected == [3, 4, 5]
    full_lrs = [call["adam_params"]["learning_rate"] for call in full.trainer.optim_step_calls]
    resumed_lrs = [call["adam_params"]["learning_rate"] for call in resumed.trainer.optim_step_calls]
    assert resumed_lrs == pytest.approx(full_lrs[2:])
    assert resumed_validation == [3, 4, 6]
    assert resumed_validation[1:] == [step for step in full_validation if step > 3]
    full_records = [json.loads(line) for line in (full_dir / "checkpoints.jsonl").read_text().splitlines()]
    resumed_records = [json.loads(line) for line in (resumed_dir / "checkpoints.jsonl").read_text().splitlines()]
    assert [(r["step"], r["kind"]) for r in resumed_records] == [
        (r["step"], r["kind"]) for r in full_records if r["step"] > 3
    ]
    assert [json.loads(line)["step"] for line in (resumed_dir / "metrics.jsonl").read_text().splitlines()] == [4, 5, 6]
