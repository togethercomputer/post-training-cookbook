"""Tests for the single-turn GRPO collector wiring and its sampling calls.

The collector tests bypass the real sampling/scoring pipeline by monkeypatching
``_sample_and_score_prompts``, to verify the retry/attempt-tracking machinery
rather than the scoring logic. ``TestSamplingCalls`` covers the part that
monkeypatching hides: that scoring and validation reach the generator through
``session.generator``, with the sampling params each is supposed to request.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from types import SimpleNamespace

import pytest

from post_training_cookbook.datasets.registry import DatasetConfig, PromptExample
from post_training_cookbook.grpo import single_turn as st
from post_training_cookbook.grpo.config import GRPOConfig
from post_training_cookbook.grpo.data_progress import DataProgress
from post_training_cookbook.grpo.loop import training_loop
from post_training_cookbook.grpo.samples import compute_group_advantages
from post_training_cookbook.grpo.single_turn import ScoredCompletion, make_single_turn_collector, run_validation
from post_training_cookbook.run_artifacts import Run

from ._fakes import RecordingSession


class _FakeDataset:
    def __init__(self, n: int) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n

    def select(self, indices: Iterable[int]) -> list[dict]:
        return [{"i": i} for i in indices]


class _FakeCodec:
    """Minimal codec — enough for build_prompt_example + step_log construction."""

    def encode_messages(self, messages, *, tools=None, add_generation_prompt=False) -> list[int]:
        return [1, 2, 3, 4]

    def decode(self, tokens: list[int]) -> str:
        return "decoded"


def _ds_cfg() -> DatasetConfig:
    return DatasetConfig(
        hf_name="x",
        hf_config=None,
        hf_split="train",
        format_question=lambda r: f"q{r['i']}",
        extract_target=lambda r: "x",
        compute_reward=lambda c, t: 0.0,
    )


def _config(**overrides) -> GRPOConfig:
    base = {
        "model": "test",
        "batch_size": 2,
        "group_size": 2,
        "max_model_length": 512,
        "max_sample_tokens": 128,
        "max_prompt_length": 64,
        "num_examples": 10,
        "num_val_examples": 0,
    }
    base.update(overrides)
    return GRPOConfig(**base)


def _scored(reward: float) -> ScoredCompletion:
    return ScoredCompletion(
        tokens=[10, 11, 12],
        logprobs=[-0.1, -0.1, -0.1],
        reward=reward,
        raw_reward=reward,
        metric=reward,
        penalty=0.0,
        is_truncated=False,
        response_budget=64,
        sampled_length=3,
        trainable_length=3,
        was_clipped=False,
        stop_reason="STOP_REASON_STOP",
    )


def _outcome(question: str, rewards: list[float]) -> st._PromptOutcome:
    pe = PromptExample(question=question, target="x", prompt_tokens=[1, 2, 3])
    group = [_scored(r) for r in rewards]
    return st._PromptOutcome(
        prompt_example=pe,
        scored_group=group,
        advantages=compute_group_advantages([sc.reward for sc in group]),
    )


def test_single_turn_collector_retries_until_target(tmp_path, monkeypatch):
    """Attempt 1 yields zero-variance groups → retry → attempt 2 has variance → stop."""
    config = _config(max_sampling_attempts=3)
    run = Run(dir=tmp_path, wandb=None, rollouts_enabled=False)

    batches = iter(
        [
            [_outcome("q0", [0.5, 0.5]), _outcome("q1", [0.5, 0.5])],  # both zero-variance
            [_outcome("q2", [1.0, 0.0]), _outcome("q3", [0.0, 1.0])],  # both have variance
        ]
    )

    def fake_sample_and_score(prompt_examples, session, *, codec, ds_cfg, config):
        return next(batches)

    monkeypatch.setattr(st, "_sample_and_score_prompts", fake_sample_and_score)

    progress = DataProgress(kind="single_turn", size=10, batch_size=config.batch_size)
    collector = make_single_turn_collector(
        config, _ds_cfg(), _FakeDataset(10), _FakeCodec(), run, data_progress=progress
    )
    samples, step_log = collector(0, session=SimpleNamespace(session_id="test0000"))

    assert step_log["dynamic_sampling/attempts"] == 2
    assert step_log["dynamic_sampling/total_groups_sampled"] == 4
    assert step_log["dynamic_sampling/kept_groups"] == 2
    assert step_log["zero_variance_skip_ratio"] == 0.5
    assert len(samples) > 0


def test_single_turn_collector_cap_exhausts_with_no_keepers(tmp_path, monkeypatch):
    """Every attempt produces only zero-variance groups → cap hit → empty samples."""
    config = _config(max_sampling_attempts=2)
    run = Run(dir=tmp_path, wandb=None, rollouts_enabled=False)

    def fake_sample_and_score(prompt_examples, session, *, codec, ds_cfg, config):
        return [_outcome(f"q{i}", [0.5, 0.5]) for i, _ in enumerate(prompt_examples)]

    monkeypatch.setattr(st, "_sample_and_score_prompts", fake_sample_and_score)

    progress = DataProgress(kind="single_turn", size=10, batch_size=config.batch_size)
    collector = make_single_turn_collector(
        config, _ds_cfg(), _FakeDataset(10), _FakeCodec(), run, data_progress=progress
    )
    samples, step_log = collector(0, session=SimpleNamespace(session_id="test0000"))

    # No keepers → collector returns ([], {}) without logging dynamic-sampling counters.
    assert samples == []
    assert step_log == {}


def test_single_turn_collector_no_retry_when_attempts_one(tmp_path, monkeypatch):
    """``max_sampling_attempts=1`` (default): single draw, no dynamic-sampling metrics emitted."""
    config = _config()  # max_sampling_attempts=1 by default
    run = Run(dir=tmp_path, wandb=None, rollouts_enabled=False)

    call_count = [0]

    def fake_sample_and_score(prompt_examples, session, *, codec, ds_cfg, config):
        call_count[0] += 1
        return [_outcome("q0", [1.0, 0.0]), _outcome("q1", [0.0, 1.0])]

    monkeypatch.setattr(st, "_sample_and_score_prompts", fake_sample_and_score)

    progress = DataProgress(kind="single_turn", size=10, batch_size=config.batch_size)
    collector = make_single_turn_collector(
        config, _ds_cfg(), _FakeDataset(10), _FakeCodec(), run, data_progress=progress
    )
    samples, step_log = collector(0, session=SimpleNamespace(session_id="test0000"))

    assert call_count[0] == 1
    assert "dynamic_sampling/attempts" not in step_log
    assert len(samples) > 0


class TestSamplingCalls:
    """The two real ``session.generator.sample_batch`` sites, exercised end to end.

    ``RecordingSession`` exposes sampling only under ``.sampling``, so a call that
    reached for ``sample_batch`` on the session directly fails here.
    """

    def test_scoring_requests_a_full_group_at_the_configured_temperature(self):
        session = RecordingSession()
        config = _config(group_size=4, temperature=0.9)
        prompt_examples = [PromptExample(question="q0", target="x", prompt_tokens=[1, 2, 3])]

        outcomes = st._sample_and_score_prompts(
            prompt_examples, session, codec=_FakeCodec(), ds_cfg=_ds_cfg(), config=config
        )

        assert len(outcomes) == 1
        assert session.last_num_samples == 4
        assert session.last_sampling_params["temperature"] == 0.9
        assert session.last_sampling_params["max_tokens"] == config.max_sample_tokens

    def test_validation_draws_one_greedy_sample(self, tmp_path):
        """Validation is a held-out measurement, so it must not sample stochastically."""
        session = RecordingSession()
        config = _config(temperature=0.9)
        run = Run(dir=tmp_path, wandb=None, rollouts_enabled=False)

        run_validation(
            0,
            val_dataset=[{"i": 0}, {"i": 1}],
            session=session,
            codec=_FakeCodec(),
            ds_cfg=_ds_cfg(),
            config=config,
            run=run,
        )

        assert session.last_num_samples == 1
        assert session.last_sampling_params["temperature"] == 0.0

    def test_validation_honours_an_explicit_val_temperature(self, tmp_path):
        session = RecordingSession()
        run = Run(dir=tmp_path, wandb=None, rollouts_enabled=False)

        run_validation(
            0,
            val_dataset=[{"i": 0}],
            session=session,
            codec=_FakeCodec(),
            ds_cfg=_ds_cfg(),
            config=_config(val_temperature=0.7),
            run=run,
        )

        assert session.last_sampling_params["temperature"] == 0.7


@pytest.mark.parametrize("routing_replay", [True, False])
def test_single_turn_replays_generator_keys_in_trainer_samples(tmp_path, monkeypatch, routing_replay):
    session = RecordingSession()
    config = _config(batch_size=1, routing_replay=routing_replay)
    run = Run(dir=tmp_path, wandb=None, rollouts_enabled=False)
    captured_keys = []
    sample_batch = session.generator.sample_batch

    def sample_group(*args, **kwargs):
        results = sample_batch(*args, **kwargs)
        for result in results:
            first = result.sequences[0]
            second = first.model_copy(update={"tokens": [ord("y")], "routed_experts_key": "second-key"})
            result.sequences.append(second)
            captured_keys.extend(sequence.routed_experts_key for sequence in result.sequences)
        return results

    monkeypatch.setattr(session.generator, "sample_batch", sample_group)
    ds_cfg = _ds_cfg()
    rewards = iter([1.0, 0.0])
    # Reward variability is needed for the collector to produce training samples.
    ds_cfg = replace(
        ds_cfg, compute_reward=lambda completion, target: next(rewards), compute_metric=lambda completion, target: 0.0
    )
    collector = make_single_turn_collector(
        config,
        ds_cfg,
        _FakeDataset(1),
        _FakeCodec(),
        run,
        data_progress=DataProgress(kind="single_turn", size=1, batch_size=1),
    )
    training_loop(1, collector, session, config, run)
    assert session.last_return_routed_experts is routing_replay
    (call,) = session.trainer.forward_backward_calls
    assert len(call["samples"]) == 2
    if routing_replay:
        assert [sample["routed_experts_key"] for sample in call["samples"]] == captured_keys
    else:
        assert all("routed_experts_key" not in sample for sample in call["samples"])


def test_single_turn_missing_capture_fails_before_scoring():
    session = RecordingSession()
    session.returns_routing = False
    with pytest.raises(RuntimeError, match="routed_experts_key"):
        st._sample_and_score_prompts(
            [PromptExample(question="q", target="x", prompt_tokens=[1, 2])],
            session,
            codec=_FakeCodec(),
            ds_cfg=_ds_cfg(),
            config=_config(routing_replay=True),
        )


def test_single_turn_validation_skips_capture_when_replay_enabled(tmp_path):
    session = RecordingSession()
    session.returns_routing = False
    run_validation(
        0,
        val_dataset=[{"i": 0}],
        session=session,
        codec=_FakeCodec(),
        ds_cfg=_ds_cfg(),
        config=_config(routing_replay=True),
        run=Run(dir=tmp_path, wandb=None, rollouts_enabled=False),
    )
    assert session.last_return_routed_experts is False
