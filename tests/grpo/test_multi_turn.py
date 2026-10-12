"""Tests for `post_training_cookbook.grpo.multi_turn`: flattening, sample building, collector, validation."""

from __future__ import annotations

import pytest

from post_training_cookbook.envs.base import Action, CompletedEpisode, Task, TruncationCause
from post_training_cookbook.grpo.data_progress import DataProgress
from post_training_cookbook.grpo.multi_turn import (
    _StepMetrics,
    build_episode_group_samples,
    flatten_episode,
    make_multi_turn_collector,
    run_multi_turn_validation,
    run_multi_turn_validation_async,
)
from post_training_cookbook.run_artifacts import Run
from tests.grpo._fakes import (
    BreakingCodec,
    FakeCodec,
    MockSession,
    OneStepEnv,
    _base_config,
    _msg,
)


def _episode(
    action: Action,
    reward: float,
    *,
    content: str = "a",
    terminated: bool = True,
    truncation_cause: TruncationCause | None = None,
) -> CompletedEpisode:
    return CompletedEpisode(
        task=Task(prompt="q", target="a"),
        messages=[_msg("user", "q"), _msg("assistant", content)],
        actions=[action],
        tool_specs=[],
        reward=reward,
        terminated=terminated,
        truncation_cause=truncation_cause,
        num_turns=1,
    )


@pytest.fixture
def run(tmp_path) -> Run:
    """A Run that writes to a tmpdir and disables rollout dumps so tests stay quiet."""
    return Run(dir=tmp_path, wandb=None, rollouts_enabled=False)


class TestFlattenEpisodeSingleTurn:
    def test_single_turn_produces_one_sequence(self):
        codec = FakeCodec()
        messages = [_msg("user", "hi"), _msg("assistant", "bye")]
        # Action tokens must match the codec's content encoding for prefix stability
        action_tokens = codec.encode("bye")
        action_logprobs = [-0.5, -1.0, -0.8]

        result = flatten_episode(codec, messages, [Action(action_tokens, action_logprobs)])

        assert len(result) == 1
        flat = result[0]
        # Prefix: user message + gen prompt, all mask=0
        prefix = codec.encode_messages(messages[:1], add_generation_prompt=True)
        assert flat.tokens[: len(prefix)] == prefix
        assert flat.loss_mask[: len(prefix)] == [0] * len(prefix)
        # Action tokens spliced in with mask=1
        n = len(action_tokens)
        assert flat.tokens[len(prefix) : len(prefix) + n] == action_tokens
        assert flat.loss_mask[len(prefix) : len(prefix) + n] == [1] * n
        assert flat.logprobs[len(prefix) : len(prefix) + n] == action_logprobs

    def test_single_turn_logprobs_alignment(self):
        codec = FakeCodec()
        messages = [_msg("user", "q"), _msg("assistant", "a")]
        action_tokens = codec.encode("a")
        action_logprobs = [-0.3]

        result = flatten_episode(codec, messages, [Action(action_tokens, action_logprobs)])
        flat = result[0]

        assert len(flat.tokens) == len(flat.logprobs) == len(flat.loss_mask)
        # All mask=0 positions have logprob 0.0
        for i, m in enumerate(flat.loss_mask):
            if m == 0:
                assert flat.logprobs[i] == 0.0


class TestFlattenEpisodeMultiTurn:
    def test_two_turn_with_extension(self):
        """Two assistant turns, prefix extension holds -> single sequence."""
        codec = FakeCodec()
        messages = [
            _msg("user", "q"),
            _msg("assistant", "a1"),
            _msg("tool", "r1"),
            _msg("assistant", "a2"),
        ]
        actions = [
            Action(codec.encode("a1"), [-0.1, -0.2]),
            Action(codec.encode("a2"), [-0.3, -0.4]),
        ]

        result = flatten_episode(codec, messages, actions)
        assert len(result) == 1

        flat = result[0]
        assert len(flat.tokens) == len(flat.logprobs) == len(flat.loss_mask)
        # Both action regions have mask=1
        mask_1_count = sum(flat.loss_mask)
        assert mask_1_count == 4  # 2 tokens per action, 2 actions

    def test_three_turn_mask_correctness(self):
        codec = FakeCodec()
        messages = [
            _msg("user", "q"),
            _msg("assistant", "X"),
            _msg("tool", "r1"),
            _msg("assistant", "Y"),
            _msg("tool", "r2"),
            _msg("assistant", "Z"),
        ]
        actions = [
            Action(codec.encode("X"), [-0.1]),
            Action(codec.encode("Y"), [-0.2]),
            Action(codec.encode("Z"), [-0.3]),
        ]

        result = flatten_episode(codec, messages, actions)
        assert len(result) == 1
        flat = result[0]
        assert sum(flat.loss_mask) == 3


class TestPrefixBreakSplitting:
    def test_prefix_mismatch_splits(self):
        """When re-encoding changes the prefix, episodes split."""
        codec = BreakingCodec()
        messages = [
            _msg("user", "q"),
            _msg("assistant", "a1"),
            _msg("tool", "r1"),
            _msg("assistant", "a2"),
        ]
        actions = [
            Action(codec.encode("a1"), [-0.1, -0.2]),
            Action(codec.encode("a2"), [-0.3, -0.4]),
        ]

        result = flatten_episode(codec, messages, actions)
        # Should split into 2 sequences due to prefix mismatch
        assert len(result) == 2
        # Both sequences should have valid structure
        for flat in result:
            assert len(flat.tokens) == len(flat.logprobs) == len(flat.loss_mask)
        # First sequence has turn 1's action tokens
        assert sum(result[0].loss_mask) == 2

    def test_prefix_stable_codec_produces_single_output(self):
        """When encoding is prefix-stable, multi-turn produces a single sequence."""
        codec = FakeCodec()
        messages = [
            _msg("user", "q"),
            _msg("assistant", "a1"),
            _msg("tool", "r1"),
            _msg("assistant", "a2"),
        ]
        actions = [
            Action(codec.encode("a1"), [-0.1, -0.2]),
            Action(codec.encode("a2"), [-0.3, -0.4]),
        ]

        result = flatten_episode(codec, messages, actions)
        assert len(result) == 1


class TestContextAssistants:
    def test_context_assistant_is_masked_out(self):
        """Leading env-provided assistant messages get mask=0, not paired with actions."""
        codec = FakeCodec()
        messages = [
            _msg("user", "hi"),
            _msg("assistant", "env-greeting"),  # from env.reset(), not sampled
            _msg("user", "q"),
            _msg("assistant", "a1"),  # sampled
        ]
        actions = [Action(codec.encode("a1"), [-0.1, -0.2])]

        result = flatten_episode(codec, messages, actions, num_context_assistants=1)

        assert len(result) == 1
        flat = result[0]
        assert len(flat.tokens) == len(flat.logprobs) == len(flat.loss_mask)
        # Only the sampled action tokens should have mask=1
        assert sum(flat.loss_mask) == 2

    def test_context_assistant_mismatch_still_raises(self):
        """Genuine action count mismatch still raises even with context assistants."""
        codec = FakeCodec()
        messages = [
            _msg("user", "hi"),
            _msg("assistant", "env-greeting"),
            _msg("user", "q"),
            _msg("assistant", "a1"),
        ]
        # 1 context assistant + 1 sampled assistant, but 0 actions -> error
        with pytest.raises(ValueError, match="sampled assistant"):
            flatten_episode(codec, messages, [], num_context_assistants=1)

    def test_context_assistant_with_prefix_break_splits(self):
        """A leading context assistant plus a mid-stream prefix break still splits
        cleanly — each segment trains on exactly one sampled action and never on
        the env-provided context assistant."""
        codec = BreakingCodec()
        messages = [
            _msg("user", "q"),
            _msg("assistant", "env"),  # from env.reset, not sampled
            _msg("tool", "r1"),
            _msg("assistant", "a1"),  # sampled
            _msg("tool", "r2"),
            _msg("assistant", "a2"),  # sampled
        ]
        actions = [
            Action(codec.encode("a1"), [-0.1, -0.2]),
            Action(codec.encode("a2"), [-0.3, -0.4]),
        ]

        result = flatten_episode(codec, messages, actions, num_context_assistants=1)

        # Prefix break forces a split: one segment per sampled action.
        assert len(result) == 2
        for flat in result:
            assert len(flat.tokens) == len(flat.logprobs) == len(flat.loss_mask)
            # Each segment trains on exactly one action's tokens.
            assert sum(flat.loss_mask) == 2
        # Total trainable tokens equal the actions' token counts — the context
        # assistant's "env" tokens never receive mask=1, in either segment.
        total_trainable = sum(flat.loss_mask.count(1) for flat in result)
        assert total_trainable == sum(len(a.tokens) for a in actions)


class TestFlattenEpisodeValidation:
    def test_mismatched_actions_raises(self):
        codec = FakeCodec()
        messages = [_msg("user", "q"), _msg("assistant", "a")]
        with pytest.raises(ValueError, match="actions"):
            flatten_episode(codec, messages, [])

    def test_empty_conversation_raises(self):
        codec = FakeCodec()
        with pytest.raises(ValueError, match="actions"):
            flatten_episode(codec, [_msg("user", "q")], [Action([1], [-0.1])])


class TestStepMetrics:
    def test_includes_sample_counts(self):
        ep = CompletedEpisode(
            task=Task(prompt="q", target="a"),
            messages=[_msg("user", "q")],
            actions=[Action([1], [-0.1])],
            tool_specs=[],
            reward=1.0,
            terminated=True,
            truncation_cause=None,
            num_turns=1,
            metrics={},
        )
        metrics = _StepMetrics()
        metrics.add(ep)
        log = metrics.finalize(num_training_samples=3, num_groups=1, zero_variance_skipped_groups=0)
        assert log["num_training_samples"] == 3
        assert log["zero_variance_skip_ratio"] == 0.0


class TestBuildEpisodeGroupSamples:
    def test_variance_produces_samples(self):
        group = [_episode(Action([10], [-0.1]), 1.0), _episode(Action([11], [-0.2]), 0.0, content="b")]

        samples = build_episode_group_samples(group, [0.5, -0.5], FakeCodec())

        assert len(samples) >= 2

    def test_a_segment_that_trains_no_slot_is_dropped_and_counted(self, caplog):
        """An action that emitted no tokens leaves a segment with an all-zero mask.

        The builder rejects such a payload, so the segment is skipped here
        instead. The count is aggregated across the group rather than warned per
        segment, which is what this asserts alongside the surviving samples.
        """
        group = [_episode(Action([10], [-0.1]), 1.0), _episode(Action([], []), 0.0)]

        with caplog.at_level("WARNING", logger="post_training_cookbook.grpo.multi_turn"):
            samples = build_episode_group_samples(group, [0.5, -0.5], FakeCodec())

        dropped = [r for r in caplog.records if "train no slot" in r.message]
        assert len(dropped) == 1, "the drop is counted once per call, not once per segment"
        assert len(samples) == 1


class TestOverlongMasking:
    def test_truncated_episode_dropped_from_samples(self):
        codec = FakeCodec()
        ep_good = _episode(Action([10], [-0.1]), 1.0)
        ep_truncated = _episode(
            Action([11], [-0.2]),
            0.0,
            content="b",
            terminated=False,
            truncation_cause=TruncationCause.MAX_TURNS,
        )

        # With masking enabled, truncated episodes are dropped entirely
        samples = build_episode_group_samples([ep_good, ep_truncated], [0.5, -0.5], codec, overlong_masking=True)
        assert len(samples) == 1

        # Without masking, both episodes produce samples
        samples_no_mask = build_episode_group_samples(
            [ep_good, ep_truncated], [0.5, -0.5], codec, overlong_masking=False
        )
        assert len(samples_no_mask) == 2

    def test_non_truncated_unaffected_by_masking(self):
        codec = FakeCodec()
        ep = _episode(Action([10], [-0.1]), 1.0)
        ep2 = _episode(Action([11], [-0.2]), 0.0, content="b")
        samples_on = build_episode_group_samples([ep, ep2], [0.5, -0.5], codec, overlong_masking=True)
        samples_off = build_episode_group_samples([ep, ep2], [0.5, -0.5], codec, overlong_masking=False)
        masks_on = [sample["loss_fn_inputs"]["weights"]["data"] for sample in samples_on]
        masks_off = [sample["loss_fn_inputs"]["weights"]["data"] for sample in samples_off]
        assert masks_on == masks_off


def test_make_multi_turn_collector_wiring(run):
    codec = FakeCodec()
    tasks = [Task(prompt="a", target="1"), Task(prompt="b", target="2")]
    config = _base_config(batch_size=2, episodes_per_task=2)

    call_count = [0]

    def factory(t: Task):
        call_count[0] += 1
        reward = 0.2 if call_count[0] % 2 == 1 else 0.8
        return OneStepEnv(reward=reward)

    progress = DataProgress(kind="multi_turn", size=len(tasks), batch_size=config.batch_size)
    collector = make_multi_turn_collector(config, codec, tasks, factory, run, data_progress=progress)
    session = MockSession()
    samples, step_log = collector(0, session)
    assert isinstance(step_log, dict)
    assert "zero_variance_skip_ratio" in step_log
    assert step_log["zero_variance_skip_ratio"] == 0.0
    assert len(samples) >= 1
    assert step_log["num_training_samples"] == len(samples)
    # Without dynamic sampling, attempt-tracking metrics shouldn't appear.
    assert "dynamic_sampling/attempts" not in step_log


def test_dynamic_sampling_oversamples_until_kept_target(run):
    """Dynamic sampling keeps drawing batches when groups are zero-variance,
    stops once ``batch_size`` non-trivial groups are accumulated, and trims
    the trailing tail beyond the target keeper."""
    codec = FakeCodec()
    tasks = [Task(prompt=p, target="x") for p in ["A", "B", "C", "D"]]
    config = _base_config(
        batch_size=2,
        episodes_per_task=2,
        max_sampling_attempts=3,
    )

    # Episodes are created in factory-call order. With episodes_per_task=2, the
    # 8 calls map to: A(0.5,0.5)=zero-var, B(0.2,0.8)=var, C(0.1,0.9)=var, D(0.3,0.7)=var.
    rewards_by_call = {1: 0.5, 2: 0.5, 3: 0.2, 4: 0.8, 5: 0.1, 6: 0.9, 7: 0.3, 8: 0.7}
    call_count = [0]

    def factory(task: Task):
        call_count[0] += 1
        return OneStepEnv(reward=rewards_by_call[call_count[0]])

    progress = DataProgress(kind="multi_turn", size=len(tasks), batch_size=config.batch_size)
    collector = make_multi_turn_collector(config, codec, tasks, factory, run, data_progress=progress)
    samples, step_log = collector(0, MockSession())

    # Attempt 1 (A,B) yields 1 keeper; attempt 2 (C,D) reaches the target.
    assert step_log["dynamic_sampling/attempts"] == 2
    # Trim drops D (the 3rd keeper), keeping [A(skip), B(kept), C(kept)].
    assert step_log["dynamic_sampling/total_groups_sampled"] == 3
    assert step_log["dynamic_sampling/kept_groups"] == 2
    assert step_log["zero_variance_skip_ratio"] == pytest.approx(1 / 3)
    assert len(samples) >= 1


def test_dynamic_sampling_stops_at_max_attempts(run):
    """When every group is zero-variance, dynamic sampling exhausts the cap and
    returns no training samples (the step is skipped by the loop)."""
    codec = FakeCodec()
    tasks = [Task(prompt=p, target="x") for p in ["A", "B", "C", "D"]]
    config = _base_config(
        batch_size=2,
        episodes_per_task=2,
        max_sampling_attempts=2,
    )

    def factory(task: Task):
        return OneStepEnv(reward=0.5)  # every episode identical → always zero-variance

    progress = DataProgress(kind="multi_turn", size=len(tasks), batch_size=config.batch_size)
    collector = make_multi_turn_collector(config, codec, tasks, factory, run, data_progress=progress)
    samples, step_log = collector(0, MockSession())

    # Both attempts run, both produce only zero-variance groups, so no training
    # samples. The training loop's ``if samples:`` guard then skips fb/optim/logging.
    assert samples == []
    assert step_log["dynamic_sampling/attempts"] == 2
    assert step_log["dynamic_sampling/kept_groups"] == 0
    assert step_log["dynamic_sampling/total_groups_sampled"] == 4


class _BoomSession(MockSession):
    """SessionClient whose sampling always raises — mimics a broken backend."""

    def next_tokens(self) -> list[int]:
        raise RuntimeError("backend down")


def test_validation_logs_all_iterations_failed_and_returns(run, monkeypatch, caplog):
    """When every validation iteration raises, ``val/all_iterations_failed`` is logged
    and the function returns instead of crashing the training loop."""
    from post_training_cookbook.grpo import multi_turn

    logged: list[tuple[int, dict]] = []
    monkeypatch.setattr(multi_turn, "log_step", lambda r, step, m: logged.append((step, m)))

    with caplog.at_level("WARNING", logger="post_training_cookbook.grpo.multi_turn"):
        run_multi_turn_validation(
            step=42,
            config=_base_config(max_turns=1, validation_iterations=2),
            codec=FakeCodec(),
            val_tasks=[Task(prompt="q", target="a")],
            env_factory=lambda _t: OneStepEnv(),
            session=_BoomSession(),
            run=run,
        )

    metrics = [m for _, m in logged]
    assert any("val/all_iterations_failed" in m for m in metrics), "all-fail summary should be logged"
    iter_failures = [r for r in caplog.records if "validation iteration failed" in r.message]
    assert len(iter_failures) == 2, "every failed iteration should emit a warning"


def test_validation_accepts_async_training_config(run):
    """Validation overrides episodes_per_task, which is why the async loop owns that check."""
    config = _base_config(max_turns=1, episodes_per_task=2, max_steps_off_policy=1)
    envs = []

    def factory(task):
        env = OneStepEnv()
        envs.append(env)
        return env

    run_multi_turn_validation(
        step=0,
        config=config,
        codec=FakeCodec(),
        val_tasks=[Task(prompt="q", target="a")],
        env_factory=factory,
        session=MockSession(),
        run=run,
    )

    assert len(envs) == 1


@pytest.mark.asyncio
async def test_async_validation_does_not_use_sync_session_run(run, monkeypatch):
    session = MockSession()
    monkeypatch.setattr(session, "run", lambda coro: pytest.fail("validation re-entered the session event loop"))

    await run_multi_turn_validation_async(
        step=0,
        config=_base_config(max_turns=1),
        codec=FakeCodec(),
        val_tasks=[Task(prompt="q", target="a")],
        env_factory=lambda _: OneStepEnv(),
        session=session,
        run=run,
    )


@pytest.mark.parametrize("routing_replay", [True, False])
def test_routing_replay_reaches_training_samples(run, routing_replay):
    config = _base_config(batch_size=1, episodes_per_task=2, routing_replay=routing_replay)
    rewards = iter([1.0, 0.0])
    collector = make_multi_turn_collector(
        config,
        FakeCodec(),
        [Task(prompt="q", target="a")],
        lambda task: OneStepEnv(reward=next(rewards)),
        run,
        data_progress=DataProgress(kind="multi_turn", size=1, batch_size=1),
    )
    session = MockSession()
    samples, _ = collector(0, session)
    assert len(samples) == 2
    assert session.last_return_routed_experts is routing_replay
    if routing_replay:
        keys = [sample["routed_experts_key"] for sample in samples]
        assert all(keys) and len(set(keys)) == len(keys)
    else:
        assert all("routed_experts_key" not in sample for sample in samples)


def test_routing_replay_fails_loudly_without_a_generator_key(run):
    config = _base_config(batch_size=1, episodes_per_task=2, routing_replay=True)
    session = MockSession()
    session.returns_routing = False
    collector = make_multi_turn_collector(
        config,
        FakeCodec(),
        [Task(prompt="a", target="1")],
        OneStepEnv,
        run,
        data_progress=DataProgress(kind="multi_turn", size=1, batch_size=config.batch_size),
    )

    with pytest.raises(RuntimeError, match="routed_experts_key"):
        collector(0, session)


def test_routing_replay_sequence_without_key_fails_loudly():
    episode = _episode(Action([ord("a")], [-0.1]), reward=1.0)
    other = _episode(Action([ord("a")], [-0.1]), reward=0.0)

    with pytest.raises(ValueError, match="routed_experts_key"):
        build_episode_group_samples([episode, other], [0.5, -0.5], FakeCodec(), routing_replay=True)


def test_validation_does_not_request_routing(run):
    session = MockSession()
    session.returns_routing = False  # would raise if validation asked for a key

    run_multi_turn_validation(
        step=0,
        config=_base_config(max_turns=1, routing_replay=True),
        codec=FakeCodec(),
        val_tasks=[Task(prompt="q", target="a")],
        env_factory=lambda _t: OneStepEnv(),
        session=session,
        run=run,
    )
    assert session.last_return_routed_experts is False


@pytest.mark.parametrize("codec_type, expected_keys", [(FakeCodec, ["second"]), (BreakingCodec, ["first", "second"])])
@pytest.mark.parametrize("routing_replay", [True, False])
def test_episode_routing_keys_follow_sequence_splits(codec_type, expected_keys, routing_replay):
    codec = codec_type()
    episode = CompletedEpisode(
        task=Task(prompt="q", target="a"),
        messages=[_msg("user", "q"), _msg("assistant", "a"), _msg("tool", "r"), _msg("assistant", "b")],
        actions=[
            Action(codec.encode("a"), [-0.1], routed_experts_key="first"),
            Action(codec.encode("b"), [-0.2], routed_experts_key="second"),
        ],
        reward=1.0,
        terminated=True,
        truncation_cause=None,
        num_turns=2,
    )
    samples = build_episode_group_samples([episode], [0.5], codec, routing_replay=routing_replay)
    if routing_replay:
        assert [sample["routed_experts_key"] for sample in samples] == expected_keys
    else:
        assert all("routed_experts_key" not in sample for sample in samples)
        assert len(samples) == len(expected_keys)
