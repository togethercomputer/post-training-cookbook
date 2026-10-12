"""Deterministic integration checks for the bounded rollout worker pool."""

from __future__ import annotations

import asyncio
import itertools
import json
from types import SimpleNamespace

import pytest

from post_training_cookbook.envs.base import Action, PolicySpan, Task
from post_training_cookbook.grpo.async_loop import _drift_ratio, _QueuedGroup, async_training_loop
from post_training_cookbook.grpo.data_progress import DataProgress
from post_training_cookbook.grpo.episode import collect_episodes
from post_training_cookbook.grpo.loop import get_lr_cosine
from post_training_cookbook.run_artifacts import Run
from tests.grpo._fakes import FakeCodec, MockSession, OneStepEnv, _base_config


@pytest.fixture
def run(tmp_path):
    return Run(dir=tmp_path, wandb=None, rollouts_enabled=False)


@pytest.mark.asyncio
async def test_rollouts_overlap_backward_and_full_stale_queue_refills(monkeypatch, run):
    session = MockSession(response_tokens=[ord("x"), ord("y")])
    config = _base_config(batch_size=2, episodes_per_task=2, groups_in_flight=1, max_steps_off_policy=0)
    tasks = [Task(prompt=str(i), target="a") for i in range(10)]
    scheduled = []
    closed = []
    queued_and_blocked = asyncio.Event()
    backward_started = asyncio.Event()
    logs = []

    class RecordedEnv(OneStepEnv):
        async def close(self):
            closed.append(self)
            if len(closed) == 8:
                assert backward_started.is_set()
                queued_and_blocked.set()

    def factory(task):
        scheduled.append(task.prompt)
        return RecordedEnv(reward=float(len(scheduled) % 2))

    async def backward(**kwargs):
        if not session.trainer.forward_backward_calls:
            backward_started.set()
            # One group fills the queue and the next worker put blocks on capacity.
            await queued_and_blocked.wait()
        return session.trainer.forward_backward(**kwargs)

    monkeypatch.setattr(session.trainer, "forward_backward_async", backward)
    monkeypatch.setattr("post_training_cookbook.grpo.loop.log_step", lambda run, step, data: logs.append(data))
    async with asyncio.timeout(5):
        await async_training_loop(2, session, config, FakeCodec(), tasks, factory, run, data_progress=_fresh())

    assert len(session.trainer.forward_backward_calls) == 2
    assert all(len(call["samples"]) == 4 for call in session.trainer.forward_backward_calls)
    assert logs[1]["async/dropped_stale_groups"] >= 2
    assert all(log["async/staleness_max"] == 0 for log in logs)
    assert len(closed) == len(scheduled)


@pytest.mark.asyncio
async def test_worker_failure_closes_siblings_before_propagating(run):
    started = asyncio.Event()
    closed = []
    tasks = [Task(prompt="blocked", target="a"), Task(prompt="failure", target="a")]

    class FailingEnv(OneStepEnv):
        async def reset(self, task):
            self.name = task.prompt
            if task.prompt == "blocked":
                started.set()
                await asyncio.Event().wait()
            await started.wait()
            raise RuntimeError("worker failed")

        async def close(self):
            await asyncio.sleep(0)
            closed.append(self.name)

    async with asyncio.timeout(5):
        # A lone failure is unwrapped, so recipes see the same shape as lockstep.
        with pytest.raises(RuntimeError, match="worker failed"):
            await async_training_loop(
                1,
                MockSession(),
                _base_config(episodes_per_task=2, groups_in_flight=2, max_steps_off_policy=1),
                FakeCodec(),
                tasks,
                lambda _: FailingEnv(),
                run,
                data_progress=_fresh(),
            )
    assert sorted(closed) == ["blocked", "blocked", "failure", "failure"]


@pytest.mark.asyncio
async def test_loop_exit_cancels_active_episodes(monkeypatch, run):
    session = MockSession()
    started = asyncio.Event()
    created = []
    closed = []

    class BlockingEnv(OneStepEnv):
        async def reset(self, task):
            if len(created) > 2:
                started.set()
                await asyncio.Event().wait()
            return await super().reset(task)

        async def close(self):
            await asyncio.sleep(0)
            closed.append(self)

    def factory(task):
        env = BlockingEnv(reward=float(len(created) % 2))
        created.append(env)
        return env

    async def backward(**kwargs):
        await started.wait()
        return session.trainer.forward_backward(**kwargs)

    monkeypatch.setattr(session.trainer, "forward_backward_async", backward)
    async with asyncio.timeout(5):
        await async_training_loop(
            1,
            session,
            _base_config(episodes_per_task=2, max_steps_off_policy=1),
            FakeCodec(),
            [Task(prompt="a", target="b")],
            factory,
            run,
            data_progress=_fresh(),
        )
    assert len(created) > 2
    assert set(closed) == set(created)


@pytest.mark.asyncio
async def test_validation_runs_with_workers_closed_then_workers_resume(monkeypatch, run):
    session = MockSession()
    config = _base_config(episodes_per_task=2, max_steps_off_policy=1, validation_interval=1)
    tasks = [Task(prompt="train", target="a")]
    active = set()
    created = 0
    validation_steps = []
    validating = False
    sample = session.generator.sample_async

    class TrackedEnv(OneStepEnv):
        async def reset(self, task):
            active.add(self)
            assert not validating
            return await super().reset(task)

        async def close(self):
            await asyncio.sleep(0)
            active.remove(self)

    def factory(task):
        nonlocal created
        created += 1
        return TrackedEnv(reward=float(created % 2))

    async def checked_sample(*args, **kwargs):
        if validating:
            assert not active
        return await sample(*args, **kwargs)

    async def validate(step):
        nonlocal validating
        assert not active
        validating = True
        validation_steps.append(step)
        await collect_episodes(session, FakeCodec(), tasks, config, lambda _: OneStepEnv())
        validating = False

    monkeypatch.setattr(session.generator, "sample_async", checked_sample)
    async with asyncio.timeout(5):
        await async_training_loop(
            2, session, config, FakeCodec(), tasks, factory, run, validate, data_progress=_fresh()
        )
    assert validation_steps == [0, 1, 2]
    assert len(session.trainer.optim_step_calls) == 2
    assert not active


@pytest.mark.asyncio
async def test_zero_variance_groups_are_replaced_until_batch_is_full(monkeypatch, run):
    session = MockSession()
    logs = []
    created = 0

    def factory(task):
        nonlocal created
        created += 1
        return OneStepEnv(reward=0.5 if created <= 2 else float(created % 2))

    monkeypatch.setattr("post_training_cookbook.grpo.loop.log_step", lambda run, step, data: logs.append(data))
    async with asyncio.timeout(5):
        await async_training_loop(
            1,
            session,
            _base_config(batch_size=2, episodes_per_task=2, max_steps_off_policy=1),
            FakeCodec(),
            [Task(prompt="a", target="b")],
            factory,
            run,
            data_progress=_fresh(),
        )
    assert logs[0]["async/dropped_zero_variance_groups"] == 1
    assert len(session.trainer.forward_backward_calls[0]["samples"]) == 4


@pytest.mark.asyncio
async def test_variance_free_rewards_end_the_step_instead_of_spinning(monkeypatch, run):
    """A reward function that separates nothing must end the step, not refill forever."""
    session = MockSession()
    logs = []
    monkeypatch.setattr("post_training_cookbook.grpo.loop.log_step", lambda run, step, data: logs.append(data))

    async with asyncio.timeout(5):
        await async_training_loop(
            1,
            session,
            _base_config(batch_size=2, episodes_per_task=2, max_steps_off_policy=1),
            FakeCodec(),
            [Task(prompt="a", target="b")],
            lambda _: OneStepEnv(reward=0.0),
            run,
            data_progress=_fresh(),
        )

    # batch_size * max_sampling_attempts groups examined, none acceptable.
    assert logs[0]["async/dropped_zero_variance_groups"] == 2
    assert logs[0]["async/accepted_groups"] == 0
    assert session.trainer.forward_backward_calls == []
    # The discarded groups still reach the reward metrics and the episode export.
    assert logs[0]["reward_mean"] == 0.0
    assert logs[0]["zero_variance_skip_ratio"] == 1.0


@pytest.mark.asyncio
async def test_group_collected_one_step_earlier_is_trained_on(monkeypatch, run):
    """The point of the feature: a group one optimizer step stale is still used."""
    session = MockSession()
    logs = []
    step_0_trained = asyncio.Event()
    envs_created = 0

    def factory(task):
        nonlocal envs_created
        envs_created += 1
        # The first group collects immediately and is trained at step 0. The second
        # is scheduled at version 0 too (the worker reads the version before the
        # optimizer step lands) but only completes afterwards, so it arrives stale.
        return OneStepEnv(reward=float(envs_created % 2), gate=None if envs_created <= 2 else step_0_trained)

    real_backward = session.trainer.forward_backward_async

    async def backward(**kwargs):
        result = await real_backward(**kwargs)
        step_0_trained.set()
        return result

    monkeypatch.setattr(session.trainer, "forward_backward_async", backward)
    monkeypatch.setattr("post_training_cookbook.grpo.loop.log_step", lambda run, step, data: logs.append(data))

    async with asyncio.timeout(5):
        await async_training_loop(
            2,
            session,
            _base_config(batch_size=1, episodes_per_task=2, groups_in_flight=1, max_steps_off_policy=1),
            FakeCodec(),
            [Task(prompt="a", target="b")],
            factory,
            run,
            data_progress=_fresh(),
        )

    assert logs[0]["async/staleness_max"] == 0
    assert logs[1]["async/staleness_max"] == 1
    assert logs[1]["async/dropped_stale_groups"] == 0
    assert len(session.trainer.forward_backward_calls) == 2


class TestDriftRatio:
    """Unit coverage for the pure drift calculation behind async/mid_rollout_drift_ratio."""

    @staticmethod
    def _group(*actions: Action) -> _QueuedGroup:
        episode = SimpleNamespace(actions=list(actions))
        return _QueuedGroup(SimpleNamespace(group=[episode]), scheduled_version=0, task_index=0)

    def test_no_segments_is_excluded_rather_than_counted_on_policy(self):
        """An unreported action must not dilute the ratio toward 0.0."""
        assert _drift_ratio([self._group(Action([1, 2], [0.0, 0.0]))]) == 0.0
        drifted = Action([1, 2], [0.0, 0.0], (PolicySpan(0, 1),))
        assert _drift_ratio([self._group(Action([1, 2], [0.0, 0.0]), drifted)]) == 1.0

    def test_split_action_counts_only_the_newer_span(self):
        action = Action([1, 2, 3, 4], [0.0] * 4, (PolicySpan(0, 0), PolicySpan(2, 1)))
        assert _drift_ratio([self._group(action)]) == 0.5

    def test_whole_action_on_a_newer_policy_is_fully_drifted(self):
        """A publish between turns leaves a later action wholly off-policy."""
        action = Action([1, 2], [0.0, 0.0], (PolicySpan(0, 3),))
        assert _drift_ratio([self._group(action)]) == 1.0

    def test_denominator_starts_at_the_first_span(self):
        """Tokens before the first span are outside the measurement, not on-policy."""
        action = Action([1, 2, 3, 4], [0.0] * 4, (PolicySpan(2, 1),))
        assert _drift_ratio([self._group(action)]) == 1.0

    def test_unordered_spans_are_sorted_before_pairing(self):
        action = Action([1, 2, 3, 4], [0.0] * 4, (PolicySpan(2, 1), PolicySpan(0, 0)))
        assert _drift_ratio([self._group(action)]) == 0.5


def _fresh() -> DataProgress:
    """Fresh task leases; claims ignore the source fields, which only guard restores."""
    return DataProgress(kind="multi_turn_async", size=1, batch_size=1)


@pytest.mark.asyncio
async def test_resume_keeps_policy_versions_relative(run):
    session = MockSession()
    config = _base_config(
        episodes_per_task=2,
        groups_in_flight=1,
        max_steps_off_policy=0,
        validation_interval=2,
        training_checkpoint_interval=2,
    )
    progress = _fresh()
    for _ in range(7):
        progress.consume(progress.claim())
    rewards = itertools.cycle([0.0, 1.0])
    validations = []

    async def validate(step):
        validations.append(step)

    async with asyncio.timeout(5):
        await async_training_loop(
            10,
            session,
            config,
            FakeCodec(),
            [Task(prompt="q", target="a")],
            lambda _: OneStepEnv(reward=next(rewards)),
            run,
            validate,
            data_progress=progress,
            start_step=7,
        )

    assert len(session.trainer.optim_step_calls) == 3
    assert [call["adam_params"]["learning_rate"] for call in session.trainer.optim_step_calls] == pytest.approx(
        [get_lr_cosine(step, 10, config.learning_rate, config.warmup_ratio, config.min_lr_ratio) for step in [7, 8, 9]]
    )
    logs = [json.loads(line) for line in (run.dir / "metrics.jsonl").read_text().splitlines()]
    assert [record["step"] for record in logs] == [8, 9, 10]
    assert all(record["async/staleness_max"] == 0 for record in logs)
    assert validations == [7, 8, 10]
    checkpoints = [json.loads(line) for line in (run.dir / "checkpoints.jsonl").read_text().splitlines()]
    assert [r["step"] for r in checkpoints if r["kind"] == "training"] == [8, 10]


@pytest.mark.asyncio
async def test_resumed_empty_batches_advance_client_step(run):
    session = MockSession()
    config = _base_config(episodes_per_task=2, max_steps_off_policy=0, training_checkpoint_interval=1)
    await async_training_loop(
        9,
        session,
        config,
        FakeCodec(),
        [Task(prompt="q", target="a")],
        lambda _: OneStepEnv(reward=0),
        run,
        data_progress=_fresh(),
        start_step=7,
    )
    assert session.trainer.optim_step_calls == []
    records = [json.loads(line) for line in (run.dir / "checkpoints.jsonl").read_text().splitlines()]
    assert [r["step"] for r in records if r["kind"] == "training"] == [8, 9]


@pytest.mark.asyncio
async def test_async_loop_relays_routing_keys(monkeypatch, run):
    session = MockSession(response_tokens=[ord("x"), ord("y")])
    config = _base_config(batch_size=1, episodes_per_task=2, max_steps_off_policy=0, routing_replay=True)
    created = []

    def factory(task):
        created.append(task.prompt)
        return OneStepEnv(reward=float(len(created) % 2))

    monkeypatch.setattr("post_training_cookbook.grpo.loop.log_step", lambda run, step, data: None)
    async with asyncio.timeout(5):
        await async_training_loop(
            1, session, config, FakeCodec(), [Task(prompt="t", target="a")], factory, run, data_progress=_fresh()
        )

    (call,) = session.trainer.forward_backward_calls
    assert session.last_return_routed_experts is True
    assert call["samples"] and all(sample["routed_experts_key"] for sample in call["samples"])
