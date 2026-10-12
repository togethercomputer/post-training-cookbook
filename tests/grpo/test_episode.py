"""Tests for the multi-turn episode collection layer (`post_training_cookbook.grpo.episode`)."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest
from together.post_training import PolicyVersionSegment

from post_training_cookbook.envs.base import Env, PolicySpan, StepResult, Task, TruncationCause
from post_training_cookbook.grpo.episode import _collect_single_episode, collect_episodes
from post_training_cookbook.types import Message, ToolSpec
from tests.grpo._fakes import (
    FakeCodec,
    MockSession,
    NeverDoneEnv,
    OneStepEnv,
    ParseFailEnv,
    RecordingSession,
    ScriptedEnv,
    ScriptedSession,
    _base_config,
    _sample_result,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _task() -> Task:
    return Task(prompt="q", target="a")


def _user_seed() -> list[Message]:
    return [Message(role="user", content="q")]


@pytest.mark.anyio
async def test_multi_step_env_reaches_terminal_on_step_n():
    """Env returns done=False for N-1 steps, done=True on step N."""
    n = 3

    def step_fn(turn: int, message: Message, prev: list[Message]) -> StepResult:
        history = [*prev, message]
        if turn < n:
            history = [*history, Message(role="tool", content=f"t{turn}")]
            return StepResult(terminated=False, truncated=False, messages=history, reward=0.0)
        return StepResult(terminated=True, truncated=False, messages=history, reward=1.0)

    env = ScriptedEnv(_user_seed(), step_fn)
    session = ScriptedSession(["a1", "a2", "a3"])

    ep = await _collect_single_episode(
        session, FakeCodec(), lambda _t: env, _task(), _base_config(max_turns=10), temperature=0.0
    )

    assert ep.num_turns == n
    assert len(ep.actions) == n
    assert ep.terminated is True
    assert ep.truncation_cause is None
    assert ep.truncated is False
    assert ep.reward == 1.0
    roles = [m["role"] for m in ep.messages]
    assert roles.count("assistant") == n
    assert roles.count("tool") == n - 1
    assert len(ep.metrics["timing/sample_durations"]) == n
    assert len(ep.metrics["timing/env_step_durations"]) == n


@pytest.mark.anyio
async def test_never_done_env_truncates_on_max_turns():
    """Env always returns done=False; loop stops at max_turns."""

    def step_fn(turn: int, message: Message, prev: list[Message]) -> StepResult:
        history = [*prev, message, Message(role="tool", content=f"t{turn}")]
        return StepResult(terminated=False, truncated=False, messages=history, reward=0.0)

    env = ScriptedEnv(_user_seed(), step_fn)
    session = ScriptedSession(["a1", "a2", "a3", "a4"])

    ep = await _collect_single_episode(
        session, FakeCodec(), lambda _t: env, _task(), _base_config(max_turns=3), temperature=0.0
    )

    assert ep.num_turns == 3
    assert len(ep.actions) == 3
    assert ep.terminated is False
    assert ep.truncation_cause == TruncationCause.MAX_TURNS
    assert ep.truncated is True


@pytest.mark.anyio
async def test_pre_existing_context_overflow_skips_sampling():
    """Env reset already exceeds max_model_length - min_response_tokens — no sample fires."""
    huge_content = "x" * 100  # encoded message takes ~102 tokens
    initial = [Message(role="user", content=huge_content)]
    env = ScriptedEnv(initial, lambda *_: pytest.fail("step should not be called"))
    session = ScriptedSession([])
    config = _base_config(max_model_length=50, min_response_tokens=10)

    ep = await _collect_single_episode(session, FakeCodec(), lambda _t: env, _task(), config, temperature=0.0)

    assert ep.num_turns == 0
    assert ep.actions == []
    assert ep.truncation_cause == TruncationCause.CONTEXT_OVERFLOW
    assert ep.truncated is True
    assert session.calls == 0


@pytest.mark.anyio
async def test_mid_episode_context_overflow_after_first_turn():
    """Short reset, but env injects a large tool output that pushes context over budget."""

    def step_fn(turn: int, message: Message, prev: list[Message]) -> StepResult:
        big = "y" * 200
        history = [*prev, message, Message(role="tool", content=big)]
        return StepResult(terminated=False, truncated=False, messages=history, reward=0.0)

    env = ScriptedEnv(_user_seed(), step_fn)
    session = ScriptedSession(["a1", "a2"])
    config = _base_config(max_model_length=120, min_response_tokens=10, max_turns=5)

    ep = await _collect_single_episode(session, FakeCodec(), lambda _t: env, _task(), config, temperature=0.0)

    assert ep.num_turns == 1
    assert len(ep.actions) == 1
    assert ep.truncation_cause == TruncationCause.CONTEXT_OVERFLOW


@pytest.mark.anyio
async def test_parse_failure_does_not_get_length_penalty():
    """Parse failures truncate but must not trigger the soft-overlong reward shaping."""
    codec = FakeCodec(parse_ok=lambda _tokens: False)
    env = ScriptedEnv(_user_seed(), lambda *_: pytest.fail("env.step should not run on parse failure"))
    session = ScriptedSession(["short"])
    config = replace(
        _base_config(max_turns=3),
        failed_parse_reward=-0.1,
        overlong_penalty_factor=10.0,  # large penalty — would dominate if applied
    )

    ep = await _collect_single_episode(session, codec, lambda _t: env, _task(), config, temperature=0.0)

    assert ep.truncation_cause == TruncationCause.PARSE_FAILURE
    assert ep.truncated is True
    assert ep.terminated is False
    assert ep.reward == pytest.approx(-0.1)
    assert ep.metrics["raw_reward"] == pytest.approx(-0.1)


@pytest.mark.anyio
async def test_env_truncation_does_not_get_length_penalty():
    """Env-driven truncation also bypasses the length penalty."""

    def step_fn(turn: int, message: Message, prev: list[Message]) -> StepResult:
        return StepResult(terminated=False, truncated=True, messages=[*prev, message], reward=0.5)

    env = ScriptedEnv(_user_seed(), step_fn)
    session = ScriptedSession(["short"])
    config = replace(_base_config(max_turns=3), overlong_penalty_factor=10.0)

    ep = await _collect_single_episode(session, FakeCodec(), lambda _t: env, _task(), config, temperature=0.0)

    assert ep.truncation_cause == TruncationCause.ENV
    assert ep.reward == pytest.approx(0.5)


@pytest.mark.anyio
async def test_env_factory_failure_propagates():
    """If the factory raises, the error surfaces; close() is never invoked because no env was built."""

    class _Boom(RuntimeError):
        pass

    def boom(_task):
        raise _Boom("factory failed")

    with pytest.raises(_Boom):
        await _collect_single_episode(ScriptedSession([]), FakeCodec(), boom, _task(), _base_config(), temperature=0.0)


@pytest.mark.anyio
async def test_env_close_runs_on_reset_failure():
    """When env.reset() raises, env.close() must still run."""
    closed: list[bool] = []

    class _ResetFails(Env):
        async def reset(self, task: Task) -> list[Message]:
            raise RuntimeError("reset boom")

        async def step(self, message: Message) -> StepResult:
            raise AssertionError("step should not run")

        async def close(self) -> None:
            closed.append(True)

        @property
        def tool_specs(self) -> list[ToolSpec]:
            return []

        @property
        def messages(self) -> list[Message]:
            return []

    with pytest.raises(RuntimeError, match="reset boom"):
        await _collect_single_episode(
            ScriptedSession([]), FakeCodec(), lambda _t: _ResetFails(), _task(), _base_config(), temperature=0.0
        )
    assert closed == [True]


@pytest.mark.anyio
async def test_close_exception_swallowed_after_normal_completion(caplog):
    """env.close() raising inside the finally block must not mask the episode result."""
    closed: list[bool] = []

    class _CloseFails(Env):
        def __init__(self) -> None:
            self._messages: list[Message] = list(_user_seed())

        async def reset(self, task: Task) -> list[Message]:
            return list(self._messages)

        async def step(self, message: Message) -> StepResult:
            self._messages = [*self._messages, dict(message)]
            return StepResult(terminated=True, truncated=False, messages=self._messages, reward=0.42)

        async def close(self) -> None:
            closed.append(True)
            raise RuntimeError("close boom")

        @property
        def tool_specs(self) -> list[ToolSpec]:
            return []

        @property
        def messages(self) -> list[Message]:
            return list(self._messages)

    with caplog.at_level("WARNING", logger="post_training_cookbook.envs.base"):
        ep = await _collect_single_episode(
            ScriptedSession(["a1"]), FakeCodec(), lambda _t: _CloseFails(), _task(), _base_config(), temperature=0.0
        )

    assert closed == [True]
    assert ep.terminated is True
    assert ep.reward == pytest.approx(0.42)
    assert any("close boom" in record.getMessage() for record in caplog.records)


@pytest.mark.anyio
async def test_close_exception_does_not_shadow_reset_failure():
    """When both reset and close raise, the original reset error must surface."""

    class _BothFail(Env):
        async def reset(self, task: Task) -> list[Message]:
            raise RuntimeError("reset boom")

        async def step(self, message: Message) -> StepResult:
            raise AssertionError("unreachable")

        async def close(self) -> None:
            raise RuntimeError("close boom")

        @property
        def tool_specs(self) -> list[ToolSpec]:
            return []

        @property
        def messages(self) -> list[Message]:
            return []

    with pytest.raises(RuntimeError, match="reset boom"):
        await _collect_single_episode(
            ScriptedSession([]), FakeCodec(), lambda _t: _BothFail(), _task(), _base_config(), temperature=0.0
        )


# ---------------------------------------------------------------------------
# `collect_episodes`: groups one task's `_collect_single_episode` runs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_collect_episodes_one_step():
    rewards_by_prompt = {"hi": 0.3, "ho": 0.7}
    tasks = [Task(prompt="hi", target="1"), Task(prompt="ho", target="2")]

    def factory(t: Task) -> Env:
        return OneStepEnv(reward=rewards_by_prompt[t.prompt])

    groups = await collect_episodes(MockSession(), FakeCodec(), tasks, _base_config(), factory)
    assert len(groups) == 2
    episodes = [ep for group in groups for ep in group]
    rewards = sorted(ep.reward for ep in episodes)
    assert rewards == [0.3, 0.7]
    assert all(ep.num_turns == 1 for ep in episodes)
    assert all(ep.terminated for ep in episodes)


@pytest.mark.asyncio
async def test_collect_episodes_multiple_per_task():
    tasks = [Task(prompt="a", target="1"), Task(prompt="b", target="2")]
    config = _base_config(episodes_per_task=3)

    groups = await collect_episodes(MockSession(), FakeCodec(), tasks, config, lambda _t: OneStepEnv(reward=0.5))
    assert len(groups) == 2
    assert all(len(group) == 3 for group in groups)
    assert all(ep.task == tasks[0] for ep in groups[0])
    assert all(ep.task == tasks[1] for ep in groups[1])


@pytest.mark.asyncio
async def test_collect_episodes_rejects_bad_max_turns():
    with pytest.raises(ValueError, match="max_turns"):
        await collect_episodes(
            MockSession(),
            FakeCodec(),
            [Task(prompt="a", target="1")],
            _base_config(max_turns=None),
            lambda _t: OneStepEnv(),
        )


@pytest.mark.asyncio
async def test_temperature_read_from_config():
    """``collect_episodes`` should propagate ``config.temperature`` to ``session.generator.sample_async``."""
    session = RecordingSession()
    await collect_episodes(
        session,
        FakeCodec(),
        [Task(prompt="a", target="1")],
        _base_config(temperature=0.0),
        lambda _t: OneStepEnv(),
    )
    assert session.last_sampling_params is not None
    assert session.last_sampling_params["temperature"] == 0.0


@pytest.mark.asyncio
async def test_parse_failure_messages_stored_on_episode():
    """Parse failure: snapshot should contain the parse-failed message even though env.step() was never called."""
    codec = FakeCodec(parse_ok=lambda _tokens: False)
    config = _base_config(max_turns=1)

    ep = await _collect_single_episode(MockSession(), codec, lambda _t: ParseFailEnv(), _task(), config, 0.7)

    assert ep.truncation_cause == TruncationCause.PARSE_FAILURE
    assert ep.truncated
    assert not ep.terminated
    # The parse-failed assistant message should be in the snapshot
    assert len(ep.messages) == 2  # user + assistant (from StepResult)
    assert ep.messages[-1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_overlong_penalty_applies_to_length_truncation():
    """Length-truncated (max_turns) episodes get the length-proportional penalty."""
    session = MockSession(response_tokens=[ord("x")] * 100)
    config = _base_config(
        max_turns=1,
        overlong_penalty_factor=1.0,
        overlong_buffer_len=48,
        max_model_length=120,
        max_sample_tokens=120,
    )

    ep = await _collect_single_episode(session, FakeCodec(), lambda _t: NeverDoneEnv(), _task(), config, 0.7)

    assert ep.truncation_cause == TruncationCause.MAX_TURNS
    assert ep.reward < 0.0


@pytest.mark.asyncio
async def test_overlong_penalty_disabled_keeps_step_reward():
    """When overlong_penalty is off, length-truncated episodes keep the env reward (0.0)."""
    config = _base_config(max_turns=1, overlong_penalty_factor=0.0)

    ep = await _collect_single_episode(MockSession(), FakeCodec(), lambda _t: NeverDoneEnv(), _task(), config, 0.7)

    assert ep.truncation_cause == TruncationCause.MAX_TURNS
    assert ep.reward == 0.0


@pytest.mark.asyncio
async def test_episode_preserves_policy_segments(monkeypatch):
    result = _sample_result([ord("x"), ord("y")])
    result.policy_segments = [
        PolicyVersionSegment(start_token=0, version=2),
        PolicyVersionSegment(start_token=1, version=3),
    ]
    session = MockSession()
    monkeypatch.setattr(session.generator, "sample_async", AsyncMock(return_value=result))

    episode = await _collect_single_episode(session, FakeCodec(), lambda _: OneStepEnv(), _task(), _base_config(), 0.7)

    # Converted to the env layer's own span type, which carries no SDK dependency.
    assert episode.actions[0].policy_segments == (PolicySpan(0, 2), PolicySpan(1, 3))


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_collection", [False, True])
async def test_collect_episodes_awaits_sibling_cleanup(cancel_collection):
    started = asyncio.Event()
    closed = []

    class WaitingEnv(OneStepEnv):
        async def reset(self, task):
            self.name = task.prompt
            if self.name == "blocked":
                started.set()
                await asyncio.Event().wait()
            await started.wait()
            raise RuntimeError("reset failed")

        async def close(self):
            await asyncio.sleep(0)
            closed.append(self.name)

    tasks = [Task(prompt="blocked", target="a")]
    if not cancel_collection:
        tasks.append(Task(prompt="failure", target="a"))
    collection = asyncio.create_task(
        collect_episodes(MockSession(), FakeCodec(), tasks, _base_config(), lambda _: WaitingEnv())
    )
    await started.wait()
    if cancel_collection:
        collection.cancel()
    with pytest.raises(asyncio.CancelledError if cancel_collection else RuntimeError):
        await collection
    assert sorted(closed) == sorted(task.prompt for task in tasks)
