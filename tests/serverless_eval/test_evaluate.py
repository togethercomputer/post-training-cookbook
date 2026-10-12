"""Tests for evaluate / _run_episode aggregation and lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from post_training_cookbook.envs.base import CompletedEpisode, Env, StepResult, Task, TruncationCause
from post_training_cookbook.serverless_eval.harness import _numeric_metrics, _summarize, evaluate
from post_training_cookbook.types import Message, ToolSpec


class _FakeEnv(Env):
    def __init__(self, *, fail_close: bool = False, max_steps: int = 1) -> None:
        self._messages: list[Message] = []
        self._steps = 0
        self._max_steps = max_steps
        self.fail_close = fail_close
        self.closed = False

    async def reset(self, task: Task) -> list[Message]:
        self._messages = [Message(role="user", content=task.prompt)]
        return list(self._messages)

    async def step(self, message: Message) -> StepResult:
        self._messages.append(message)
        self._steps += 1
        done = self._steps >= self._max_steps
        return StepResult(
            terminated=done,
            truncated=False,
            messages=list(self._messages),
            reward=1.0 if done else 0.0,
            metrics={"accuracy": 1.0 if done else 0.0},
        )

    @property
    def tool_specs(self) -> list[ToolSpec]:
        return []

    @property
    def messages(self) -> list[Message]:
        return list(self._messages)

    async def close(self) -> None:
        self.closed = True
        if self.fail_close:
            raise RuntimeError("close boom")


@dataclass
class _FakeChoiceMessage:
    content: str | None = "ok"
    reasoning: str | None = None
    tool_calls: list[Any] | None = None


@dataclass
class _FakeChoice:
    message: _FakeChoiceMessage
    finish_reason: str | None = "stop"


@dataclass
class _FakeResponse:
    choices: list[_FakeChoice]


class _FakeCompletions:
    def __init__(self, *, fail_on: int | None = None, finish_reason: str | None = "stop") -> None:
        self.calls = 0
        self.fail_on = fail_on
        self.finish_reason = finish_reason

    async def create(self, **kwargs: Any) -> _FakeResponse:
        self.calls += 1
        if self.fail_on is not None and self.calls == self.fail_on:
            raise RuntimeError("endpoint boom")
        choice = _FakeChoice(message=_FakeChoiceMessage(content="ok"), finish_reason=self.finish_reason)
        return _FakeResponse(choices=[choice])


class _FakeChat:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.completions = completions


class _FakeClient:
    def __init__(self, *, fail_on: int | None = None, finish_reason: str | None = "stop") -> None:
        self.chat = _FakeChat(_FakeCompletions(fail_on=fail_on, finish_reason=finish_reason))

    async def close(self) -> None:
        return None


def _episode(metrics: dict) -> CompletedEpisode:
    return CompletedEpisode(
        task=Task(prompt="p", target="t"),
        messages=[],
        reward=metrics.get("accuracy", 0.0),
        terminated=True,
        truncation_cause=None,
        num_turns=1,
        metrics=metrics,
    )


def test_numeric_metrics_collects_numeric_keys_and_skips_bool():
    episodes = [
        _episode({"accuracy": 1.0, "tool_calls": 2.0, "flag": True}),
        _episode({"accuracy": 0.0, "tool_calls": 0.0}),
    ]
    assert _numeric_metrics(episodes) == {"accuracy": [1.0, 0.0], "tool_calls": [2.0, 0.0]}


def test_summary_reports_each_metric_key_denominator():
    summary = _summarize([_episode({"accuracy": 1.0, "tool_calls": 2.0}), _episode({"accuracy": 0.0})], 0)

    assert summary.metric_means == {"accuracy": 0.5, "tool_calls": 2.0}
    assert summary.metric_counts == {"accuracy": 2, "tool_calls": 1}
    assert summary.num_episodes == 2


@pytest.mark.asyncio
async def test_evaluate_terminates_and_aggregates():
    closed: list[_FakeEnv] = []

    def factory(_task: Task) -> _FakeEnv:
        env = _FakeEnv()
        closed.append(env)
        return env

    summary = await evaluate(
        client=_FakeClient(),
        model="m",
        tasks=[Task(prompt="a", target="t"), Task(prompt="b", target="t")],
        env_factory=factory,
        max_turns=2,
    )

    assert summary.num_episodes == 2
    assert summary.num_failed == 0
    assert summary.reward_mean == 1.0
    assert summary.terminated_ratio == 1.0
    assert all(env.closed for env in closed)


@pytest.mark.asyncio
async def test_evaluate_max_turns_truncation():
    summary = await evaluate(
        client=_FakeClient(),
        model="m",
        tasks=[Task(prompt="a", target="t")],
        env_factory=lambda _task: _FakeEnv(max_steps=99),
        max_turns=2,
    )

    assert summary.episodes[0].truncation_cause == TruncationCause.MAX_TURNS
    assert summary.episodes[0].num_turns == 2
    assert summary.terminated_ratio == 0.0


@pytest.mark.asyncio
async def test_evaluate_counts_failed_episodes():
    summary = await evaluate(
        client=_FakeClient(fail_on=1),
        model="m",
        tasks=[Task(prompt="a", target="t"), Task(prompt="b", target="t")],
        env_factory=lambda _task: _FakeEnv(),
        max_turns=1,
    )

    assert summary.num_failed == 1
    assert summary.num_episodes == 1


@pytest.mark.asyncio
async def test_evaluate_closes_env_even_when_close_raises():
    env_holder: list[_FakeEnv] = []

    def factory(_task: Task) -> _FakeEnv:
        env = _FakeEnv(fail_close=True)
        env_holder.append(env)
        return env

    summary = await evaluate(
        client=_FakeClient(),
        model="m",
        tasks=[Task(prompt="a", target="t")],
        env_factory=factory,
        max_turns=1,
    )

    assert summary.num_episodes == 1
    assert env_holder[0].closed


@pytest.mark.asyncio
async def test_evaluate_raises_when_every_episode_fails():
    with pytest.raises(RuntimeError, match="all 1 episodes failed"):
        await evaluate(
            client=_FakeClient(fail_on=1),
            model="m",
            tasks=[Task(prompt="a", target="t")],
            env_factory=lambda _task: _FakeEnv(),
            max_turns=1,
        )


@pytest.mark.asyncio
async def test_evaluate_rejects_zero_max_turns():
    with pytest.raises(ValueError, match="max_turns must be >= 1"):
        await evaluate(
            client=_FakeClient(),
            model="m",
            tasks=[Task(prompt="a", target="t")],
            env_factory=lambda _task: _FakeEnv(),
            max_turns=0,
        )


@pytest.mark.asyncio
async def test_evaluate_rejects_empty_tasks():
    with pytest.raises(ValueError, match="tasks must not be empty"):
        await evaluate(
            client=_FakeClient(),
            model="m",
            tasks=[],
            env_factory=lambda _task: _FakeEnv(),
            max_turns=1,
        )


def test_summarize_rejects_empty_episodes():
    with pytest.raises(ValueError, match="zero episodes"):
        _summarize([], 0)


@pytest.mark.asyncio
async def test_evaluate_rejects_zero_max_concurrency():
    with pytest.raises(ValueError, match="max_concurrency must be >= 1"):
        await evaluate(
            client=_FakeClient(),
            model="m",
            tasks=[Task(prompt="a", target="t")],
            env_factory=lambda _task: _FakeEnv(),
            max_turns=1,
            max_concurrency=0,
        )


@pytest.mark.asyncio
async def test_evaluate_records_length_capped_turns():
    summary = await evaluate(
        client=_FakeClient(finish_reason="length"),
        model="m",
        tasks=[Task(prompt="a", target="t")],
        env_factory=lambda _task: _FakeEnv(),
        max_turns=1,
    )

    assert summary.metric_means["eval/length_capped"] == 1.0


def test_numeric_metrics_skips_non_numeric_values_per_episode():
    episodes = [
        _episode({"accuracy": 1.0, "note": "n/a"}),
        _episode({"accuracy": None, "note": 2.0}),
        _episode({"accuracy": 0.0, "flag": True}),
    ]
    assert _numeric_metrics(episodes) == {"accuracy": [1.0, 0.0], "note": [2.0]}


class _BoomStepEnv(_FakeEnv):
    async def step(self, message: Message) -> StepResult:
        raise KeyError("bad tool args")


@pytest.mark.asyncio
async def test_evaluate_scores_env_step_exceptions_as_env_truncation():
    closed: list[_BoomStepEnv] = []

    def factory(_task: Task) -> _BoomStepEnv:
        env = _BoomStepEnv()
        closed.append(env)
        return env

    summary = await evaluate(
        client=_FakeClient(),
        model="m",
        tasks=[Task(prompt="a", target="t")],
        env_factory=factory,
        max_turns=1,
    )

    assert summary.num_failed == 0
    assert summary.num_episodes == 1
    assert summary.reward_mean == 0.0
    assert summary.episodes[0].truncation_cause == TruncationCause.ENV
    assert closed[0].closed
