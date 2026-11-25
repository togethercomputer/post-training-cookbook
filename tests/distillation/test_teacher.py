"""Teacher generation: the resumable cache, failures, the budget stop and request kwargs."""

from __future__ import annotations

import asyncio
import json
from functools import partial
from types import SimpleNamespace

import httpx
import pytest
from together import APIConnectionError

from post_training_cookbook.distillation import teacher
from post_training_cookbook.distillation.teacher import (
    Spend,
    TeacherConfig,
    TeacherRequest,
    generate,
    load_cache,
)

# Every FakeClient built in the current test, cleared by the autouse fixture.
CLIENTS: list[FakeClient] = []


class FakeClient:
    """Stands in for ``AsyncTogether``: echoes the prompt upper-cased, records every call."""

    def __init__(self, *, api_key, max_retries, timeout, fail_on=(), bug_on=(), finish_reason="stop", reasoning=None):
        self.api_key = api_key
        self.calls: list[dict] = []
        self.closed = False
        self.fail_on = set(fail_on)
        self.bug_on = set(bug_on)
        self.finish_reason = finish_reason
        self.reasoning = reasoning
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        CLIENTS.append(self)

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        await asyncio.sleep(0)
        prompt = kwargs["messages"][-1]["content"]
        if prompt in self.fail_on:
            raise APIConnectionError(message="upstream 503", request=httpx.Request("POST", "https://api.together.xyz"))
        if prompt in self.bug_on:
            raise KeyError("a bug, not a transport error")
        message = SimpleNamespace(content=prompt.upper(), reasoning_content=self.reasoning)
        usage = SimpleNamespace(model_dump=lambda: {"prompt_tokens": 1_000, "completion_tokens": 1_000})
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason=self.finish_reason)], usage=usage
        )

    async def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def _stub_teacher(monkeypatch):
    CLIENTS.clear()
    monkeypatch.setenv("KEY_A", "a")
    monkeypatch.setenv("KEY_B", "b")
    monkeypatch.setattr(teacher, "AsyncTogether", FakeClient)


def _requests(*prompts: str) -> list[TeacherRequest]:
    return [TeacherRequest(key=p, messages=[{"role": "user", "content": p}]) for p in prompts]


def _config(**kwargs) -> TeacherConfig:
    return TeacherConfig(**{"model": "org/Teacher", "api_key_envs": ("KEY_A",), "concurrency": 2, **kwargs})


def test_outputs_are_cached_and_a_rerun_makes_no_calls(tmp_path) -> None:
    cache = tmp_path / "teacher.jsonl"

    outputs, spend = generate(_requests("a", "b", "c"), _config(), cache)
    assert {k: o.content for k, o in outputs.items()} == {"a": "A", "b": "B", "c": "C"}
    assert spend.calls == 3

    outputs, spend = generate(_requests("a", "b", "c"), _config(), cache)
    assert spend.calls == 0
    assert set(outputs) == {"a", "b", "c"}
    assert len(cache.read_text().splitlines()) == 3


def test_failures_are_cached_and_retried_on_the_next_run(tmp_path, monkeypatch) -> None:
    cache = tmp_path / "teacher.jsonl"

    with monkeypatch.context() as patch:
        patch.setattr(teacher, "AsyncTogether", partial(FakeClient, fail_on={"b"}))
        outputs, spend = generate(_requests("a", "b"), _config(), cache)
    assert not outputs["b"].ok and "upstream 503" in outputs["b"].error
    assert spend.failures == 1

    outputs, spend = generate(_requests("a", "b"), _config(), cache)
    assert spend.calls == 1
    assert outputs["b"].ok and outputs["b"].content == "B"
    assert load_cache(cache)["b"].ok  # the later line wins


def test_duplicate_keys_are_sent_once(tmp_path) -> None:
    _, spend = generate(_requests("a", "a", "b"), _config(), tmp_path / "t.jsonl")
    assert spend.calls == 2


def test_budget_stops_generation(tmp_path) -> None:
    # Each fake call costs (1000 * 3 + 1000 * 15) / 1e6 = $0.018.
    config = _config(concurrency=1, price_per_million=(3.0, 15.0), budget_usd=0.03)
    outputs, spend = generate(_requests(*"abcdef"), config, tmp_path / "t.jsonl")
    assert spend.calls == 2
    assert len(outputs) == 2
    assert spend.usd(config.price_per_million) == pytest.approx(0.036)


def test_budget_stop_caches_in_flight_responses(tmp_path) -> None:
    config = _config(concurrency=2, price_per_million=(3.0, 15.0), budget_usd=0.01)
    cache = tmp_path / "t.jsonl"

    outputs, spend = generate(_requests("a", "b", "c"), config, cache)

    assert set(outputs) == set(load_cache(cache)) == {"a", "b"}
    assert spend.calls == 2


def test_each_api_key_gets_its_own_client_and_all_are_closed(tmp_path) -> None:
    config = _config(api_key_envs=("KEY_A", "KEY_B"))
    generate(_requests(*"abcdefgh"), config, tmp_path / "t.jsonl")
    assert sorted(c.api_key for c in CLIENTS) == ["a", "b"]
    assert all(c.closed for c in CLIENTS)
    assert sum(len(c.calls) for c in CLIENTS) == 8


def test_request_kwargs_pass_template_kwargs_and_reasoning_effort(tmp_path) -> None:
    config = _config(chat_template_kwargs={"thinking": False}, reasoning_effort="low", max_tokens=64)
    generate(_requests("a"), config, tmp_path / "t.jsonl")
    (call,) = CLIENTS[0].calls
    assert call["chat_template_kwargs"] == {"thinking": False}
    assert call["reasoning_effort"] == "low"
    assert call["max_tokens"] == 64 and call["model"] == "org/Teacher"


def test_optional_kwargs_are_left_out_when_unset(tmp_path) -> None:
    generate(_requests("a"), _config(), tmp_path / "t.jsonl")
    (call,) = CLIENTS[0].calls
    assert "chat_template_kwargs" not in call and "reasoning_effort" not in call


def test_truncation_and_reasoning_are_recorded(tmp_path, monkeypatch) -> None:
    cache = tmp_path / "t.jsonl"
    monkeypatch.setattr(teacher, "AsyncTogether", partial(FakeClient, finish_reason="length", reasoning="hmm"))
    outputs, _ = generate(_requests("a"), _config(), cache)
    assert outputs["a"].truncated
    assert outputs["a"].reasoning_content == "hmm"
    assert json.loads(cache.read_text())["finish_reason"] == "length"


def test_a_missing_api_key_fails_loudly(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("KEY_A")
    with pytest.raises(RuntimeError, match="KEY_A is not set"):
        generate(_requests("a"), _config(), tmp_path / "t.jsonl")


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="concurrency"):
        TeacherConfig(model="m", concurrency=0)
    with pytest.raises(ValueError, match="api_key_envs"):
        TeacherConfig(model="m", api_key_envs=())
    with pytest.raises(ValueError, match="price_per_million"):
        TeacherConfig(model="m", budget_usd=1.0)


def test_spend_without_a_price_is_zero() -> None:
    assert Spend(input_tokens=10, output_tokens=10).usd(None) == 0.0


def test_unexpected_exceptions_raise_instead_of_being_cached(tmp_path, monkeypatch) -> None:
    cache = tmp_path / "t.jsonl"
    monkeypatch.setattr(teacher, "AsyncTogether", partial(FakeClient, bug_on={"a"}))
    with pytest.raises(KeyError, match="a bug"):
        generate(_requests("a"), _config(concurrency=1), cache)
    assert load_cache(cache) == {}


def test_rows_from_another_teacher_are_not_reused(tmp_path) -> None:
    cache = tmp_path / "t.jsonl"
    generate(_requests("a"), _config(model="org/Old"), cache)

    outputs, spend = generate(_requests("a"), _config(model="org/New"), cache)

    assert spend.calls == 1
    assert outputs["a"].model == "org/New"
    # Switching back reuses nothing either: the last line for the key is now the new model's.
    _, spend = generate(_requests("a"), _config(model="org/New"), cache)
    assert spend.calls == 0


def test_a_torn_last_line_keeps_earlier_rows_and_the_next_append_is_readable(tmp_path) -> None:
    cache = tmp_path / "t.jsonl"
    generate(_requests("a", "b"), _config(), cache)
    with cache.open("a", encoding="utf-8") as f:
        f.write('{"key": "c", "ok": tr')  # crash mid-write

    assert set(load_cache(cache)) == {"a", "b"}

    outputs, spend = generate(_requests("a", "b", "c"), _config(), cache)
    assert spend.calls == 1 and outputs["c"].content == "C"
    assert set(load_cache(cache)) == {"a", "b", "c"}


def test_request_meta_is_not_shared() -> None:
    first, second = TeacherRequest(key="a", messages=[]), TeacherRequest(key="b", messages=[])
    assert first.meta == {} and first.meta is not second.meta
