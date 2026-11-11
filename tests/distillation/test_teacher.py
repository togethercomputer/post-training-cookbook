"""Teacher generation: the resumable cache, failures, the budget stop and request kwargs."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from post_training_cookbook.distillation.teacher import (
    Spend,
    TeacherConfig,
    TeacherRequest,
    generate,
    load_cache,
)


class FakeClient:
    """Stands in for ``AsyncTogether``: echoes the prompt upper-cased, records every call."""

    instances: list[FakeClient] = []

    def __init__(self, *, api_key, max_retries, timeout, fail_on=(), finish_reason="stop", reasoning=None):
        self.api_key = api_key
        self.calls: list[dict] = []
        self.closed = False
        self.fail_on = set(fail_on)
        self.finish_reason = finish_reason
        self.reasoning = reasoning
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        FakeClient.instances.append(self)

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        prompt = kwargs["messages"][-1]["content"]
        if prompt in self.fail_on:
            raise RuntimeError("upstream 503")
        message = SimpleNamespace(content=prompt.upper(), reasoning_content=self.reasoning)
        usage = SimpleNamespace(model_dump=lambda: {"prompt_tokens": 1_000, "completion_tokens": 1_000})
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason=self.finish_reason)], usage=usage
        )

    async def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def _keys(monkeypatch):
    FakeClient.instances.clear()
    monkeypatch.setenv("KEY_A", "a")
    monkeypatch.setenv("KEY_B", "b")


def _requests(*prompts: str) -> list[TeacherRequest]:
    return [TeacherRequest(key=p, messages=[{"role": "user", "content": p}]) for p in prompts]


def _factory(**fake_kwargs):
    return lambda **kwargs: FakeClient(**kwargs, **fake_kwargs)


def _config(**kwargs) -> TeacherConfig:
    return TeacherConfig(**{"model": "org/Teacher", "api_key_envs": ("KEY_A",), "concurrency": 2, **kwargs})


def test_outputs_are_cached_and_a_rerun_makes_no_calls(tmp_path) -> None:
    cache = tmp_path / "teacher.jsonl"

    outputs, spend = generate(_requests("a", "b", "c"), _config(), cache, client_factory=_factory())
    assert {k: o.content for k, o in outputs.items()} == {"a": "A", "b": "B", "c": "C"}
    assert spend.calls == 3

    outputs, spend = generate(_requests("a", "b", "c"), _config(), cache, client_factory=_factory())
    assert spend.calls == 0
    assert set(outputs) == {"a", "b", "c"}
    assert len(cache.read_text().splitlines()) == 3


def test_failures_are_cached_and_retried_on_the_next_run(tmp_path) -> None:
    cache = tmp_path / "teacher.jsonl"

    outputs, spend = generate(_requests("a", "b"), _config(), cache, client_factory=_factory(fail_on={"b"}))
    assert outputs["b"].ok is False and "upstream 503" in outputs["b"].error
    assert spend.failures == 1

    outputs, spend = generate(_requests("a", "b"), _config(), cache, client_factory=_factory())
    assert spend.calls == 1
    assert outputs["b"].ok and outputs["b"].content == "B"
    assert load_cache(cache)["b"].ok  # the later line wins


def test_duplicate_keys_are_sent_once(tmp_path) -> None:
    _, spend = generate(_requests("a", "a", "b"), _config(), tmp_path / "t.jsonl", client_factory=_factory())
    assert spend.calls == 2


def test_budget_stops_generation(tmp_path) -> None:
    # Each fake call costs (1000 * 3 + 1000 * 15) / 1e6 = $0.018.
    config = _config(concurrency=1, price_per_million=(3.0, 15.0), budget_usd=0.03)
    outputs, spend = generate(_requests(*"abcdef"), config, tmp_path / "t.jsonl", client_factory=_factory())
    assert spend.calls == 2
    assert len(outputs) == 2
    assert spend.usd(config.price_per_million) == pytest.approx(0.036)


def test_each_api_key_gets_its_own_client_and_all_are_closed(tmp_path) -> None:
    config = _config(api_key_envs=("KEY_A", "KEY_B"))
    generate(_requests(*"abcdefgh"), config, tmp_path / "t.jsonl", client_factory=_factory())
    assert sorted(c.api_key for c in FakeClient.instances) == ["a", "b"]
    assert all(c.closed for c in FakeClient.instances)
    assert sum(len(c.calls) for c in FakeClient.instances) == 8


def test_request_kwargs_pass_template_kwargs_and_reasoning_effort(tmp_path) -> None:
    config = _config(chat_template_kwargs={"thinking": False}, reasoning_effort="low", max_tokens=64)
    generate(_requests("a"), config, tmp_path / "t.jsonl", client_factory=_factory())
    (call,) = FakeClient.instances[0].calls
    assert call["chat_template_kwargs"] == {"thinking": False}
    assert call["reasoning_effort"] == "low"
    assert call["max_tokens"] == 64 and call["model"] == "org/Teacher"


def test_optional_kwargs_are_left_out_when_unset(tmp_path) -> None:
    generate(_requests("a"), _config(), tmp_path / "t.jsonl", client_factory=_factory())
    (call,) = FakeClient.instances[0].calls
    assert "chat_template_kwargs" not in call and "reasoning_effort" not in call


def test_truncation_and_reasoning_are_recorded(tmp_path) -> None:
    cache = tmp_path / "t.jsonl"
    outputs, _ = generate(
        _requests("a"), _config(), cache, client_factory=_factory(finish_reason="length", reasoning="hmm")
    )
    assert outputs["a"].truncated
    assert outputs["a"].reasoning_content == "hmm"
    assert json.loads(cache.read_text())["finish_reason"] == "length"


def test_a_missing_api_key_fails_loudly(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("KEY_A")
    with pytest.raises(RuntimeError, match="KEY_A is not set"):
        generate(_requests("a"), _config(), tmp_path / "t.jsonl", client_factory=_factory())


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="concurrency"):
        TeacherConfig(model="m", concurrency=0)
    with pytest.raises(ValueError, match="api_key_envs"):
        TeacherConfig(model="m", api_key_envs=())
    with pytest.raises(ValueError, match="price_per_million"):
        TeacherConfig(model="m", budget_usd=1.0)


def test_spend_without_a_price_is_zero() -> None:
    assert Spend(input_tokens=10, output_tokens=10).usd(None) == 0.0
