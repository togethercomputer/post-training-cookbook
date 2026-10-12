"""Overlap checks for demos/tinker-migration/grpo_gsm8k_async.py."""

from __future__ import annotations

import asyncio
import inspect
import types
from unittest.mock import MagicMock

import pytest

from tests.demos._dataset_fakes import Rows

# Copied verbatim into the async demo so each file stays standalone; see its module docstring.
SHARED_FUNCTIONS = (
    "reward",
    "build_prompt",
    "score_group",
    "length_metrics",
    "group_advantages",
    "build_datums",
)

# The prompt recipe lives in module constants and Config defaults, which the functions close
# over — comparing sources alone would let those drift while every helper stayed identical.
SHARED_CONSTANTS = ("QUESTION_SUFFIX", "CONVO_PREFIX")


class _RecordingFuture:
    def __init__(self, value, log: list[str], name: str):
        self._value = value
        self._log = log
        self._name = name

    async def result_async(self):
        self._log.append(f"poll:{self._name}")
        await asyncio.sleep(0)
        return self._value


class _CountingSamplingClient:
    """Records the peak number of concurrently in-flight sample requests."""

    def __init__(self):
        self.in_flight = 0
        self.peak = 0

    async def sample_async(self, prompt, num_samples, sampling_params):
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        await asyncio.sleep(0)
        self.in_flight -= 1
        return types.SimpleNamespace(prompt=prompt, num_samples=num_samples)


class _FakeTrainingClient:
    def __init__(self, log: list[str]):
        self._log = log

    async def forward_backward_async(self, data, loss_fn):
        self._log.append(f"submit:fwd_bwd:{loss_fn}:{len(data)}")
        return _RecordingFuture(types.SimpleNamespace(metrics={"loss": 1.5}), self._log, "fwd_bwd")

    async def optim_step_async(self, adam_params):
        self._log.append("submit:optim")
        return _RecordingFuture(types.SimpleNamespace(metrics=None), self._log, "optim")


@pytest.mark.parametrize("name", SHARED_FUNCTIONS)
def test_shared_helpers_match_the_sync_demo(name, grpo_demo, async_demo):
    assert inspect.getsource(getattr(async_demo, name)) == inspect.getsource(getattr(grpo_demo, name))


@pytest.mark.parametrize("name", SHARED_CONSTANTS)
def test_shared_constants_match_the_sync_demo(name, grpo_demo, async_demo):
    assert getattr(async_demo, name) == getattr(grpo_demo, name)


def test_shared_config_defaults_match_the_sync_demo(grpo_demo, async_demo):
    assert vars(async_demo.Config()) == vars(grpo_demo.Config())


@pytest.mark.asyncio
async def test_sample_groups_caps_concurrent_requests(async_demo):
    client = _CountingSamplingClient()

    results = await async_demo.sample_groups(
        client, prompts=list("abcdef"), sampling_params=None, group_size=4, max_in_flight=2
    )

    assert [result.prompt for result in results] == list("abcdef")
    assert client.peak <= 2


@pytest.mark.asyncio
async def test_train_step_submits_both_before_polling(async_demo):
    log: list[str] = []
    metrics = await async_demo.train_step(_FakeTrainingClient(log), datums=["d1", "d2"], adam_params=object())

    assert metrics == {"loss": 1.5}
    assert log[:2] == ["submit:fwd_bwd:ppo:2", "submit:optim"]
    assert set(log[2:]) == {"poll:fwd_bwd", "poll:optim"}


class _AsyncTrainingClient:
    async def save_weights_and_get_sampling_client_async(self):
        return object()


class _AsyncServiceClient:
    async def create_lora_training_client_async(self, base_model, rank):
        return _AsyncTrainingClient()


# The async twin of the sync demo's eval-slice guard: an empty eval slice has to disable the
# eval rather than divide by zero, whether the run opted out via the cadence or via the size.
@pytest.mark.asyncio
@pytest.mark.parametrize(("eval_every", "eval_size"), [(0, 4), (1, 0)])
async def test_main_trains_every_batch_without_an_eval_slice(async_demo, monkeypatch, eval_every, eval_size):
    rows = Rows([{"question": str(i), "answer": str(i)} for i in range(5)])
    monkeypatch.setattr(async_demo.datasets, "load_dataset", lambda *_: {"train": rows, "test": rows})
    monkeypatch.setattr(async_demo, "get_tokenizer", lambda *_: object())
    monkeypatch.setattr(async_demo.renderers, "get_renderer", lambda *_: MagicMock())
    monkeypatch.setattr(async_demo.model_info, "get_recommended_renderer_name", lambda *_: "renderer")
    monkeypatch.setattr(async_demo.tinker, "ServiceClient", lambda: _AsyncServiceClient())
    monkeypatch.setattr(async_demo, "build_prompt", lambda _renderer, question: question)

    async def sample_groups(_client, prompts, _params, _group_size):
        return [types.SimpleNamespace(sequences=[object(), object()]) for _ in prompts]

    steps: list[int] = []

    async def train_step(*_):
        steps.append(1)
        return {}

    monkeypatch.setattr(async_demo, "sample_groups", sample_groups)
    monkeypatch.setattr(async_demo, "score_group", lambda *_: [0.0, 1.0])
    monkeypatch.setattr(async_demo, "build_datums", lambda *_: [object()])
    monkeypatch.setattr(async_demo, "length_metrics", lambda *_: {})
    monkeypatch.setattr(async_demo, "train_step", train_step)

    await async_demo.main(async_demo.Config(batch_size=1, max_steps=5, eval_every=eval_every, eval_size=eval_size))

    assert len(steps) == 5


@pytest.mark.asyncio
async def test_main_overlaps_sample_and_train(async_demo, monkeypatch):
    n_steps = 3
    rows = Rows([{"question": str(i), "answer": str(i)} for i in range(n_steps)])
    monkeypatch.setattr(async_demo.datasets, "load_dataset", lambda *_: {"train": rows, "test": rows})
    monkeypatch.setattr(async_demo, "get_tokenizer", lambda *_: object())
    monkeypatch.setattr(async_demo.renderers, "get_renderer", lambda *_: MagicMock())
    monkeypatch.setattr(async_demo.model_info, "get_recommended_renderer_name", lambda *_: "renderer")
    monkeypatch.setattr(async_demo, "build_prompt", lambda _renderer, question: question)
    monkeypatch.setattr(async_demo, "score_group", lambda *_: [0.0, 1.0])
    monkeypatch.setattr(async_demo, "build_datums", lambda *_: [object()])
    monkeypatch.setattr(async_demo, "length_metrics", lambda *_: {})

    sampling = False
    pair = asyncio.Barrier(2)
    overlapped = False
    synced_during_sample = False
    n_samples = 0
    n_trains = 0
    n_publishes = 0

    class _RecordingTrainingClient:
        async def save_weights_and_get_sampling_client_async(self):
            nonlocal synced_during_sample, n_publishes
            if sampling:
                synced_during_sample = True
            n_publishes += 1
            return object()

    class _RecordingServiceClient:
        async def create_lora_training_client_async(self, base_model, rank):
            return _RecordingTrainingClient()

    monkeypatch.setattr(async_demo.tinker, "ServiceClient", lambda: _RecordingServiceClient())

    async def sample_groups(_client, prompts, _params, _group_size):
        nonlocal n_samples, overlapped, sampling
        n_samples += 1
        sampling = True
        try:
            if n_samples > 1:
                await pair.wait()
                overlapped = True
            return [types.SimpleNamespace(sequences=[object(), object()]) for _ in prompts]
        finally:
            sampling = False

    async def train_step(*_):
        nonlocal n_trains, overlapped
        n_trains += 1
        if n_trains < n_steps:
            await pair.wait()
            overlapped = True
        return {}

    monkeypatch.setattr(async_demo, "sample_groups", sample_groups)
    monkeypatch.setattr(async_demo, "train_step", train_step)

    await async_demo.main(async_demo.Config(batch_size=1, max_steps=n_steps, eval_size=0))

    assert overlapped
    assert not synced_during_sample
    assert n_samples == n_steps
    assert n_trains == n_steps
    assert n_publishes == 1 + (n_steps - 1)


@pytest.mark.asyncio
async def test_main_refuses_a_batch_larger_than_the_train_split(async_demo, monkeypatch):
    rows = Rows([{"question": "1", "answer": "1"}])
    monkeypatch.setattr(async_demo.datasets, "load_dataset", lambda *_: {"train": rows, "test": rows})

    with pytest.raises(SystemExit, match="exceeds the 1-row train split"):
        await async_demo.main(async_demo.Config(batch_size=2, eval_size=0))
