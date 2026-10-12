"""Datum construction and training-loop checks for the sync GRPO demo."""

from __future__ import annotations

import types
from unittest.mock import MagicMock

import pytest

from tests.demos._dataset_fakes import Rows


class _Prompt:
    """Stands in for a ModelInput: knows its token count and grows by appending tokens."""

    def __init__(self, length: int):
        self.length = length

    def append(self, tokens):
        return _Prompt(self.length + len(tokens))


class _TrainingClient:
    def save_weights_and_get_sampling_client(self):
        return object()


class _ServiceClient:
    def create_lora_training_client(self, base_model, rank):
        return _TrainingClient()


@pytest.mark.parametrize(
    ("rewards", "expected"),
    [
        ([1.0, 1.0], None),  # every rollout right — nothing to learn from
        ([0.0, 0.0], None),  # every rollout wrong — likewise
        ([1.0, 0.0], [0.5, -0.5]),
    ],
)
def test_group_advantages_skips_groups_without_signal(grpo_demo, rewards, expected):
    assert grpo_demo.group_advantages(rewards) == expected


def test_length_metrics_reports_the_truncated_fraction(grpo_demo):
    results = [
        types.SimpleNamespace(
            sequences=[
                types.SimpleNamespace(tokens=[1, 2], stop_reason="stop"),
                types.SimpleNamespace(tokens=[1, 2, 3, 4], stop_reason="length"),
            ]
        )
    ]

    assert grpo_demo.length_metrics(results) == {"len/mean": 3.0, "len/truncated_frac": 0.5}


def test_build_datums_shifts_and_pads_to_the_prompt_boundary(grpo_demo, monkeypatch):
    monkeypatch.setattr(grpo_demo.tinker.types, "EncodedTextChunk", lambda tokens: tokens)
    monkeypatch.setattr(grpo_demo.tinker.types, "Datum", lambda model_input, loss_fn_inputs: loss_fn_inputs)
    sequence = types.SimpleNamespace(tokens=[10, 11, 12], logprobs=[-0.1, -0.2, -0.3])

    (inputs,) = grpo_demo.build_datums(_Prompt(4), [sequence], [0.5])

    # model_input drops the final sampled token, so every array is prompt_len - 1 + len(tokens).
    assert {len(value) for value in inputs.values()} == {6}
    assert inputs["target_tokens"] == [0, 0, 0, 10, 11, 12]
    assert inputs["logprobs"] == [0.0, 0.0, 0.0, -0.1, -0.2, -0.3]
    assert inputs["advantages"] == [0.0, 0.0, 0.0, 0.5, 0.5, 0.5]


def test_build_datums_rejects_a_sequence_without_logprobs(grpo_demo):
    sequence = types.SimpleNamespace(tokens=[10, 11], logprobs=None)

    with pytest.raises(ValueError, match="missing logprobs"):
        grpo_demo.build_datums(_Prompt(4), [sequence], [0.5])


# An empty eval slice has to disable the eval rather than divide by zero, whether the run
# opted out via the cadence or via the size.
@pytest.mark.parametrize(("eval_every", "eval_size"), [(0, 4), (1, 0)])
def test_main_trains_every_batch_without_an_eval_slice(grpo_demo, monkeypatch, eval_every, eval_size):
    rows = Rows([{"question": str(i), "answer": str(i)} for i in range(5)])
    monkeypatch.setattr(grpo_demo.datasets, "load_dataset", lambda *_: {"train": rows, "test": rows})
    monkeypatch.setattr(grpo_demo, "get_tokenizer", lambda *_: object())
    monkeypatch.setattr(grpo_demo.renderers, "get_renderer", lambda *_: MagicMock())
    monkeypatch.setattr(grpo_demo.model_info, "get_recommended_renderer_name", lambda *_: "renderer")
    monkeypatch.setattr(grpo_demo.tinker, "ServiceClient", lambda: _ServiceClient())
    monkeypatch.setattr(grpo_demo, "build_prompt", lambda _renderer, question: question)
    monkeypatch.setattr(
        grpo_demo,
        "sample_groups",
        lambda _client, prompts, _params, _group_size: [
            types.SimpleNamespace(sequences=[object(), object()]) for _ in prompts
        ],
    )
    monkeypatch.setattr(grpo_demo, "score_group", lambda *_: [0.0, 1.0])
    monkeypatch.setattr(grpo_demo, "build_datums", lambda *_: [object()])
    monkeypatch.setattr(grpo_demo, "length_metrics", lambda *_: {})
    steps: list[int] = []

    def train_step(*_):
        steps.append(1)
        return {}

    monkeypatch.setattr(grpo_demo, "train_step", train_step)

    grpo_demo.main(grpo_demo.Config(batch_size=1, max_steps=5, eval_every=eval_every, eval_size=eval_size))

    assert len(steps) == 5


def test_main_refuses_a_batch_larger_than_the_train_split(grpo_demo, monkeypatch):
    rows = Rows([{"question": "1", "answer": "1"}])
    monkeypatch.setattr(grpo_demo.datasets, "load_dataset", lambda *_: {"train": rows, "test": rows})

    with pytest.raises(SystemExit, match="exceeds the 1-row train split"):
        grpo_demo.main(grpo_demo.Config(batch_size=2, eval_size=0))
