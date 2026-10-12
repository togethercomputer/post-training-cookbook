"""Submit-order checks for the SFT migration demo."""

from __future__ import annotations

import types

import pytest


class _Future:
    def __init__(self, value, log: list[str], name: str):
        self._value = value
        self._log = log
        self._name = name

    def result(self):
        self._log.append(f"resolve:{self._name}")
        return self._value


class _TrainingClient:
    def __init__(self, log: list[str]):
        self._log = log

    def forward_backward(self, datums, loss_fn):
        self._log.append(f"submit:fwd_bwd:{loss_fn}:{len(datums)}")
        return _Future(types.SimpleNamespace(metrics={"loss": 1.25}), self._log, "fwd_bwd")

    def optim_step(self, adam_params):
        self._log.append("submit:optim")
        return _Future(types.SimpleNamespace(metrics=None), self._log, "optim")


def test_train_step_uses_cross_entropy_and_submits_before_resolving(sft_demo):
    log: list[str] = []
    metrics = sft_demo.train_step(_TrainingClient(log), ["d1", "d2"], object())

    assert metrics == {"loss": 1.25}
    assert log == [
        "submit:fwd_bwd:cross_entropy:2",
        "submit:optim",
        "resolve:fwd_bwd",
        "resolve:optim",
    ]


@pytest.mark.parametrize(("step", "expected"), [(0, 1e-4), (9, 1e-5)])
def test_decayed_lr_spans_the_full_epoch(sft_demo, step, expected):
    assert sft_demo.decayed_lr(1e-4, step, n_train_batches=10) == pytest.approx(expected)


def test_main_refuses_a_batch_larger_than_the_train_split(sft_demo, monkeypatch):
    train_rows = types.SimpleNamespace(shuffle=lambda seed: ["row"])
    monkeypatch.setattr(sft_demo.datasets, "load_dataset", lambda *_: {"train": train_rows})

    with pytest.raises(SystemExit, match="exceeds the 1-row train split"):
        sft_demo.main(sft_demo.Config(batch_size=2))
