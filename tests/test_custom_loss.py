"""Tests for the custom-loss forward/backward bridge."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from together.post_training import ForwardBackwardResult, LossFnOutput, TensorData

from post_training_cookbook.custom_loss import compute_custom_forward_backward, compute_logprobs


def _forward_result(logprobs: list[list[float]] | None) -> ForwardBackwardResult:
    outputs = None
    if logprobs is not None:
        outputs = [
            LossFnOutput(tensors={"logprobs": TensorData(data=list(data), dtype="float32")}) for data in logprobs
        ]
    return ForwardBackwardResult(loss=0.0, loss_fn_outputs=outputs)


class _FakeTrainer:
    """The ``session.trainer`` surface.

    Deliberately a separate object from the session: the helper must reach the
    backend through ``session.trainer``, and a fake that aliased itself would let
    a call that skipped the sub-client still resolve.
    """

    def __init__(self, session: _FakeSession) -> None:
        self._session = session
        self.forward_kwargs: dict[str, Any] | None = None
        self.custom_forward_backward_kwargs: dict[str, Any] | None = None

    def forward(self, **kwargs: Any) -> ForwardBackwardResult:
        self.forward_kwargs = kwargs
        return _forward_result(self._session.logprobs)

    def custom_forward_backward(self, **kwargs: Any) -> Any:
        self.custom_forward_backward_kwargs = kwargs
        return self._session.custom_forward_backward_response


class _FakeSession:
    def __init__(
        self,
        *,
        logprobs: list[list[float]] | None,
        custom_forward_backward_response: Any | None = None,
    ) -> None:
        self.logprobs = logprobs
        self.custom_forward_backward_response = custom_forward_backward_response or {"metrics": {}}
        self.trainer = _FakeTrainer(self)


@pytest.fixture
def samples() -> list[dict[str, str]]:
    return [{"id": "chosen"}, {"id": "rejected"}]


def test_compute_logprobs_returns_forward_data_and_forwards_timeout(samples: list[dict[str, str]]) -> None:
    session = _FakeSession(logprobs=[[-0.1, -0.2], [-0.3]])

    result = compute_logprobs(session, samples, timeout=12.5)

    assert result == [[-0.1, -0.2], [-0.3]]
    assert session.trainer.forward_kwargs == {
        "samples": samples,
        "loss": {"type": "LOSS_TYPE_CROSS_ENTROPY"},
        "timeout": 12.5,
    }


def test_compute_logprobs_omits_timeout_when_unset(samples: list[dict[str, str]]) -> None:
    session = _FakeSession(logprobs=[[-0.1]])

    compute_logprobs(session, samples)

    assert session.trainer.forward_kwargs == {"samples": samples, "loss": {"type": "LOSS_TYPE_CROSS_ENTROPY"}}


def test_compute_logprobs_raises_when_forward_has_no_outputs(samples: list[dict[str, str]]) -> None:
    session = _FakeSession(logprobs=None)

    with pytest.raises(RuntimeError, match="forward pass returned no per-slot logprobs"):
        compute_logprobs(session, samples)


def test_compute_logprobs_requests_cross_entropy(samples: list[dict[str, str]]) -> None:
    session = _FakeSession(logprobs=[[-0.1], [-0.2]])

    compute_logprobs(session, samples)

    assert session.trainer.forward_kwargs["loss"] == {"type": "LOSS_TYPE_CROSS_ENTROPY"}


def test_compute_custom_forward_backward_sends_gradients_and_returns_metrics(
    samples: list[dict[str, str]],
) -> None:
    session = _FakeSession(
        logprobs=[[1.0, 2.0], [3.0]],
        custom_forward_backward_response={"metrics": {"tokens": 3}, "ok": True},
    )

    def loss_fn(leaves: list[torch.Tensor]) -> torch.Tensor:
        assert [leaf.tolist() for leaf in leaves] == [[1.0, 2.0], [3.0]]
        assert all(leaf.requires_grad for leaf in leaves)
        return (leaves[0] * torch.tensor([2.0, 3.0])).sum() + (leaves[1] * 4.0).sum()

    result = compute_custom_forward_backward(session, samples, loss_fn, timeout=99.0)

    assert result.loss == pytest.approx(20.0)  # 1*2 + 2*3 + 3*4 = 20
    assert result.metrics == {"tokens": 3}
    assert result.raw == {"metrics": {"tokens": 3}, "ok": True}
    assert session.trainer.forward_kwargs == {
        "samples": samples,
        "loss": {"type": "LOSS_TYPE_CROSS_ENTROPY"},
        "timeout": 99.0,
    }
    assert session.trainer.custom_forward_backward_kwargs == {
        "samples": samples,
        "gradients": [
            {"data": [2.0, 3.0], "dtype": "D_TYPE_FLOAT32"},
            {"data": [4.0], "dtype": "D_TYPE_FLOAT32"},
        ],
        "timeout": 99.0,
    }


def test_compute_custom_forward_backward_defaults_missing_gradient_to_zero(
    samples: list[dict[str, str]],
) -> None:
    session = _FakeSession(logprobs=[[1.0], [2.0]])

    def loss_fn(leaves: list[torch.Tensor]) -> torch.Tensor:
        return leaves[0].sum()

    compute_custom_forward_backward(session, samples, loss_fn)

    assert session.trainer.custom_forward_backward_kwargs == {
        "samples": samples,
        "gradients": [
            {"data": [1.0], "dtype": "D_TYPE_FLOAT32"},
            {"data": [0.0], "dtype": "D_TYPE_FLOAT32"},
        ],
    }


def test_compute_custom_forward_backward_validates_forward_batch_size(
    samples: list[dict[str, str]],
) -> None:
    session = _FakeSession(logprobs=[[-0.1]])

    with pytest.raises(RuntimeError, match="forward pass returned 1 logprob arrays for 2 samples"):
        compute_custom_forward_backward(session, samples, lambda leaves: torch.tensor(1.0))


def test_compute_custom_forward_backward_requires_scalar_tensor(
    samples: list[dict[str, str]],
) -> None:
    session = _FakeSession(logprobs=[[-0.1], [-0.2]])

    with pytest.raises(ValueError, match="custom_loss_fn must return a scalar tensor"):
        compute_custom_forward_backward(session, samples, lambda leaves: leaves[0])


def test_compute_custom_forward_backward_requires_tensor_loss(
    samples: list[dict[str, str]],
) -> None:
    session = _FakeSession(logprobs=[[-0.1], [-0.2]])

    with pytest.raises(ValueError, match="custom_loss_fn must return a scalar tensor"):
        compute_custom_forward_backward(session, samples, lambda leaves: 1.0)
