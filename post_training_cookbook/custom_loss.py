"""Two-pass custom-loss helper backed by ``SessionClient.trainer``.

Use this when the API's built-in cross-entropy / GRPO losses are not enough
and the loss must be computed in client-side Python. The flow:

1. ``SessionClient.trainer.forward`` scores the batch without accumulating
   gradients and returns one log-probability per slot of the target tokens,
   zeroed at every slot the sample's ``weights`` tensor excludes.
2. Treat those logprobs as autograd leaves, evaluate ``custom_loss_fn`` on
   them, and ``.backward()`` to materialise ``dL/d(logprobs)``.
3. Submit the per-sample gradients to ``SessionClient.trainer.custom_forward_backward``
   so the session can finish the backward pass and accumulate parameter
   grads.

The session's optimiser state only updates on a later ``session.trainer.optim_step``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from together.post_training import LossConfig

if TYPE_CHECKING:
    from together.post_training import Sample, SessionClient

CustomLossFn = Callable[[list[torch.Tensor]], torch.Tensor]


@dataclass
class CustomLossResult:
    """Result of one custom-loss forward/backward step."""

    loss: float
    """Scalar value returned by ``custom_loss_fn``."""

    metrics: dict[str, float]
    """Any metrics returned by the ``custom_forward_backward`` operation."""

    raw: Any
    """Underlying SDK ``custom_forward_backward`` response."""


def compute_custom_forward_backward(
    session: SessionClient,
    batch: list[Sample],
    custom_loss_fn: CustomLossFn,
    *,
    timeout: float | None = None,
) -> CustomLossResult:
    """Run one training step with a Python-side custom loss.

    Args:
        session: Live post-training session.
        batch: Forward-backward training samples (same format as
            ``session.trainer.forward_backward``).
        custom_loss_fn: Receives a list with one ``(S_i,)`` float32 tensor
            of per-slot logprobs per sample (same order as ``batch``)
            and must return a scalar ``torch.Tensor``.
        timeout: Per-operation timeout in seconds. ``None`` uses the SDK
            default.
    """
    raw_logprobs = compute_logprobs(session, batch, timeout=timeout)
    if len(raw_logprobs) != len(batch):
        raise RuntimeError(f"forward pass returned {len(raw_logprobs)} logprob arrays for {len(batch)} samples")

    leaves: list[torch.Tensor] = [torch.tensor(data, dtype=torch.float32, requires_grad=True) for data in raw_logprobs]

    loss = custom_loss_fn(leaves)
    if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
        raise ValueError(f"custom_loss_fn must return a scalar tensor, got {loss!r}")
    loss.backward()

    gradients: list[dict[str, Any]] = []
    for leaf in leaves:
        grad = leaf.grad if leaf.grad is not None else torch.zeros_like(leaf)
        gradients.append({"data": grad.tolist(), "dtype": "D_TYPE_FLOAT32"})

    kwargs: dict[str, Any] = {"samples": batch, "gradients": gradients}
    if timeout is not None:
        kwargs["timeout"] = timeout
    raw = session.trainer.custom_forward_backward(**kwargs)
    metrics = raw.get("metrics", {}) if isinstance(raw, dict) else {}
    return CustomLossResult(loss=float(loss.item()), metrics=dict(metrics), raw=raw)


def compute_logprobs(
    session: SessionClient,
    batch: list[Sample],
    *,
    timeout: float | None = None,
) -> list[list[float]]:
    """Score a batch without accumulating gradients and return per-sample slot logprobs.

    Useful for snapshotting reference-model logprobs (e.g. for DPO) without
    advancing the session's optimiser state. Returns one list per sample in
    the same order as ``batch``. Only the slots the loss trains on carry a
    real log-probability; a slot whose ``weights`` entry is zero comes back
    as ``0.0``.

    Raises:
        RuntimeError: If the forward pass returns no per-slot logprobs.
    """
    kwargs: dict[str, Any] = {"samples": batch, "loss": LossConfig(type="LOSS_TYPE_CROSS_ENTROPY")}
    if timeout is not None:
        kwargs["timeout"] = timeout
    result = session.trainer.forward(**kwargs)
    if result.loss_fn_outputs is None:
        raise RuntimeError("forward pass returned no per-slot logprobs; retry the request or reduce the batch size")
    return [list(output.tensors["logprobs"].data) for output in result.loss_fn_outputs]
