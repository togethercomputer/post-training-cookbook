"""Loss-agnostic payload builders shared by the training recipes.

Pure functions that turn a token sequence into the SDK's ``ModelInput``/``Sample``
payloads, target-aligned for the loss.

Slot ``i`` scores the prediction made at input position ``i`` against
``target_tokens[i]``, so a run of N tokens yields N-1 slots: the first token is
an input that is never a target, and the last is a target that is never an input.
The payload is therefore *trimmed* — ``model_input`` is ``tokens[:-1]`` and every
loss array is ``arr[1:]``, all off the same flat run.

The builders own that trim. Their *inputs* are all in token space, unshifted and
length N, and ``target_tokens`` is derived rather than accepted.

``https://docs.together.ai/docs/post-training/concepts#data-and-samples`` documents the
layout, and why an array trimmed from the wrong end trains on the wrong tokens
without an error.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from together.post_training import (
    EncodedTextChunk,
    ModelInput,
    ModelInputChunk,
    Sample,
    TensorData,
)


def _model_input(tokens: Sequence[int]) -> ModelInput:
    return ModelInput(chunks=[ModelInputChunk(encoded_text=EncodedTextChunk(tokens=list(tokens)))])


def make_prompt(tokens: Sequence[int]) -> ModelInput:
    """Build a sampling model input from a flat token list.

    Sampling conditions on every token given, so this is not trimmed: the trim
    belongs to training payloads, where the final token is a target.
    """
    return _model_input(tokens)


def is_trainable(loss_mask: Sequence[float]) -> bool:
    """Whether a token-space mask still trains a slot once the builders trim it.

    The trim drops entry 0 — slot 0 predicts ``tokens[1]`` — so a mask whose only
    nonzero entry is the first one trains nothing, and the builders reject it.
    Filter with this rather than letting construction raise mid-run.
    """
    return any(weight != 0.0 for weight in loss_mask[1:])


def _float_tensor(values: Sequence[float]) -> TensorData:
    return {"data": [float(value) for value in values], "dtype": "float32"}


def build_sample(
    *,
    tokens: Sequence[int],
    loss_mask: Sequence[float],
    advantages: Sequence[float] | None = None,
    logprobs: Sequence[float] | None = None,
) -> Sample:
    """Assemble a target-aligned :class:`Sample` from token-space arrays.

    Every array is trimmed together with the targets. The prompt boundary is read
    only off the zeros at the head of ``loss_mask``, so there is no separate
    prompt length to contradict it.

    Args:
        tokens: The full run, ``prompt + response``, in token space.
        loss_mask: One weight per token, zero over the prompt.
        advantages: Optional per-token advantages, required by the policy losses.
        logprobs: Optional per-token rollout logprobs, required by the policy
            losses.

    Returns:
        A ``Sample`` whose ``model_input`` holds ``len(tokens) - 1`` tokens and
        whose every ``loss_fn_inputs`` array holds ``len(tokens) - 1`` slots. The
        optional arrays are omitted from the payload when ``None`` rather than
        zero-filled.

    Raises:
        ValueError: If any array's length disagrees with ``tokens``, or if
            ``loss_mask`` trains no slot once the trim drops its first entry.
    """
    optional_arrays = {"advantages": advantages, "logprobs": logprobs}
    for name, values in {"loss_mask": loss_mask, **optional_arrays}.items():
        if values is not None and len(values) != len(tokens):
            raise ValueError(f"{name} holds {len(values)} entries but tokens holds {len(tokens)}")

    if not is_trainable(loss_mask):
        raise ValueError(
            f"a {len(tokens)}-token run with a {len(loss_mask)}-entry loss_mask trains no slot: slot i "
            "predicts tokens[i + 1], so the trim drops entry 0. Filter with is_trainable() before building."
        )

    return Sample(
        model_input=_model_input(tokens[:-1]),
        loss_fn_inputs={
            # No int() on the token ids: the SDK rejects float data declared
            # int64 rather than truncating it, and coercing would turn a corrupt
            # token id into a plausible one with nothing left to detect it.
            "target_tokens": {"data": list(tokens[1:]), "dtype": "int64"},
            "weights": _float_tensor(loss_mask[1:]),
            **{name: _float_tensor(values[1:]) for name, values in optional_arrays.items() if values is not None},
        },
    )


def build_cross_entropy_sample(tokens: Sequence[int], loss_mask: Sequence[float] | None = None) -> Sample:
    """Build a cross-entropy training :class:`Sample`, carrying only weights.

    ``loss_mask`` selects whether ``tokens[i]`` is trained on and need not be
    binary — cross entropy accepts fractional weights. Omitting it trains on
    every token, which is pure language-model memorisation.

    Raises:
        ValueError: As :func:`build_sample`.
    """
    return build_sample(tokens=tokens, loss_mask=[1.0] * len(tokens) if loss_mask is None else loss_mask)


def extract_loss_mask(sample: Sample) -> list[float]:
    """Read the target-aligned loss mask back off a built Sample.

    The ``weights`` array is already target-aligned, unlike the token-space
    mask that went in, which is a slot longer.

    Returns:
        The ``weights`` array as floats, one entry per slot. Float because cross
        entropy accepts fractional weights — rounding them would move a trained
        slot out of the sum.
    """
    return [float(value) for value in sample["loss_fn_inputs"]["weights"]["data"]]


def trained_slots(samples: Iterable[Sample]) -> float:
    """Sum the loss weights of a batch: the denominator for a per-slot mean.

    Returns:
        The total weight mass. A float because cross entropy accepts fractional
        weights; an all-0-or-1 batch gives a whole number.
    """
    return math.fsum(weight for sample in samples for weight in sample["loss_fn_inputs"]["weights"]["data"])
