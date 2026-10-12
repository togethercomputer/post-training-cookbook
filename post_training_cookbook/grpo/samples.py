"""GRPO-specific sample construction: group-relative advantages and sample assembly."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from together.post_training import Sample

from post_training_cookbook.samples import build_sample

_REWARD_VARIANCE_EPS = 1e-8


class GroupAdvantages(NamedTuple):
    """Result of computing GRPO advantages for a group of rewards.

    ``values`` is ``None`` when the group has zero reward variance (nothing
    to learn from). ``identical_nonzero`` is set in that case when every
    reward is the same nonzero value, useful for distinguishing "all wrong"
    from "all right" in diagnostics.
    """

    values: list[float] | None
    identical_nonzero: bool = False

    @property
    def zero_variance(self) -> bool:
        return self.values is None


def compute_group_advantages(rewards: list[float]) -> GroupAdvantages:
    """Compute group-relative GRPO advantages, or signal skip on zero variance."""
    if not rewards:
        return GroupAdvantages(None)
    arr = np.asarray(rewards, dtype=float)
    if float(np.std(arr)) < _REWARD_VARIANCE_EPS:
        has_nonzero = any(abs(r) > _REWARD_VARIANCE_EPS for r in rewards)
        return GroupAdvantages(None, identical_nonzero=has_nonzero)
    baseline = float(np.mean(arr))
    return GroupAdvantages([r - baseline for r in rewards])


def build_policy_sample(
    *,
    tokens: Sequence[int],
    logprobs: Sequence[float],
    loss_mask: Sequence[float],
    advantage: float,
    routed_experts_key: str | None = None,
) -> Sample:
    """Build one training :class:`Sample` for a policy loss.

    :func:`post_training_cookbook.samples.build_sample` owns the trim to target-aligned
    slots.

    Args:
        tokens: The flat prompt-plus-response run, in token space.
        logprobs: The generator's logprobs, one per token, zero over the prompt.
        loss_mask: One weight per token. The prompt boundary is expressed only as
            the run of zeros at its head, so there is no prompt length to pass
            and none to get wrong.
        advantage: One scalar for the whole sequence, broadcast onto the trained
            slots off ``loss_mask``. The group-relative baseline stays the
            caller's job, see :func:`compute_group_advantages`.
        routed_experts_key: Opaque generator routing capture, forwarded unchanged
            for the trimmed model input.

    Returns:
        A ``Sample`` carrying ``target_tokens``, ``weights``, ``advantages`` and
        ``logprobs``, plus ``routed_experts_key`` when one was provided.

    Raises:
        ValueError: As :func:`post_training_cookbook.samples.build_sample`.
    """
    sample = build_sample(
        tokens=tokens,
        loss_mask=loss_mask,
        advantages=[advantage if weight != 0.0 else 0.0 for weight in loss_mask],
        logprobs=logprobs,
    )
    if routed_experts_key is not None:
        sample["routed_experts_key"] = routed_experts_key
    return sample
