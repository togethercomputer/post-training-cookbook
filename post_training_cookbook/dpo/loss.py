"""DPO loss closure for use with ``compute_custom_forward_backward``."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from post_training_cookbook.custom_loss import CustomLossFn


def masked_logp_sum(logprobs: torch.Tensor, mask: list[float]) -> torch.Tensor:
    """Sum the logprobs of the slots the mask trains.

    Both arrays are target-aligned and the same length; a token-space mask is
    one entry longer, so it would score every slot one position early.

    Returns:
        The mask-weighted sum, as a scalar tensor. It carries gradient when
        ``logprobs`` does.

    Raises:
        ValueError: If the two lengths disagree.
    """
    if len(mask) != logprobs.shape[0]:
        raise ValueError(
            f"mask holds {len(mask)} slots but logprobs holds {logprobs.shape[0]}; "
            "if the service clipped the sample, lower max_length"
        )
    return (logprobs * torch.tensor(mask, dtype=logprobs.dtype)).sum()


def make_dpo_loss(
    ref_sums: list[float],
    masks: list[list[float]],
    *,
    beta: float,
    label_smoothing: float = 0.0,
) -> CustomLossFn:
    """Build a DPO loss closure that ``compute_custom_forward_backward`` can call.

    The batch is flat-interleaved: indices ``2k`` are chosen and ``2k+1`` are
    rejected.

    Args:
        ref_sums: Per-sample sums of reference-model per-slot logprobs, masked
            to the response slots.
        masks: Each sample's target-aligned mask, as ``extract_loss_mask``
            returns it.
        beta: The DPO temperature on the margin.
        label_smoothing: Weight given to the reversed preference.

    Returns:
        A closure over this batch, ready for
        ``compute_custom_forward_backward``.

    Raises:
        ValueError: If the two lists disagree in length, or if the batch does
            not hold an even number of samples.
    """
    if len(ref_sums) != len(masks):
        raise ValueError(f"ref_sums ({len(ref_sums)}) and masks ({len(masks)}) lengths disagree")
    if len(ref_sums) % 2 != 0:
        raise ValueError(f"ref_sums length must be even (pairs), got {len(ref_sums)}")
    n_pairs = len(ref_sums) // 2

    def loss_fn(leaves: list[torch.Tensor]) -> torch.Tensor:
        if len(leaves) != 2 * n_pairs:
            raise RuntimeError(f"Expected {2 * n_pairs} logprob tensors, got {len(leaves)}")

        logp_sums = [masked_logp_sum(leaf, mask) for leaf, mask in zip(leaves, masks)]
        ref_t = [torch.tensor(s, dtype=logp_sums[0].dtype) for s in ref_sums]

        per_pair_losses: list[torch.Tensor] = []
        for k in range(n_pairs):
            ci, ri = 2 * k, 2 * k + 1
            delta_chosen = logp_sums[ci] - ref_t[ci]
            delta_rejected = logp_sums[ri] - ref_t[ri]
            beta_logits = beta * (delta_chosen - delta_rejected)
            per_pair_losses.append(
                -(1.0 - label_smoothing) * F.logsigmoid(beta_logits) - label_smoothing * F.logsigmoid(-beta_logits)
            )
        return torch.stack(per_pair_losses).mean()

    return loss_fn
