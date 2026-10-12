"""Tests for the DPO loss closure."""

import math

import pytest
import torch

from post_training_cookbook.dpo.loss import make_dpo_loss, masked_logp_sum


def test_masked_logp_sum_zeros_outside_mask() -> None:
    logp = torch.tensor([-1.0, -2.0, -3.0, -4.0])
    out = masked_logp_sum(logp, [0, 1, 1, 0])
    assert math.isclose(out.item(), -5.0)


def test_masked_logp_sum_rejects_a_length_mismatch() -> None:
    """A mask of a different length means one side is still in token space.

    Both arrays are target-aligned and therefore equal length: the mask is read
    off the built sample and `forward` returns one logprob per slot. Scoring the
    shorter prefix instead would silently drop the tail of the response.
    """
    logp = torch.tensor([-1.0, -1.0, -1.0])

    with pytest.raises(ValueError, match="mask holds 5 slots but logprobs holds 3"):
        masked_logp_sum(logp, [1, 1, 1, 1, 1])


def test_make_dpo_loss_matches_hand_value() -> None:
    masks = [[1, 1], [1, 1]]
    ref_sums = [0.0, 0.0]
    chosen_lp = torch.tensor([-0.4, -0.6], requires_grad=True)  # sum = -1.0
    rejected_lp = torch.tensor([-1.0, -1.0], requires_grad=True)  # sum = -2.0

    loss_fn = make_dpo_loss(ref_sums, masks, beta=1.0, label_smoothing=0.0)
    loss = loss_fn([chosen_lp, rejected_lp])

    expected = -math.log(1.0 / (1.0 + math.exp(-1.0)))
    assert math.isclose(loss.item(), expected, rel_tol=1e-6)


def test_make_dpo_loss_backward_produces_gradients() -> None:
    masks = [[1, 1], [1, 1], [1, 1], [1, 1]]
    ref_sums = [0.0, 0.0, 0.0, 0.0]
    leaves = [
        torch.tensor([-0.3, -0.4], requires_grad=True),
        torch.tensor([-0.7, -0.8], requires_grad=True),
        torch.tensor([-0.2, -0.3], requires_grad=True),
        torch.tensor([-0.9, -1.0], requires_grad=True),
    ]
    loss_fn = make_dpo_loss(ref_sums, masks, beta=0.5)
    loss = loss_fn(leaves)
    loss.backward()
    for leaf in leaves:
        assert leaf.grad is not None
        assert leaf.grad.shape == leaf.shape


def test_make_dpo_loss_requires_paired_batch() -> None:
    with pytest.raises(ValueError):
        make_dpo_loss([0.0, 0.0, 0.0], [[1], [1], [1]], beta=0.1)


def test_make_dpo_loss_validates_leaf_count() -> None:
    loss_fn = make_dpo_loss([0.0, 0.0], [[1], [1]], beta=0.1)
    with pytest.raises(RuntimeError):
        loss_fn([torch.tensor([-1.0])])


def test_label_smoothing_caps_loss() -> None:
    masks = [[1, 1], [1, 1]]
    ref_sums = [0.0, 0.0]
    chosen_lp = torch.tensor([0.0, 0.0], requires_grad=True)
    rejected_lp = torch.tensor([-100.0, -100.0], requires_grad=True)

    plain = make_dpo_loss(ref_sums, masks, beta=1.0, label_smoothing=0.0)
    smoothed = make_dpo_loss(ref_sums, masks, beta=1.0, label_smoothing=0.5)
    assert smoothed([chosen_lp, rejected_lp]).item() > plain([chosen_lp, rejected_lp]).item()
