"""Tests for GRPO sample construction and advantage computation.

The payload layout itself is pinned in ``tests/test_samples.py``; this file
covers what is GRPO-specific — how a scored group becomes samples, and the
advantage bookkeeping around it.
"""

import pytest

from post_training_cookbook.grpo.config import GRPOConfig
from post_training_cookbook.grpo.samples import compute_group_advantages
from post_training_cookbook.grpo.single_turn import (
    ScoredCompletion,
    build_scored_group_samples,
    skip_counts,
)


def _completion(tokens: list[int], logprobs: list[float], reward: float = 1.0) -> ScoredCompletion:
    return ScoredCompletion(
        tokens=tokens,
        logprobs=logprobs,
        reward=reward,
        raw_reward=reward,
        metric=reward,
        penalty=0.0,
        is_truncated=False,
        response_budget=1024,
        sampled_length=len(tokens),
        trainable_length=len(tokens),
        was_clipped=False,
        stop_reason="STOP_REASON_STOP",
    )


@pytest.mark.parametrize(
    ("rewards", "zero_variance", "identical_nonzero"),
    [
        ([1.0, 0.0], False, False),  # mean-centered, has variance
        ([0.0, 1.0], False, False),
        ([0.5, 0.5, 0.5], True, True),  # uniform nonzero
        ([0.5, 0.5], True, True),
        ([0.0, 0.0], True, False),  # uniform zero
        ([], True, False),  # empty group
    ],
)
def test_compute_group_advantages_classification(rewards, zero_variance, identical_nonzero):
    result = compute_group_advantages(rewards)
    assert result.zero_variance is zero_variance
    assert result.identical_nonzero is identical_nonzero
    if zero_variance:
        assert result.values is None
    else:
        assert result.values is not None
        assert sum(result.values) == pytest.approx(0.0)


class TestBuildScoredGroupSamples:
    def test_builds_samples_with_advantages(self):
        cfg = GRPOConfig(model="test-model", overlong_masking=False)
        group = [_completion([10], [-0.1]), _completion([20], [-0.2], reward=0.0)]
        result = compute_group_advantages([sc.reward for sc in group])
        assert result.values is not None
        samples = build_scored_group_samples(group, result.values, [1, 2], cfg)
        assert len(samples) == 2


class TestSkipCounts:
    def test_counts_zero_variance_and_identical_nonzero(self):
        a = compute_group_advantages([1.0, 1.0])
        b = compute_group_advantages([0.0, 0.0])
        c = compute_group_advantages([1.0, 0.0])
        zv, inz = skip_counts([a, b, c])
        assert zv == 2
        assert inz == 1


class TestDegenerateCompletionsAreSkipped:
    """The builder rejects a payload that trains no slot, so the caller filters.

    Without this the first degenerate completion in a group would abort the run.
    A `len(response) == 0` filter would miss the one-token row, so the filter is
    on the mask.
    """

    @pytest.mark.parametrize(
        ("group", "advantages", "prompt_tokens", "expected"),
        [
            pytest.param(
                [_completion([], []), _completion([20], [-0.2], reward=0.0)],
                [1.0, -1.0],
                [1, 2],
                [[1, 2]],
                id="empty-completion",
            ),
            pytest.param([_completion([20], [-0.2])], [1.0], [], [], id="one-token-run"),
        ],
    )
    def test_a_completion_that_trains_no_slot_is_dropped(
        self,
        group: list[ScoredCompletion],
        advantages: list[float],
        prompt_tokens: list[int],
        expected: list[list[int]],
    ) -> None:
        cfg = GRPOConfig(model="test-model", overlong_masking=False)

        samples = build_scored_group_samples(group, advantages, prompt_tokens, cfg)

        assert [s["model_input"]["chunks"][0]["encoded_text"]["tokens"] for s in samples] == expected


@pytest.mark.parametrize("key", [None, ""])
def test_single_turn_replay_rejects_missing_training_key(key):
    completion = _completion([10], [-0.1])._replace(routed_experts_key=key)
    with pytest.raises(ValueError, match="routed_experts_key"):
        build_scored_group_samples([completion], [0.5], [1, 2], GRPOConfig(model="test", routing_replay=True))


def test_single_turn_replay_rejects_clipped_training_completion():
    completion = _completion([10], [-0.1])._replace(
        routed_experts_key="full-capture",
        was_clipped=True,
        is_truncated=True,
        sampled_length=2,
    )
    with pytest.raises(ValueError, match="clipped completion"):
        build_scored_group_samples([completion], [0.5], [1, 2], GRPOConfig(model="test", routing_replay=True))
    assert (
        build_scored_group_samples(
            [completion],
            [0.5],
            [1, 2],
            GRPOConfig(model="test", routing_replay=True, overlong_masking=True),
        )
        == []
    )
