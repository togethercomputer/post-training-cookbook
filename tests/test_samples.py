"""Tests for the shared payload builders in post_training_cookbook.samples.

The parametrize table below is the payload contract: every array in full, as
literals, so a change to the layout has to be stated here before it can pass.
Each array is pinned by value — a mis-shifted sibling is invisible to a test
that only checks lengths, or only checks ``target_tokens``.
"""

import pytest
from together.lib.post_training._losses import INPUT_DTYPES
from together.post_training import Sample

from post_training_cookbook.grpo.samples import build_policy_sample
from post_training_cookbook.samples import (
    build_cross_entropy_sample,
    is_trainable,
    make_prompt,
    trained_slots,
)


def _input_tokens(sample: Sample) -> list[int]:
    return sample["model_input"]["chunks"][0]["encoded_text"]["tokens"]


def _array(sample: Sample, name: str) -> list[float]:
    return sample["loss_fn_inputs"][name]["data"]


# Token ids are deliberately all distinct, so an unshifted `target_tokens` could
# not pass by coincidence — a uniform run is the one case where token space and
# target-aligned agree.
@pytest.mark.parametrize(
    (
        "tokens",
        "loss_mask",
        "logprobs",
        "advantage",
        "model_input",
        "target_tokens",
        "weights",
        "advantages",
        "out_logprobs",
    ),
    [
        pytest.param(
            # P=2 leaves a zero prefix of length P-1 = 1.
            [101, 102, 103, 104, 105],
            [0, 0, 1, 1, 1],
            [0.0, 0.0, -2.1, -0.5, -1.3],
            1.5,
            [101, 102, 103, 104],
            [102, 103, 104, 105],
            [0.0, 1.0, 1.0, 1.0],
            [0.0, 1.5, 1.5, 1.5],
            [0.0, -2.1, -0.5, -1.3],
            id="first-response-slot",
        ),
        pytest.param(
            # A one-token prompt: the zero prefix vanishes entirely (P-1 = 0).
            [7, 8, 9],
            [1, 1, 1],
            [-0.1, -0.2, -0.3],
            2.0,
            [7, 8],
            [8, 9],
            [1.0, 1.0],
            [2.0, 2.0],
            [-0.2, -0.3],
            id="one-token-prompt",
        ),
        pytest.param(
            # The shortest trainable sequence: 2 tokens, 1 slot. Exercises the
            # boundary the two-token minimum sits just below.
            [11, 12],
            [0, 1],
            [0.0, -0.9],
            3.0,
            [11],
            [12],
            [1.0],
            [3.0],
            [-0.9],
            id="single-token-response",
        ),
        pytest.param(
            # The last slot must carry real values. A layout that padded the tail
            # would show 0.0 in the final entry of every array here.
            [21, 22, 23, 24],
            [0, 0, 0, 1],
            [0.0, 0.0, 0.0, -1.7],
            -0.5,
            [21, 22, 23],
            [22, 23, 24],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -0.5],
            [0.0, 0.0, -1.7],
            id="last-slot",
        ),
        pytest.param(
            # Multi-turn: tool responses and headers sit untrained between the
            # agent's actions, and advantage tracks the mask slot for slot.
            [31, 32, 33, 34, 35, 36],
            [0, 0, 1, 0, 1, 1],
            [0.0, 0.0, -0.5, 0.0, -0.3, -0.4],
            3.0,
            [31, 32, 33, 34, 35],
            [32, 33, 34, 35, 36],
            [0.0, 1.0, 0.0, 1.0, 1.0],
            [0.0, 3.0, 0.0, 3.0, 3.0],
            [0.0, -0.5, 0.0, -0.3, -0.4],
            id="interleaved-multi-turn-mask",
        ),
        pytest.param(
            # A negative advantage: the broadcast carries the sign, not a
            # magnitude.
            [41, 42, 43],
            [0, 1, 1],
            [0.0, -0.6, -0.7],
            -2.5,
            [41, 42],
            [42, 43],
            [1.0, 1.0],
            [-2.5, -2.5],
            [-0.6, -0.7],
            id="negative-advantage",
        ),
    ],
)
def test_policy_payload_contract(
    tokens, loss_mask, logprobs, advantage, model_input, target_tokens, weights, advantages, out_logprobs
):
    sample = build_policy_sample(tokens=tokens, logprobs=logprobs, loss_mask=loss_mask, advantage=advantage)

    assert _input_tokens(sample) == model_input
    assert _array(sample, "target_tokens") == target_tokens
    assert _array(sample, "weights") == weights
    assert _array(sample, "advantages") == advantages
    assert _array(sample, "logprobs") == pytest.approx(out_logprobs)

    assert set(sample["loss_fn_inputs"]) == {"target_tokens", "weights", "advantages", "logprobs"}


@pytest.mark.parametrize(
    ("tokens", "loss_mask", "target_tokens", "weights"),
    [
        pytest.param([10, 11, 12], None, [11, 12], [1.0, 1.0], id="default-mask-trains-everything"),
        pytest.param([10, 11, 12], [0, 1, 1], [11, 12], [1.0, 1.0], id="prompt-prefix-masked"),
        pytest.param([10, 11, 12, 13], [1, 0, 1, 1], [11, 12, 13], [0.0, 1.0, 1.0], id="interior-gap"),
        pytest.param([10, 11, 12], [0.5, 0.5, 0.5], [11, 12], [0.5, 0.5], id="fractional-weights"),
    ],
)
def test_cross_entropy_payload_contract(tokens, loss_mask, target_tokens, weights):
    sample = build_cross_entropy_sample(tokens, loss_mask)

    assert _input_tokens(sample) == tokens[:-1]
    assert _array(sample, "target_tokens") == target_tokens
    assert _array(sample, "weights") == weights
    assert set(sample["loss_fn_inputs"]) == {"target_tokens", "weights"}


class TestRejectsPayloadsThatCannotTrain:
    """The checks that would otherwise pass vacuously.

    ``pre-trimmed-array`` is the row that makes the misaligned state
    unrepresentable rather than merely zero-weighted.
    """

    @pytest.mark.parametrize(
        ("tokens", "loss_mask", "match"),
        [
            pytest.param([10], None, "a 1-token run", id="one-token-run"),
            pytest.param([], None, "a 0-token run", id="empty-run"),
            pytest.param([10, 11, 12], [1, 1], r"loss_mask holds 2 entries but tokens holds 3", id="mask-too-short"),
            pytest.param(
                [101, 102, 103, 104, 105],
                [0, 1, 1, 1],
                r"loss_mask holds 4 entries but tokens holds 5",
                id="pre-trimmed-array",
            ),
            pytest.param([10, 11, 12], [0, 0, 0], "trains no slot", id="all-zero-mask"),
            pytest.param([10, 11], [1, 0], "trains no slot", id="only-the-dropped-slot"),
        ],
    )
    def test_cross_entropy_rejects(self, tokens, loss_mask, match):
        with pytest.raises(ValueError, match=match):
            build_cross_entropy_sample(tokens, loss_mask)

    def test_length_mismatch_names_an_extra_array(self):
        with pytest.raises(ValueError, match=r"logprobs holds 2 entries but tokens holds 3"):
            build_policy_sample(tokens=[1, 2, 3], logprobs=[0.0, -1.0], loss_mask=[0, 1, 1], advantage=1.0)


class TestWireForm:
    def test_no_metadata_rides_along(self):
        """A tensor carries only ``data`` and ``dtype`` — no stale ``shape``."""
        sample = build_policy_sample(tokens=[1, 2, 3], logprobs=[0.0, -1.0, -2.0], loss_mask=[0, 1, 1], advantage=1.0)

        for name, tensor in sample["loss_fn_inputs"].items():
            assert set(tensor) == {"data", "dtype"}, name

    def test_target_tokens_are_ints_and_weights_are_floats(self):
        sample = build_cross_entropy_sample([10, 11, 12], [0, 1, 1])

        assert all(isinstance(value, int) for value in _array(sample, "target_tokens"))
        assert all(isinstance(value, float) for value in _array(sample, "weights"))

    def test_every_emitted_key_is_one_the_sdk_accepts(self):
        """Ties the hardcoded dtypes to the SDK's registry by test, not by import."""
        emitted = build_policy_sample(
            tokens=[1, 2, 3], logprobs=[0.0, -1.0, -2.0], loss_mask=[0, 1, 1], advantage=1.0
        )["loss_fn_inputs"]

        assert set(emitted) <= set(INPUT_DTYPES)
        for name, tensor in emitted.items():
            assert tensor["dtype"] in INPUT_DTYPES[name], name

    def test_routed_experts_key_omitted_unless_set(self):
        bare = build_policy_sample(tokens=[1, 2, 3], logprobs=[0.0, -1.0, -2.0], loss_mask=[0, 1, 1], advantage=1.0)
        keyed = build_policy_sample(
            tokens=[1, 2, 3],
            logprobs=[0.0, -1.0, -2.0],
            loss_mask=[0, 1, 1],
            advantage=1.0,
            routed_experts_key="k",
        )
        assert "routed_experts_key" not in bare
        assert keyed["routed_experts_key"] == "k"


class TestIsTrainable:
    @pytest.mark.parametrize(
        ("loss_mask", "expected"),
        [
            pytest.param([0, 1, 1], True, id="response-trained"),
            pytest.param([0, 0, 0], False, id="nothing-trained"),
            pytest.param([1, 0], False, id="only-the-dropped-slot"),
            pytest.param([0, 0.5], True, id="fractional-counts"),
            pytest.param([1], False, id="single-token"),
            pytest.param([], False, id="empty"),
        ],
    )
    def test_accounts_for_the_dropped_first_entry(self, loss_mask, expected):
        assert is_trainable(loss_mask) is expected


def test_make_prompt_wraps_tokens():
    prompt = make_prompt([1, 2, 3])

    assert prompt["chunks"][0]["encoded_text"]["tokens"] == [1, 2, 3]


class TestTrainedSlots:
    @pytest.mark.parametrize(
        ("tokens", "loss_mask", "expected"),
        [
            # The trim drops the mask's first entry, so [0, 1, 1, 1] leaves 3.
            pytest.param([10, 11, 12, 13], [0, 1, 1, 1], 3, id="response-trained"),
            pytest.param([10, 11, 12], [1, 1, 0], 1, id="trailing-gap"),
            pytest.param([10, 11, 12, 13], [1, 0, 1, 1], 2, id="interior-gap"),
            # Weight mass, not a slot count: truncating would report 1, not 1.5.
            pytest.param([10, 11, 12, 13], [0.5, 0.5, 0.5, 0.5], 1.5, id="fractional-weights"),
        ],
    )
    def test_sums_the_weights_left_after_the_trim(self, tokens, loss_mask, expected):
        sample = build_cross_entropy_sample(tokens, loss_mask)

        assert trained_slots([sample]) == pytest.approx(expected)

    def test_sums_across_a_batch(self):
        batch = [
            build_cross_entropy_sample([10, 11, 12], [0, 1, 1]),
            build_cross_entropy_sample([20, 21, 22, 23], [1, 1, 1, 1]),
        ]

        assert trained_slots(batch) == 2 + 3
