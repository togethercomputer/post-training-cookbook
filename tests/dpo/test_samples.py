"""Tests for DPO preference-pair sample construction."""

import pytest

pytest.importorskip("together.post_training")

from post_training_cookbook.dpo.samples import build_preference_pair  # noqa: E402
from post_training_cookbook.samples import extract_loss_mask  # noqa: E402
from tests._codec_fakes import DummyCodec  # noqa: E402

MSGS = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]


def _tokens(sample) -> list[int]:
    return list(sample["model_input"]["chunks"][0]["encoded_text"]["tokens"])


def test_preference_pair_shares_prompt_prefix() -> None:
    chosen_msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yes"},
    ]
    rejected_msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "no"},
    ]
    built = build_preference_pair(
        chosen_messages=chosen_msgs,
        rejected_messages=rejected_msgs,
        codec=DummyCodec(),
        max_length=10_000,
    )
    assert built is not None
    chosen, rejected = built
    prompt_len = len("user: hi\n")
    assert _tokens(chosen)[:prompt_len] == _tokens(rejected)[:prompt_len]


def test_preference_pair_loss_mask_is_response_only() -> None:
    chosen_msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ok"},
    ]
    rejected_msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "no"},
    ]
    built = build_preference_pair(
        chosen_messages=chosen_msgs,
        rejected_messages=rejected_msgs,
        codec=DummyCodec(),
        max_length=10_000,
    )
    assert built is not None
    chosen, _ = built
    mask = extract_loss_mask(chosen)

    # The builder trims: one slot per input token, and the mask it reports is the
    # token-space mask less its first entry. The header stays untrained, but the
    # body's final token is trained.
    prompt_text_len = len("user: hi\n")
    header_len = len("assistant: ")
    body_len = len("ok\n")

    token_space_mask = [0] * prompt_text_len + [0] * header_len + [1] * body_len
    expected = token_space_mask[1:]

    assert mask == expected


@pytest.mark.parametrize(
    ("max_length", "expected_none"),
    [
        pytest.param(len("user: hi\nassistant: "), True, id="no-response-token"),
        pytest.param(len("user: hi\nassistant: ") + 2, False, id="one-response-token"),
    ],
)
def test_a_side_with_no_trained_token_skips_the_pair(max_length: int, expected_none: bool) -> None:
    """At the shorter budget the prompt and header fit but no response token does.

    That leaves an all-zero mask. Returning ``None`` keeps it a counted skip in
    the training loop instead of an unpacking error mid-run.
    """
    built = build_preference_pair(
        chosen_messages=MSGS,
        rejected_messages=MSGS,
        codec=DummyCodec(),
        max_length=max_length,
    )

    assert (built is None) is expected_none


def test_both_sides_keep_a_trained_final_slot() -> None:
    built = build_preference_pair(
        chosen_messages=MSGS,
        rejected_messages=MSGS,
        codec=DummyCodec(),
        max_length=len("user: hi\nassistant: ") + 2,
    )

    assert built is not None
    chosen, rejected = built
    assert extract_loss_mask(chosen)[-1] == 1
    assert extract_loss_mask(rejected)[-1] == 1
