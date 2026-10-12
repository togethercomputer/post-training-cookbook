"""End-to-end SFT payload tests: render -> truncate -> build.

``test_rendering.py`` stops at the pre-trim ``TrainableSequence.loss_mask``, and
``test_samples.py`` starts at the builder's inputs. Nothing covered the join,
which is where a future truncation change would break alignment silently.
"""

from __future__ import annotations

import pytest
from together.post_training import Sample

from post_training_cookbook.sft.train import (
    TRAIN_ON_ALL,
    TRAIN_ON_LAST,
    build_conversation_samples,
    render_conversation,
)
from tests._codec_fakes import DummyCodec


def _build(
    messages: list[dict[str, str]], *, train_on_what: str = TRAIN_ON_LAST, max_length: int = 10_000
) -> list[Sample]:
    return build_conversation_samples(messages, codec=DummyCodec(), train_on_what=train_on_what, max_length=max_length)


CONVERSATION = [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "ok"},
]


# 23 tokens render in full ("user: hi\n" + "assistant: " + "ok\n"); 22 clips the
# stop token off the tail while leaving the body trained. Clipping happens in
# token space, before the trim, so alignment has to survive it.
@pytest.mark.parametrize("max_length", [10_000, 22], ids=["untruncated", "truncated"])
def test_the_payload_matches_the_rendered_tokens(max_length: int) -> None:
    """The rendered token run, trimmed once, is what goes on the wire."""
    [sequence] = render_conversation(
        CONVERSATION, codec=DummyCodec(), train_on_what=TRAIN_ON_LAST, max_length=max_length
    )
    [sample] = _build(CONVERSATION, max_length=max_length)

    tokens = sequence.tokens
    # model_input by value, not by length: it is the one array that narrows from
    # the tail, so a truncation change could move it without touching the rest.
    assert sample["model_input"]["chunks"][0]["encoded_text"]["tokens"] == tokens[:-1]
    assert sample["loss_fn_inputs"]["target_tokens"]["data"] == tokens[1:]
    assert sample["loss_fn_inputs"]["weights"]["data"] == [float(w) for w in sequence.loss_mask[1:]]


MULTI_TURN = [
    {"role": "user", "content": "q1"},
    {"role": "assistant", "content": "a1"},
    {"role": "user", "content": "q2"},
    {"role": "assistant", "content": "a2"},
]


@pytest.mark.parametrize(
    ("messages", "train_on_what", "expected"),
    [
        pytest.param(CONVERSATION, TRAIN_ON_LAST, "ok\n", id="last-turn"),
        pytest.param(MULTI_TURN, TRAIN_ON_ALL, "a1\na2\n", id="every-assistant-turn"),
    ],
)
def test_the_trained_slots_decode_to_the_assistant_body(
    messages: list[dict[str, str]], train_on_what: str, expected: str
) -> None:
    """Headers stay untrained, and the body's final token is a trained slot."""
    [sample] = _build(messages, train_on_what=train_on_what)

    tokens = sample["loss_fn_inputs"]["target_tokens"]["data"]
    weights = sample["loss_fn_inputs"]["weights"]["data"]
    trained = "".join(chr(token) for token, weight in zip(tokens, weights, strict=True) if weight)

    assert trained == expected
    assert weights[-1] == 1.0


def test_a_sequence_with_no_trained_token_is_dropped(caplog: pytest.LogCaptureFixture) -> None:
    """A prompt longer than the budget leaves nothing to train on.

    The builder rejects such a payload, so this path has to drop the sequence
    rather than submit a sample that costs a forward pass and teaches nothing.
    """
    prompt_only = len("user: hi\nassistant: ")

    with caplog.at_level("WARNING"):
        samples = _build(CONVERSATION, max_length=prompt_only)

    assert samples == []
    assert "trains no slot" in caplog.text
