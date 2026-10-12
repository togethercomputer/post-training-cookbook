"""Tests for SFT conversation rendering and training weight masks."""

from post_training_cookbook.sft.train import (
    TRAIN_ON_ALL,
    TRAIN_ON_LAST,
    _build_batch_samples,
    messages_column,
    render_conversation,
)
from tests._codec_fakes import DummyCodec


def test_headers_never_trained_on() -> None:
    """Role headers (e.g. 'assistant: ') always get weight 0."""
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ok"},
    ]
    sequences = render_conversation(
        messages,
        codec=DummyCodec(),
        train_on_what=TRAIN_ON_ALL,
        max_length=10_000,
    )
    assert len(sequences) == 1
    weights = sequences[0].loss_mask
    # "user: " = 6 chars (w=0), "hi\n" = 3 chars (w=0), "assistant: " = 11 chars (w=0), "ok\n" = 3 chars (w=1)
    header_weights = weights[:6] + weights[9:20]  # both headers
    assert all(w == 0 for w in header_weights)


def test_train_on_all_trains_every_assistant_turn() -> None:
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "a2"},
    ]
    sequences = render_conversation(
        messages,
        codec=DummyCodec(),
        train_on_what=TRAIN_ON_ALL,
        max_length=10_000,
    )
    assert len(sequences) == 1
    weights = sequences[0].loss_mask
    # Layout: "user: hi\n" (9) | "assistant: a1\n" (14) | "user: go\n" (9) | "assistant: a2\n" (14)
    first_body = weights[9 + 11 : 9 + 14]  # "a1\n"
    second_body = weights[32 + 11 : 32 + 14]  # "a2\n"
    assert first_body == [1, 1, 1]
    assert second_body == [1, 1, 1]


def test_train_on_last_only_trains_final_assistant() -> None:
    messages = [
        {"role": "assistant", "content": "first"},
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "last"},
    ]
    sequences = render_conversation(
        messages,
        codec=DummyCodec(),
        train_on_what=TRAIN_ON_LAST,
        max_length=10_000,
    )
    assert len(sequences) == 1
    weights = sequences[0].loss_mask
    # "assistant: " (11) + "first\n" (6) | "user: " (6) + "x\n" (2) | "assistant: " (11) + "last\n" (5)
    first_body = weights[11:17]  # "first\n" — should NOT be trained on
    last_body = weights[36:41]  # "last\n" — should be trained on
    assert first_body == [0, 0, 0, 0, 0, 0]
    assert last_body == [1, 1, 1, 1, 1]


def test_user_content_never_trained_on() -> None:
    messages = [
        {"role": "user", "content": "secret"},
        {"role": "assistant", "content": "ok"},
    ]
    sequences = render_conversation(
        messages,
        codec=DummyCodec(),
        train_on_what=TRAIN_ON_ALL,
        max_length=10_000,
    )
    assert len(sequences) == 1
    weights = sequences[0].loss_mask
    # "user: " (6) + "secret\n" (7) — all should be 0
    user_weights = weights[:13]
    assert all(w == 0 for w in user_weights)


class _ThinkStrippingCodec(DummyCodec):
    """Codec that drops <think>...</think> blocks from non-final assistant messages.

    Mirrors the real-world chat template behavior where reasoning traces are
    stripped from history once the next turn begins.
    """

    def encode_messages(self, messages, *, tools=None, add_generation_prompt=False):
        cleaned = []
        for idx, msg in enumerate(messages):
            is_last = idx == len(messages) - 1
            content = msg["content"]
            if msg["role"] == "assistant" and not is_last and "<think>" in content:
                pre, _, rest = content.partition("<think>")
                _, _, post = rest.partition("</think>")
                content = pre + post
            cleaned.append({"role": msg["role"], "content": content})
        return super().encode_messages(cleaned, tools=tools, add_generation_prompt=add_generation_prompt)


def test_prefix_mismatch_splits_into_multiple_sequences() -> None:
    """When the codec strips earlier content, render emits one sequence per stable run."""
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "<think>plan</think>a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]
    sequences = render_conversation(
        messages,
        codec=_ThinkStrippingCodec(),
        train_on_what=TRAIN_ON_ALL,
        max_length=10_000,
    )
    # Two sequences: one capturing a1 with its thinking trace, another rooted at
    # the stripped history that captures a2.
    assert len(sequences) == 2

    tokens_a1, weights_a1 = sequences[0].tokens, sequences[0].loss_mask
    decoded_a1 = "".join(chr(t) for t in tokens_a1)
    assert "<think>plan</think>a1" in decoded_a1
    trained_a1 = "".join(chr(t) for t, w in zip(tokens_a1, weights_a1) if w == 1)
    assert "<think>plan</think>a1" in trained_a1
    assert "a2" not in trained_a1

    tokens_a2, weights_a2 = sequences[1].tokens, sequences[1].loss_mask
    decoded_a2 = "".join(chr(t) for t in tokens_a2)
    assert "<think>" not in decoded_a2  # earlier thinking was stripped
    assert "a2" in decoded_a2
    trained_a2 = "".join(chr(t) for t, w in zip(tokens_a2, weights_a2) if w == 1)
    assert trained_a2.endswith("a2\n")
    assert "a1" not in trained_a2


class _NoStopTokenCodec(DummyCodec):
    """Codec with no stop token for one continuation, like GLM-5 on back-to-back
    assistant turns — the only way a row can be unrenderable rather than merely split."""

    def turn_terminator(self, next_role):
        if next_role == "assistant":
            raise ValueError("no stop token for an assistant turn followed by 'assistant'")
        return []


def test_unrenderable_rows_are_dropped_not_fatal() -> None:
    """A row whose turns have no stop token costs that row, not the run — and a batch where
    every row is unrenderable yields no samples, which the caller must not submit."""
    ok_row = {"messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]}
    bad_row = {
        "messages": [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a1"},
            {"role": "assistant", "content": "a2"},
        ]
    }
    kwargs = {
        "to_messages": messages_column,
        "codec": _NoStopTokenCodec(),
        "train_on_what": TRAIN_ON_ALL,
        "max_length": 10_000,
    }

    assert _build_batch_samples([bad_row] * 4, **kwargs) == []
    assert len(_build_batch_samples([bad_row, ok_row], **kwargs)) == 1


def test_train_on_last_with_stripping_emits_single_sequence() -> None:
    """TRAIN_ON_LAST only walks the final assistant turn, so stripping never triggers a split."""
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "<think>plan</think>a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]
    sequences = render_conversation(
        messages,
        codec=_ThinkStrippingCodec(),
        train_on_what=TRAIN_ON_LAST,
        max_length=10_000,
    )
    assert len(sequences) == 1
    tokens, weights = sequences[0].tokens, sequences[0].loss_mask
    trained = "".join(chr(t) for t, w in zip(tokens, weights) if w == 1)
    assert trained.endswith("a2\n")
    assert "a1" not in trained
