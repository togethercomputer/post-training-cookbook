"""Build per-side Samples for a DPO preference pair."""

from __future__ import annotations

from typing import TYPE_CHECKING

from post_training_cookbook.codecs import MessageCodec
from post_training_cookbook.sft.train import TRAIN_ON_LAST, build_conversation_samples
from post_training_cookbook.types import Message

if TYPE_CHECKING:
    from together.post_training import Sample


def build_preference_pair(
    *,
    chosen_messages: list[Message],
    rejected_messages: list[Message],
    codec: MessageCodec,
    max_length: int,
) -> tuple[Sample, Sample] | None:
    """Tokenise a (chosen, rejected) preference pair into two SFT-style Samples.

    Each ``*_messages`` value is the full conversation including the user
    prompt and the corresponding assistant response (this matches the
    ``HuggingFaceH4/ultrafeedback_binarized`` schema). The chat template is
    applied with ``train_on_what=TRAIN_ON_LAST`` so the loss-mask is 1 only
    on the final assistant turn and 0 over the shared prompt.

    ``TRAIN_ON_LAST`` yields a single trained turn, so each side renders to
    exactly one sequence.

    Returns:
        The ``(chosen, rejected)`` samples, or ``None`` when either side
        survives ``max_length`` with no trained token — a prompt that fills the
        budget on its own.
    """
    chosen = _render_side(chosen_messages, codec=codec, max_length=max_length)
    rejected = _render_side(rejected_messages, codec=codec, max_length=max_length)
    if chosen is None or rejected is None:
        return None
    return chosen, rejected


def _render_side(messages: list[Message], *, codec: MessageCodec, max_length: int) -> Sample | None:
    """Render one side of a pair, or ``None`` when it trains no slot."""
    rendered = build_conversation_samples(
        messages,
        codec=codec,
        train_on_what=TRAIN_ON_LAST,
        max_length=max_length,
    )
    if not rendered:
        return None
    [side] = rendered  # TRAIN_ON_LAST yields exactly one sequence
    return side
