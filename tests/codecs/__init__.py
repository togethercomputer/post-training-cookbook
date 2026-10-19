"""Shared reference models for the codec test suite.

One reference model per codec *class*, not per prefix: prefixes that alias an
existing codec (e.g. qwen3.6 → Qwen35Codec) share its tokenizer and tool-call
format, and routing for them is covered by the resolve_codec_class tests. Where an
alias's chat template behaves differently (Qwen3.6 and Qwen3.8 take
preserve_thinking natively), the difference is tested against that model directly.
`conftest.py` asserts this mapping leaves no registered codec class unexercised.

The shared contract lives in `test_codec.py`; `test_<family>.py` holds what it
cannot phrase generically. Helpers both use live here.
"""

from __future__ import annotations

import itertools

import pytest
from huggingface_hub.errors import GatedRepoError

from post_training_cookbook.codecs import get_codec
from post_training_cookbook.codecs.base import MessageCodec
from post_training_cookbook.codecs.schema_coercion import param_types_by_tool
from post_training_cookbook.sequence_split import TrainableSequence
from post_training_cookbook.types import Message, ToolSpec

REFERENCE_MODELS: dict[str, str] = {
    "qwen3.5": "Qwen/Qwen3.5-0.8B",
    "glm-5": "zai-org/GLM-5.3",
    "kimi-k3": "moonshotai/Kimi-K3",
    "minimax-m3": "MiniMaxAI/MiniMax-M3",
    "muse-spark": "meta-models/Muse-Spark-Open-816B-A42B",
}

# GLM-5.3's and Muse Spark's templates have no thinking-off mode, so their codecs reject it.
THINKING_OFF_MODELS = [model for family, model in REFERENCE_MODELS.items() if family not in ("glm-5", "muse-spark")]

# Shared by the format-specific arg-parser suites, so every parser is provably
# tested against the identical declared schema.
ARG_SCHEMA = param_types_by_tool(
    [
        ToolSpec(
            name="lookup",
            description="",
            parameters={
                "type": "object",
                "properties": {
                    "zip": {"type": "string"},
                    "count": {"type": "integer"},
                    "flag": {"type": "boolean"},
                    "opt": {"type": "null"},
                    "note": {"type": "string"},
                    "tags": {"type": "array"},
                    "meta": {"type": "object"},
                },
            },
        )
    ]
)

TWO_TURN_CONVERSATION: list[Message] = [
    Message(role="user", content="u1"),
    Message(role="assistant", content="a1"),
    Message(role="user", content="u2"),
    Message(role="assistant", content="a2"),
]


# Gated reference repos this environment's HF token cannot open, so each is only tried once.
_inaccessible_models: set[str] = set()


def load_reference_codec(model: str, **kwargs) -> MessageCodec:
    """`get_codec`, skipping the calling test when the model's Hub repo is gated and not open to this token."""
    if model not in _inaccessible_models:
        try:
            return get_codec(model, **kwargs)
        except OSError as error:
            if not isinstance(error, GatedRepoError) and not isinstance(error.__cause__, GatedRepoError):
                raise
            _inaccessible_models.add(model)
    pytest.skip(f"{model} is gated on the Hub; set HF_TOKEN for an account with access")


def single_token(codec: MessageCodec, text: str) -> int:
    """The one token `text` encodes to — unpacking so a non-special string fails loudly."""
    [token_id] = codec.encode(text)
    return token_id


def trained_runs(sequences: list[TrainableSequence]) -> list[list[int]]:
    """One token run per trained assistant turn, in order, however the conversation split."""
    runs: list[list[int]] = []
    for seq in sequences:
        for trained, group in itertools.groupby(zip(seq.tokens, seq.loss_mask), key=lambda pair: pair[1]):
            if trained:
                runs.append([token for token, _ in group])
    return runs
