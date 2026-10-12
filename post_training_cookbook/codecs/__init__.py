from __future__ import annotations

from post_training_cookbook.codecs.base import MessageCodec, PreserveThinkingForkedWarning
from post_training_cookbook.codecs.glm5 import Glm5Codec
from post_training_cookbook.codecs.kimi_k3 import KimiK3Codec
from post_training_cookbook.codecs.minimax_m3 import MiniMaxM3Codec
from post_training_cookbook.codecs.qwen3_5 import Qwen35Codec
from post_training_cookbook.types import Message, ToolSpec

__all__ = [
    "Glm5Codec",
    "KimiK3Codec",
    "MessageCodec",
    "MiniMaxM3Codec",
    "PreserveThinkingForkedWarning",
    "Qwen35Codec",
    "get_codec",
    "parse_completion",
    "resolve_codec_class",
]

# Longest matching prefix wins
_CODECS: list[tuple[str, type[MessageCodec]]] = [
    ("glm-5.3", Glm5Codec),
    ("kimi-k3", KimiK3Codec),
    ("minimax-m3", MiniMaxM3Codec),
    ("qwen3.5", Qwen35Codec),
    ("qwen3.6", Qwen35Codec),
    ("qwen3.8", Qwen35Codec),
]


def resolve_codec_class(model_name: str) -> type[MessageCodec]:
    """Pick the codec class for a HF model name by longest matching prefix."""
    name = model_name.rsplit("/", 1)[-1].lower()
    matches = [(prefix, codec_cls) for prefix, codec_cls in _CODECS if name.startswith(prefix)]
    if not matches:
        supported = [prefix for prefix, _ in _CODECS]
        raise ValueError(f"No codec for model {model_name!r}. Supported prefixes: {supported}")
    return max(matches, key=lambda item: len(item[0]))[1]


def parse_completion(
    model_name: str,
    *,
    content: str | None,
    reasoning: str | None,
    tools: list[ToolSpec] | None = None,
) -> tuple[Message, bool]:
    """Parse an endpoint-split completion with the model family's response grammar."""
    return resolve_codec_class(model_name).parse_completion(content=content, reasoning=reasoning, tools=tools)


def get_codec(
    model_name: str,
    *,
    hf_token: str | None = None,
    enable_thinking: bool | None = None,
    preserve_thinking: bool | None = None,
    thinking_effort: str | None = None,
) -> MessageCodec:
    """Resolve the right codec for a HF model name like 'Qwen/Qwen3.5-9B'.

    ``thinking_effort`` is family-specific (``low``/``high``/``max`` for Kimi-K3 and GLM-5);
    ``None`` keeps the template default, and families without one reject any value.
    """
    return resolve_codec_class(model_name)(
        model_name,
        hf_token=hf_token,
        enable_thinking=enable_thinking,
        preserve_thinking=preserve_thinking,
        thinking_effort=thinking_effort,
    )
