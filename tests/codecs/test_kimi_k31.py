"""Kimi-K3.1 codec: every render matches, byte for byte, what K3.1's released encoder renders.

``fixtures/kimi_k31_encoder_renders.json`` holds that encoder's output for each case: the text, and the token
ids as a count and a sha256. The cases cover a tool loop turn by turn (each prompt, then the whole episode),
reasoning kept on turns before the latest user message, every thinking effort the codec accepts, thinking off,
and the argument, reasoning, tool-result and attribute shapes the encoder normalizes. The format and its parser
are Kimi-K3's, covered by ``test_kimi_k3.py`` and the shared suite in ``test_codec.py``.
"""

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path

import pytest

from post_training_cookbook.codecs import get_codec
from post_training_cookbook.codecs.base import MessageCodec
from tests.codecs import REFERENCE_MODELS, load_reference_codec

_FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "kimi_k31_encoder_renders.json").read_text())
_K31 = REFERENCE_MODELS["kimi-k3.1"]


@functools.cache
def _codec(thinking: bool | None, thinking_effort: str | None) -> MessageCodec:
    return load_reference_codec(_K31, enable_thinking=thinking, thinking_effort=thinking_effort)


@pytest.mark.parametrize("case", _FIXTURE["cases"], ids=[case["id"] for case in _FIXTURE["cases"]])
def test_renders_like_the_released_encoder(case: dict):
    codec = _codec(case["thinking"], case["thinking_effort"])
    tokens = codec.encode_messages(
        case["messages"], tools=case["tools"], add_generation_prompt=case["add_generation_prompt"]
    )
    assert codec.decode(tokens) == case["text"]
    # Equal text can still tokenize differently: the encoder tokenizes each segment on its own.
    digest = hashlib.sha256(",".join(map(str, tokens)).encode()).hexdigest()
    assert (len(tokens), digest) == (case["num_tokens"], case["token_ids_sha256"])


def test_medium_effort_is_rejected():
    """K3.1's encoder has a ``medium`` effort that the pinned Kimi-K3 encoder cannot render."""
    with pytest.raises(ValueError, match="Kimi-K3.1 thinking_effort"):
        get_codec(_K31, thinking_effort="medium")
