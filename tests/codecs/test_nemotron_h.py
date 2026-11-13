"""Nemotron 3.5 codec: what the shared contract in test_codec.py cannot phrase generically.

The shared suite renders Nemotron 3.5 with thinking off (``SHARED_SUITE_THINKING_OFF``), since its reasoning-free
turns do not extend a thinking-on prompt. These tests cover the thinking modes, that prefix break, the
effort and history-reasoning flags, endpoint-split completions, and the turns the model samples
(``NEMOTRON_35_SAMPLED_TURNS``, round-tripped by the shared suite).
"""

from __future__ import annotations

import json

import pytest

from post_training_cookbook.codecs import Nemotron35Codec, get_codec, parse_completion
from post_training_cookbook.sft.train import TRAIN_ON_ALL, render_conversation
from post_training_cookbook.types import Message, ToolSpec
from tests.codecs import REFERENCE_MODELS, TWO_TURN_CONVERSATION

_MODEL = REFERENCE_MODELS["nemotron-3.5"]
_EOS = "<|im_end|>"

NEMOTRON_35_TOOLS: list[ToolSpec] = [
    ToolSpec(
        name="lookup",
        description="Look up a record.",
        parameters={"type": "object", "properties": {"zip": {"type": "string"}, "n": {"type": "integer"}}},
    )
]


def _call(zip_code: str, n: int) -> str:
    return (
        f"<tool_call>\n<function=lookup>\n<parameter=zip>\n{zip_code}\n</parameter>\n"
        f"<parameter=n>\n{n}\n</parameter>\n</function>\n</tool_call>\n"
    )


# Turns as the model samples them after each thinking mode's generation prompt, through the stop token.
NEMOTRON_35_SAMPLED_TURNS: dict[bool | None, dict[str, str]] = {
    True: {
        "reasoning-then-call": "Need the zip lookup.</think>\n" + _call("02139", 2) + _EOS,
        "reasoning-then-answer": "Found it.</think>Holder H-6666 is in 02139." + _EOS,
    },
    False: {
        "parallel-calls": "\n" + _call("02139", 2) + _call("32286", 3) + _EOS,
        "answer": "Holder H-6666 is in 02139." + _EOS,
    },
}


@pytest.fixture(scope="module")
def nemotron() -> Nemotron35Codec:
    return get_codec(_MODEL)


def _render(codec, messages: list[Message], **kwargs) -> str:
    return codec.decode(codec.encode_messages(messages, **kwargs))


@pytest.mark.parametrize(
    "enable_thinking, prompt_tail",
    [(None, "assistant\n<think>\n"), (True, "assistant\n<think>\n"), (False, "assistant\n<think></think>")],
    ids=["default", "enabled", "disabled"],
)
def test_gen_prompt_follows_thinking_mode(enable_thinking: bool | None, prompt_tail: str):
    """Thinking is on unless turned off."""
    codec = get_codec(_MODEL, enable_thinking=enable_thinking)
    assert _render(codec, [Message(role="user", content="q")], add_generation_prompt=True).endswith(prompt_tail)


def test_reasoning_free_turn_needs_thinking_off_for_sft(nemotron: Nemotron35Codec):
    """With thinking on, a turn without reasoning re-renders as `<think></think>`, not the primed `<think>\\n`,
    so SFT on reasoning-free conversations must turn thinking off."""
    with pytest.raises(ValueError, match="rewrote the assistant prompt"):
        render_conversation(TWO_TURN_CONVERSATION, codec=nemotron, train_on_what=TRAIN_ON_ALL, max_length=4096)
    codec = get_codec(_MODEL, enable_thinking=False)
    assert len(render_conversation(TWO_TURN_CONVERSATION, codec=codec, train_on_what=TRAIN_ON_ALL, max_length=4096))


def test_sampled_call_parses_to_declared_types(nemotron: Nemotron35Codec):
    sampled = NEMOTRON_35_SAMPLED_TURNS[True]["reasoning-then-call"]
    msg, ok = nemotron.parse_response(nemotron.encode(sampled), tools=NEMOTRON_35_TOOLS)
    assert ok
    assert msg["reasoning_content"] == "Need the zip lookup."
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"zip": "02139", "n": 2}


def test_turn_cut_mid_reasoning_is_reasoning(nemotron: Nemotron35Codec):
    msg, ok = nemotron.parse_response(nemotron.encode("Still thinking about"))
    assert ok
    assert msg.get("reasoning_content") == "Still thinking about"
    assert "content" not in msg


_HISTORY: list[Message] = [
    Message(role="user", content="u1"),
    Message(role="assistant", reasoning_content="earlier reasoning", content="a1"),
    Message(role="user", content="u2"),
]


@pytest.mark.parametrize("preserve_thinking, keeps", [(None, False), (False, False), (True, True)])
def test_history_reasoning_follows_preserve_thinking(preserve_thinking: bool | None, keeps: bool):
    """The template truncates history reasoning by default; `preserve_thinking=True` turns that off."""
    codec = get_codec(_MODEL, preserve_thinking=preserve_thinking)
    assert ("earlier reasoning" in _render(codec, _HISTORY, add_generation_prompt=True)) is keeps


def test_medium_effort_marks_only_the_last_user_message():
    codec = get_codec(_MODEL, thinking_effort="medium")
    rendered = _render(codec, _HISTORY, add_generation_prompt=True)
    assert rendered.count("{reasoning effort: efficient}") == 1
    assert "u2\n\n{reasoning effort: efficient}" in rendered


def test_variants_render_with_the_pinned_checkpoint(nemotron: Nemotron35Codec):
    """Every name that routes here renders with the pinned Super VL tokenizer and template."""
    variant = get_codec("nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-FP8")
    assert variant.encode_messages(_HISTORY, add_generation_prompt=True) == nemotron.encode_messages(
        _HISTORY, add_generation_prompt=True
    )


def test_parse_completion_matches_the_sampled_turn(nemotron: Nemotron35Codec):
    """An endpoint that splits reasoning from content parses like the sampled tokens."""
    sampled = NEMOTRON_35_SAMPLED_TURNS[True]["reasoning-then-call"]
    expected = nemotron.parse_response(nemotron.encode(sampled), tools=NEMOTRON_35_TOOLS)
    content = "\n" + _call("02139", 2)
    actual = parse_completion(_MODEL, content=content, reasoning="Need the zip lookup.", tools=NEMOTRON_35_TOOLS)
    assert actual == expected
