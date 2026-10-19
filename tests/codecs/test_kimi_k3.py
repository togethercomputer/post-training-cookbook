"""Kimi-K3 codec: what the shared contract in test_codec.py cannot phrase generically.

The shared suite parses turns the encoder renders and re-renders the sampled ones built here
(``KIMI_K3_SAMPLED_TURNS``: literals the encoder keeps verbatim). These tests cover other shapes the model
samples but the encoder never renders: a turn with no response channel, wire-typed arguments, and
malformed tool sections.
"""

from __future__ import annotations

import json

import pytest

from post_training_cookbook.codecs import KimiK3Codec, get_codec
from post_training_cookbook.types import Message, ToolSpec
from tests.codecs import REFERENCE_MODELS

# The wire markers, spelled out so the tests pin the format rather than the codec's constants.
_OPEN, _CLOSE, _SEP, _END_OF_MSG = "<|open|>", "<|close|>", "<|sep|>", "<|end_of_msg|>"

KIMI_K3_TOOLS: list[ToolSpec] = [
    ToolSpec(
        name="fetch_holder",
        description="Look up a holder record.",
        parameters={
            "type": "object",
            "properties": {"holder_id": {"type": "string"}, "n": {"type": "integer"}, "flag": {"type": "boolean"}},
            "required": ["holder_id"],
        },
    )
]


def _open(name: str, **attrs: str) -> str:
    return _OPEN + name + "".join(f' {key}="{value}"' for key, value in attrs.items()) + _SEP


def _close(name: str) -> str:
    return f"{_CLOSE}{name}{_SEP}"


def _arg(key: str, type_: str, value: str) -> str:
    return _open("argument", key=key, type=type_) + value + _close("argument")


def _call(*args: str, **attrs: str) -> str:
    return _open("call", **({"tool": "fetch_holder", "index": "1"} | attrs)) + "".join(args) + _close("call")


def _tools(*calls: str) -> str:
    return _open("tools") + "".join(calls) + _close("tools")


_END = _close("message") + _END_OF_MSG


@pytest.fixture(scope="session")
def kimi() -> KimiK3Codec:
    return get_codec(REFERENCE_MODELS["kimi-k3"])


# Turns as the model samples them, for the shared round trip in test_codec.py. The encoder keeps argument
# literals and text verbatim, so canonicalizing either (2.50 → 2.5, stripping) would split the episode there.
KIMI_K3_SAMPLED_TURNS: dict[bool | None, dict[str, str]] = {
    None: {
        "argument-literals": "r"
        + _close("think")
        + _open("response")
        + _close("response")
        + _tools(
            _call(
                _arg("holder_id", "string", "H-1"),
                _arg("n", "number", "3"),
                _arg("ratio", "number", "2.50"),
                _arg("big", "number", "1e3"),
                _arg("meta", "object", '{"k":1,"s":"é"}'),
            )
        )
        + _END,
        "edge-whitespace": "\n  edge whitespace\n"
        + _close("think")
        + _open("response")
        + "  two\n\nparagraphs \n"
        + _close("response")
        + _END,
    }
}


def _args(msg: Message) -> list[dict[str, object]]:
    return [json.loads(call["function"]["arguments"]) for call in msg.get("tool_calls", [])]


@pytest.mark.parametrize(
    "sampled, reasoning, content, arguments",
    [
        pytest.param(
            "look it up" + _tools(_call(_arg("holder_id", "string", "H-1"))) + _END,
            "look it up" + _tools(_call(_arg("holder_id", "string", "H-1"))),
            None,
            [],
            id="tools-inside-open-think-are-reasoning",
        ),
        pytest.param(
            "say "
            + _open("response")
            + "x"
            + _close("response")
            + _close("think")
            + _open("response")
            + "done"
            + _END,
            "say " + _open("response") + "x" + _close("response"),
            "done",
            [],
            id="literal-markers-inside-think",
        ),
        pytest.param(
            "reasoned" + _open("response") + "answer",
            "reasoned" + _open("response") + "answer",
            None,
            [],
            id="response-inside-open-think-is-reasoning",
        ),
        pytest.param(
            _close("think") + _tools(_call(_open("json", type="object") + '{"holder_id": "H-1"}' + _close("json"))),
            None,
            None,
            [{"holder_id": "H-1"}],
            id="json-element-arguments",
        ),
        pytest.param(
            _close("think")
            + _tools(
                _call(_arg("holder_id", "number", "32286"), _arg("n", "string", "5"), _arg("b", "boolean", "True"))
            ),
            None,
            None,
            [{"holder_id": 32286, "n": "5", "b": "True"}],
            id="wire-type-beats-declared-type",
        ),
        pytest.param(
            _close("think")
            + _tools(_call(_arg("a", "number", "3"), _arg("b", "null", "null"), _arg("c", "string", "4"))),
            None,
            None,
            [{"a": 3, "b": None, "c": "4"}],
            id="wire-type-without-a-schema",
        ),
        pytest.param(
            _close("think")
            + _tools(
                _call(_arg("x", "number", "NaN"), _arg("y", "number", "Infinity"), _arg("z", "object", '{"k": 1e999}'))
            ),
            None,
            None,
            [{"x": "NaN", "y": "Infinity", "z": '{"k": 1e999}'}],
            id="non-finite-numbers-stay-text",
        ),
    ],
)
def test_sampled_shapes_parse(
    kimi: KimiK3Codec,
    sampled: str,
    reasoning: str | None,
    content: str | None,
    arguments: list[dict[str, object]],
):
    msg, ok = kimi.parse_response(kimi.encode(sampled), tools=KIMI_K3_TOOLS)
    assert ok
    assert (msg.get("reasoning_content"), msg.get("content")) == (reasoning, content)
    # Compared as JSON text: `True == 1` and `3 == 3.0` in Python, but not to the tool.
    assert json.dumps(_args(msg)) == json.dumps(arguments)


def test_cut_mid_reasoning_keeps_it():
    """Thinking on, the prompt leaves the think channel open, so a turn cut by the token budget
    before any marker is reasoning."""
    kimi = get_codec(REFERENCE_MODELS["kimi-k3"], enable_thinking=True)
    assert kimi.parse_response(kimi.encode("still thinking")) == (
        Message(role="assistant", reasoning_content="still thinking"),
        True,
    )


def test_endpoint_split_completion_parses():
    """Endpoint completions arrive with reasoning already split off into its own field."""
    content = _tools(_call(_arg("n", "number", "2")))
    msg, ok = KimiK3Codec.parse_completion(content=content, reasoning="r", tools=KIMI_K3_TOOLS)
    assert ok
    assert (msg["reasoning_content"], _args(msg)) == ("r", [{"n": 2}])


@pytest.mark.parametrize("content", [None, "", "\n"])
def test_endpoint_reasoning_without_content_stays_reasoning(content: str | None):
    """A completion cut off while thinking has no answer: its reasoning is never graded as one."""
    msg, ok = KimiK3Codec.parse_completion(content=content, reasoning="still thinking", tools=KIMI_K3_TOOLS)
    assert ok
    assert msg["reasoning_content"] == "still thinking"
    assert not msg.get("content", "").strip()
    assert "tool_calls" not in msg


def test_unclosed_call_fails(kimi: KimiK3Codec):
    tools_section = _open("tools") + _open("call", tool="fetch_holder", index="1") + _close("tools")
    msg, ok = kimi.parse_response(kimi.encode(_close("think") + tools_section + _END), tools=KIMI_K3_TOOLS)
    assert not ok
    assert "tool_calls" not in msg


@pytest.mark.parametrize(
    "bad_call",
    [
        _call(tool=""),
        _call(_open("json", type="object") + "{not json" + _close("json")),
        _call(_open("json", type="object") + "[1, 2]" + _close("json")),
        _call(_open("json", type="object") + '{"holder_id": NaN}' + _close("json")),
        _OPEN + 'call tool=fetch_holder index="1"' + _SEP + _close("call"),
    ],
    ids=["no-tool-name", "malformed-json", "non-object-json", "non-finite-json", "unquoted-attribute"],
)
def test_malformed_call_is_dropped(kimi: KimiK3Codec, bad_call: str):
    """Serving drops a complete but malformed call and runs the rest; the turn is not truncated."""
    sampled = _close("think") + _tools(bad_call, _call(_arg("n", "number", "2"))) + _END
    msg, ok = kimi.parse_response(kimi.encode(sampled), tools=KIMI_K3_TOOLS)
    assert ok
    assert json.dumps(_args(msg)) == json.dumps([{"n": 2}])


def test_call_marker_text_in_a_value_is_not_a_truncation(kimi: KimiK3Codec):
    sampled = _close("think") + _tools(_call(_arg("holder_id", "string", "see " + _OPEN + "call"))) + _END
    msg, ok = kimi.parse_response(kimi.encode(sampled), tools=KIMI_K3_TOOLS)
    assert ok
    assert _args(msg) == [{"holder_id": "see " + _OPEN + "call"}]


@pytest.mark.parametrize(
    "sampled",
    [
        "think a bit" + _CLOSE,
        "think a bit" + _CLOSE + "think",
        "r" + _close("think") + _open("response") + "ok" + _CLOSE + "response",
        "r" + _close("think") + _open("response") + "ok" + _close("response") + _CLOSE + "message",
    ],
    ids=["think-close", "think-close-name", "response-close-name", "message-close-name"],
)
def test_turn_cut_inside_a_marker_fails(kimi: KimiK3Codec, sampled: str):
    """A marker is several tokens, so a length cut can land inside one: that half-marker is a
    truncation even in a text channel, where a cut between markers parses ok."""
    assert not kimi.parse_response(kimi.encode(sampled))[1]
