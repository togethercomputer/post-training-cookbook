"""MiniMax-M3 codec: what the shared contract in test_codec.py cannot phrase generically.

The shared suite parses turns the template renders and re-renders the sampled ones built here
(``MINIMAX_M3_SAMPLED_TURNS``: verbatim whitespace under each thinking mode). These tests cover the thinking
modes, nested argument values, endpoint-split completions, and other shapes the model samples but
the template never renders: a turn cut mid-reasoning and malformed tool-call blocks.
"""

from __future__ import annotations

import json

import pytest

from post_training_cookbook.codecs import MiniMaxM3Codec, get_codec, parse_completion
from post_training_cookbook.types import Message, ToolCall, ToolCallFunction, ToolSpec
from tests.codecs import REFERENCE_MODELS

_MODEL = REFERENCE_MODELS["minimax-m3"]

# The wire markers, spelled out so the tests pin the format rather than the codec's constants.
_NS, _EOS = "]<]minimax[>[", "[e~["
_THINK_OPEN, _THINK_CLOSE = "<mm:think>", "</mm:think>"

MINIMAX_M3_TOOLS: list[ToolSpec] = [
    ToolSpec(
        name="lookup",
        description="Look up a record.",
        parameters={
            "type": "object",
            "properties": {
                "zip": {"type": "string"},
                "n": {"type": "integer"},
                "tags": {"type": "array"},
                "meta": {"type": "object", "properties": {"id": {"type": "string"}}},
                "ids": {"type": "array", "items": {"type": "string"}},
                "rows": {"type": "array", "items": {"type": "object"}},
                "owner": {"anyOf": [{"type": "object", "properties": {"zip": {"type": "string"}}}, {"type": "null"}]},
            },
        },
    )
]


def _el(tag: str, inner: str) -> str:
    return f"{_NS}<{tag}>{inner}{_NS}</{tag}>"


def _invoke(*args: str, name: str = "lookup") -> str:
    return f'{_NS}<invoke name="{name}">' + "".join(args) + f"{_NS}</invoke>\n"


def _block(*invokes: str) -> str:
    return f"{_NS}<tool_call>\n" + "".join(invokes) + f"{_NS}</tool_call>"


@pytest.fixture(scope="session")
def minimax() -> MiniMaxM3Codec:
    return get_codec(_MODEL)


@pytest.mark.parametrize(
    "enable_thinking, prompt_tail",
    [(None, "\nai\n"), (True, f"\nai\n{_THINK_OPEN}"), (False, f"\nai\n{_THINK_CLOSE}")],
    ids=["adaptive", "enabled", "disabled"],
)
def test_gen_prompt_follows_thinking_mode(enable_thinking: bool | None, prompt_tail: str):
    """`None` keeps the template's adaptive default, where the model opens the think block itself."""
    codec = get_codec(_MODEL, enable_thinking=enable_thinking)
    rendered = codec.decode(codec.encode_messages([Message(role="user", content="q")], add_generation_prompt=True))
    assert rendered.endswith(prompt_tail)


# Turns as the model samples them under each thinking mode, for the shared round trip in test_codec.py.
# Reasoning and content come back verbatim, so a parsed turn re-renders to what was sampled.
MINIMAX_M3_SAMPLED_TURNS: dict[bool | None, dict[str, str]] = {
    None: {
        "edge-whitespace": f"{_THINK_OPEN}\n  edge whitespace\n{_THINK_CLOSE}  two\n\nparagraphs \n{_EOS}",
        "no-reasoning": f"{_THINK_CLOSE}answer{_EOS}",
    },
    True: {
        "reasoning-then-call": f"reasoned{_THINK_CLOSE}calling "
        + _block(_invoke(_el("zip", "32286"), _el("n", "3")))
        + _EOS
    },
    False: {
        "nested-values": _block(
            _invoke(
                _el("tags", _el("item", "a") + _el("item", _el("k", "1"))),
                _el("meta", _el("item", "x")),
            )
        )
        + _EOS
    },
}


def _args(msg: Message) -> list[dict[str, object]]:
    return [json.loads(call["function"]["arguments"]) for call in msg.get("tool_calls", [])]


@pytest.mark.parametrize(
    "sampled, arguments",
    [
        pytest.param(
            _block(_invoke(_el("zip", "32286"), _el("n", "3"))),
            [{"zip": "32286", "n": 3}],
            id="declared-scalar-types",
        ),
        pytest.param(
            _block(_invoke(_el("tags", _el("item", "a") + _el("item", "2")), _el("meta", _el("item", "x")))),
            [{"tags": ["a", 2], "meta": {"item": "x"}}],
            id="declared-containers-beat-tag-names",
        ),
        pytest.param(
            _block(_invoke(_el("tags", ""), _el("meta", ""))),
            [{"tags": [], "meta": {}}],
            id="empty-containers",
        ),
        pytest.param(
            _block(
                _invoke(
                    _el("meta", _el("id", "32286")),
                    _el("ids", _el("item", "32286") + _el("item", "true") + _el("item", "null")),
                    _el("rows", _el("item", "")),
                    _el("owner", _el("zip", "02139")),
                )
            ),
            [{"meta": {"id": "32286"}, "ids": ["32286", "true", "null"], "rows": [{}], "owner": {"zip": "02139"}}],
            id="nested-schemas",
        ),
        pytest.param(
            _block(_invoke(_el("extra", _el("item", "true") + _el("item", _el("deep", "x"))))),
            [{"extra": [True, {"deep": "x"}]}],
            id="undeclared-nested",
        ),
        pytest.param(
            _block(_invoke(_el("tags", _el("item", "1") + _el("item", "NaN")))),
            [{"tags": [1, "NaN"]}],
            id="nested-non-finite-stays-text",
        ),
        pytest.param(
            _block(_invoke(_el("n", "1")), _invoke(_el("n", "2"))),
            [{"n": 1}, {"n": 2}],
            id="two-invokes",
        ),
    ],
)
def test_arguments_parse(minimax: MiniMaxM3Codec, sampled: str, arguments: list[dict[str, object]]):
    msg, ok = minimax.parse_response(minimax.encode(_THINK_CLOSE + sampled + _EOS), tools=MINIMAX_M3_TOOLS)
    assert ok
    assert json.dumps(_args(msg)) == json.dumps(arguments)


def test_nested_args_round_trip(minimax: MiniMaxM3Codec):
    """Nested values the template writes untyped read back as the types their schemas declare."""
    arguments = {"meta": {"id": "32286"}, "ids": ["32286", "true", "null"], "rows": [{}]}
    call = ToolCall(id=None, function=ToolCallFunction(name="lookup", arguments=json.dumps(arguments)))
    prompt = [Message(role="user", content="q")]
    prompt_tokens = minimax.encode_messages(prompt, tools=MINIMAX_M3_TOOLS, add_generation_prompt=True)
    full = minimax.encode_messages(prompt + [Message(role="assistant", tool_calls=[call])], tools=MINIMAX_M3_TOOLS)
    msg, ok = minimax.parse_response(full[len(prompt_tokens) :], tools=MINIMAX_M3_TOOLS)
    assert ok
    assert json.dumps(_args(msg)) == json.dumps([arguments])


def test_tool_call_inside_unclosed_think_is_reasoning(minimax: MiniMaxM3Codec):
    """Serving reads everything after an unclosed `<mm:think>` as reasoning and never executes a
    call written there, so the parser does not either."""
    block = _block(_invoke(_el("n", "1")))
    msg, ok = minimax.parse_response(minimax.encode(_THINK_OPEN + "r" + block + _EOS))
    assert ok
    assert msg == Message(role="assistant", reasoning_content="r" + block)


def test_cut_mid_reasoning_keeps_it():
    """A thinking-mode turn cut by the token budget never closes its think block."""
    codec = get_codec(_MODEL, enable_thinking=True)
    msg, ok = codec.parse_response(codec.encode("still thinking"))
    assert ok
    assert msg == Message(role="assistant", reasoning_content="still thinking")


@pytest.mark.parametrize(
    "sampled",
    [f"{_NS}<tool_call>\n" + _invoke(_el("n", "1")), "answer " + _NS],
    ids=["unclosed-block", "lone-marker"],
)
def test_truncated_tool_calls_fail(minimax: MiniMaxM3Codec, sampled: str):
    _, ok = minimax.parse_response(minimax.encode(_THINK_CLOSE + sampled), tools=MINIMAX_M3_TOOLS)
    assert not ok


@pytest.mark.parametrize(
    "bad",
    [_invoke(_el("n", "1"), name=""), _invoke(f"{_NS}<n>1"), "stray text"],
    ids=["no-tool-name", "unbalanced-argument", "text-in-block"],
)
def test_malformed_invoke_is_dropped(minimax: MiniMaxM3Codec, bad: str):
    """Serving drops a malformed invoke inside a closed block and runs the rest; nothing was cut off."""
    sampled = _THINK_CLOSE + _block(bad, _invoke(_el("n", "2"))) + _EOS
    msg, ok = minimax.parse_response(minimax.encode(sampled), tools=MINIMAX_M3_TOOLS)
    assert ok
    assert json.dumps(_args(msg)) == json.dumps([{"n": 2}])


def test_endpoint_split_completion_parses():
    """Endpoint completions arrive with reasoning already split off into its own field."""
    msg, ok = parse_completion(
        _MODEL, content=_block(_invoke(_el("zip", "02139"))), reasoning="check the zip", tools=MINIMAX_M3_TOOLS
    )
    assert ok
    assert (msg["reasoning_content"], _args(msg)) == ("check the zip", [{"zip": "02139"}])


@pytest.mark.parametrize(
    "content, reasoning, expected",
    [
        (None, "still thinking", Message(role="assistant", reasoning_content="still thinking")),
        ("visible</mm:think>text", None, Message(role="assistant", content="visible</mm:think>text")),
    ],
    ids=["reasoning-only", "content-keeps-a-stray-close"],
)
def test_endpoint_channels_are_not_re_split(content: str | None, reasoning: str | None, expected: Message):
    """The server already split the channels; reasoning is never graded as an answer, and content
    that contains a think close is not mistaken for reasoning."""
    assert parse_completion(_MODEL, content=content, reasoning=reasoning, tools=MINIMAX_M3_TOOLS) == (expected, True)


def test_null_argument_is_omitted(minimax: MiniMaxM3Codec):
    """The template drops a null argument rather than rendering a value that would read back as text."""
    call = ToolCall(id=None, function=ToolCallFunction(name="lookup", arguments='{"zip": "1", "n": null}'))
    rendered = minimax.decode(minimax.encode_messages([Message(role="assistant", tool_calls=[call])]))
    assert _el("zip", "1") in rendered
    assert f"{_NS}<n>" not in rendered
