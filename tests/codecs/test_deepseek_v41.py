"""DeepSeek-V4.1 specifics the shared contracts in test_codec.py miss: DSML call grammar, numeric effort."""

from __future__ import annotations

import json

import pytest

from post_training_cookbook.codecs import DeepSeekV41Codec, get_codec
from post_training_cookbook.types import Message, ToolSpec
from tests.codecs import REFERENCE_MODELS

_MODEL = REFERENCE_MODELS["deepseek-v4.1-flash"]

_LOOKUP = ToolSpec(
    name="lookup",
    description="look up a record",
    parameters={
        "type": "object",
        "properties": {"zip": {"type": "string"}, "count": {"type": "integer"}, "flag": {"type": "boolean"}},
    },
)


def _param(key: str, value: str, string: str = "true") -> str:
    return f'<｜DSML｜ parameter name="{key}" string="{string}">{value}</｜DSML｜ parameter>'


def _invoke(name: str, *params: str) -> str:
    return f'<｜DSML｜ invoke name="{name}">\n' + "\n".join(params) + "\n</｜DSML｜ invoke>"


def _calls(*invokes: str) -> str:
    """A calls block as the template writes it, with the blank line before it."""
    return "\n\n<｜DSML｜ calls>\n" + "\n".join(invokes) + "\n</｜DSML｜ calls>"


@pytest.fixture(scope="module")
def codec() -> DeepSeekV41Codec:
    return DeepSeekV41Codec(_MODEL)


@pytest.mark.parametrize("field", ["content", "content_blocks"])
@pytest.mark.parametrize(
    "block",
    [
        {"type": "text", "text": "q"},
        {"type": "image_url", "image_url": {"url": "image.png"}},
        {"type": "input_audio", "input_audio": {"data": "audio", "format": "wav"}},
    ],
    ids=["text", "image", "audio"],
)
def test_content_blocks_are_rejected(codec: DeepSeekV41Codec, field: str, block: dict):
    """Fail before a text tokenizer can replace media with placeholders or silently drop it."""
    with pytest.raises(ValueError, match="only string message content"):
        codec.encode_messages([{"role": "user", field: [block]}])


@pytest.mark.parametrize("content", [None, 42, {"type": "text", "text": "q"}], ids=["null", "number", "mapping"])
def test_non_string_content_is_rejected(codec: DeepSeekV41Codec, content: object):
    with pytest.raises(ValueError, match="only string message content"):
        codec.encode_messages([Message(role="user", content=content)])


def test_endpoint_reasoning_never_executes():
    """A call written in endpoint-split reasoning stays reasoning; only content is parsed for calls."""
    reasoning = _calls(_invoke("lookup"))
    msg, ok = DeepSeekV41Codec.parse_completion(content="done", reasoning=reasoning, tools=[_LOOKUP])
    assert ok
    assert msg == Message(role="assistant", content="done", reasoning_content=reasoning)


@pytest.mark.parametrize(
    "parameter, expected",
    [
        # Mistyped on the wire: coerced to the declared type.
        (_param("count", "500"), {"count": 500}),
        (_param("zip", "32286", "false"), {"zip": "32286"}),
        (_param("flag", "true"), {"flag": True}),
        # Not coercible, or undeclared: the wire value stands.
        (_param("count", "many"), {"count": "many"}),
        (_param("other", "[1,2]", "false"), {"other": [1, 2]}),
    ],
)
def test_argument_types(parameter: str, expected: dict):
    msg, ok = DeepSeekV41Codec.parse_text("</think>" + _calls(_invoke("lookup", parameter)), tools=[_LOOKUP])
    assert ok
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == expected


@pytest.mark.parametrize(
    "invoke",
    [
        # The serving parser's tolerances: spacing inside tags, a self-closing invoke, a JSON body.
        '<｜DSML｜ invoke  name="lookup" >\n<｜DSML｜ parameter name="zip"  string="true" >1</｜DSML｜ parameter>'
        "\n</｜DSML｜ invoke>",
        '<｜DSML｜ invoke name="lookup"/>',
        _invoke("lookup", '{"zip": "1"}'),
    ],
)
def test_tolerated_layouts(invoke: str):
    msg, ok = DeepSeekV41Codec.parse_text("</think>" + _calls(invoke), tools=[_LOOKUP])
    assert ok
    assert [call["function"]["name"] for call in msg["tool_calls"]] == ["lookup"]


@pytest.mark.parametrize(
    "rest",
    [
        # V4-style unspaced tags; a bare invoke with no calls block; text after the block.
        '\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name="lookup">\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>',
        _invoke("lookup"),
        _calls(_invoke("lookup")) + "more",
        # A repeated parameter; an empty name.
        _calls(_invoke("lookup", _param("zip", "1"), _param("zip", "2"))),
        _calls(_invoke("")),
    ],
)
def test_malformed_calls_fail(codec: DeepSeekV41Codec, rest: str):
    _, ok = codec.parse_response(codec.encode("r</think>" + rest), tools=[_LOOKUP])
    assert not ok


@pytest.mark.parametrize("effort, budget", [("1", 1), ("88", 88), ("100", 100)])
def test_numeric_effort(codec: DeepSeekV41Codec, effort: str, budget: int):
    numeric = DeepSeekV41Codec(_MODEL, tokenizer=codec._tokenizer, thinking_effort=effort)
    prompt = numeric.decode(numeric.encode_messages([Message(role="user", content="q")], add_generation_prompt=True))
    assert f"Reasoning Effort: {budget} " in prompt


@pytest.mark.parametrize("effort", ["0", "101", "medium", " 75"])
def test_rejects_other_efforts(effort: str):
    with pytest.raises(ValueError, match="thinking_effort"):
        get_codec(_MODEL, thinking_effort=effort)


def test_rejects_another_tokenizer():
    class Other:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            return [7]

    with pytest.raises(ValueError, match="not a DeepSeek-V4.1 tokenizer"):
        DeepSeekV41Codec(tokenizer=Other())
