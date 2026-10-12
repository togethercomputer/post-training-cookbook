"""Tests for the chat-completions adapter: request encoding and response decoding."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from post_training_cookbook.serverless_eval.completions import (
    _response_to_message,
    _to_api_messages,
    _to_tools_param,
    complete_turn,
)
from post_training_cookbook.types import Message, ToolCall, ToolCallFunction, ToolSpec


def _assistant_call(name: str, arguments: str, *, call_id: str | None = "orig") -> Message:
    return Message(
        role="assistant",
        tool_calls=[ToolCall(id=call_id, function=ToolCallFunction(name=name, arguments=arguments))],
    )


def test_plain_messages_pass_through():
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="hi"),
        Message(role="assistant", content="hello"),
    ]
    assert _to_api_messages(messages) == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_tool_call_ids_are_replayed_as_the_endpoint_issued_them():
    messages = [
        Message(
            role="assistant",
            tool_calls=[
                ToolCall(id="a", function=ToolCallFunction(name="f", arguments="{}")),
                ToolCall(id="b", function=ToolCallFunction(name="g", arguments="{}")),
            ],
        ),
        Message(role="tool", content="second", tool_call_id="b"),
        Message(role="tool", content="first", tool_call_id="a"),
    ]
    api = _to_api_messages(messages)

    assert [tc["id"] for tc in api[0]["tool_calls"]] == ["a", "b"]
    assert api[1]["tool_call_id"] == "b"
    assert api[2]["tool_call_id"] == "a"


def test_ids_are_generated_when_the_model_returned_none():
    messages = [
        _assistant_call("f", "{}", call_id=None),
        Message(role="tool", content="r1"),
        _assistant_call("g", "{}", call_id=None),
        Message(role="tool", content="r2"),
    ]
    api = _to_api_messages(messages)

    assert api[0]["tool_calls"][0]["id"] == "call_0"
    assert api[1]["tool_call_id"] == "call_0"
    assert api[2]["tool_calls"][0]["id"] == "call_1"
    assert api[3]["tool_call_id"] == "call_1"


def test_reasoning_content_is_dropped_from_request():
    messages = [Message(role="assistant", content="answer", reasoning_content="hidden chain of thought")]
    assert _to_api_messages(messages) == [{"role": "assistant", "content": "answer"}]


def test_to_tools_param_shape():
    tools = [ToolSpec(name="lookup", description="look it up", parameters={"type": "object"})]
    assert _to_tools_param(tools) == [
        {
            "type": "function",
            "function": {"name": "lookup", "description": "look it up", "parameters": {"type": "object"}},
        }
    ]


@dataclass
class _FakeFunction:
    name: str
    arguments: str


@dataclass
class _FakeToolCall:
    id: str | None
    function: _FakeFunction


@dataclass
class _FakeChoiceMessage:
    content: str | None = None
    reasoning: str | None = None
    tool_calls: list[_FakeToolCall] | None = None


_QWEN35 = "Qwen/Qwen3.5-0.8B"


def test_response_to_message_keeps_content_and_reasoning():
    msg = _response_to_message(_FakeChoiceMessage(content="hi", reasoning="because"), [], _QWEN35)
    assert msg == {"role": "assistant", "content": "hi", "reasoning_content": "because"}


def test_response_to_message_keeps_tool_calls():
    choice = _FakeChoiceMessage(
        tool_calls=[_FakeToolCall(id="call_7", function=_FakeFunction(name="lookup", arguments='{"q": 1}'))]
    )
    msg = _response_to_message(choice, [], _QWEN35)
    assert msg["tool_calls"] == [ToolCall(id="call_7", function=ToolCallFunction(name="lookup", arguments='{"q": 1}'))]


def test_response_to_message_normalizes_a_blank_tool_call_id():
    blank = _response_to_message(
        _FakeChoiceMessage(tool_calls=[_FakeToolCall(id="", function=_FakeFunction(name="f", arguments="{}"))]),
        [],
        _QWEN35,
    )
    assert blank["tool_calls"][0]["id"] is None


_LOOKUP = ToolSpec(
    name="lookup",
    description="look an order up",
    parameters={"type": "object", "properties": {"zip": {"type": "string"}, "count": {"type": "integer"}}},
)

_LEAKED_CALL = (
    "I should look it up.\n"
    "<tool_call>\n<function=lookup>\n"
    "<parameter=zip>\n32286\n</parameter>\n<parameter=count>\n3\n</parameter>\n"
    "</function>\n</tool_call>"
)

_EXPECTED_CALL = ToolCall(id=None, function=ToolCallFunction(name="lookup", arguments='{"zip": "32286", "count": 3}'))
_PARSED_CALL = ToolCall(id="call_7", function=ToolCallFunction(name="lookup", arguments='{"q": 1}'))


def _mixed_choice() -> _FakeChoiceMessage:
    return _FakeChoiceMessage(
        content=_LEAKED_CALL,
        tool_calls=[_FakeToolCall(id="call_7", function=_FakeFunction(name="lookup", arguments='{"q": 1}'))],
    )


def test_parsed_tool_calls_win_over_leaked_by_default():
    msg = _response_to_message(_mixed_choice(), [_LOOKUP], _QWEN35)
    assert msg["tool_calls"] == [_PARSED_CALL]
    assert msg["content"] == _LEAKED_CALL


def test_append_unparsed_tool_calls_keeps_parsed_and_recovers_leaked():
    msg = _response_to_message(_mixed_choice(), [_LOOKUP], _QWEN35, append_unparsed_tool_calls=True)
    assert msg["tool_calls"] == [_PARSED_CALL, _EXPECTED_CALL]
    assert msg["content"] == _LEAKED_CALL


def test_recovers_tool_call_from_reasoning():
    msg = _response_to_message(_FakeChoiceMessage(content="", reasoning=_LEAKED_CALL), [_LOOKUP], _QWEN35)
    assert msg["tool_calls"] == [_EXPECTED_CALL]


def test_recovers_tool_call_from_content():
    msg = _response_to_message(_FakeChoiceMessage(content=_LEAKED_CALL), [_LOOKUP], _QWEN35)
    assert msg["tool_calls"] == [_EXPECTED_CALL]
    assert msg["content"] == "I should look it up."


def test_closed_reasoning_call_remains_deliberation_when_content_has_prose():
    msg = _response_to_message(_FakeChoiceMessage(content="here you go", reasoning=_LEAKED_CALL), [_LOOKUP], _QWEN35)
    assert "tool_calls" not in msg
    assert msg["content"] == "here you go"
    assert msg["reasoning_content"] == _LEAKED_CALL


def test_recovers_only_the_call_outside_closed_reasoning():
    msg = _response_to_message(_FakeChoiceMessage(content=_LEAKED_CALL, reasoning=_LEAKED_CALL), [_LOOKUP], _QWEN35)
    assert msg["tool_calls"] == [_EXPECTED_CALL]


def test_recovers_every_leaked_call():
    msg = _response_to_message(_FakeChoiceMessage(content="", reasoning=_LEAKED_CALL * 2), [_LOOKUP], _QWEN35)
    assert msg["tool_calls"] == [_EXPECTED_CALL, _EXPECTED_CALL]


def test_recovery_without_schemas_untypes_args():
    """Why the tools must be threaded through: unschemaed, a zip code decays to an int."""
    untyped = _response_to_message(_FakeChoiceMessage(content="", reasoning=_LEAKED_CALL), [], _QWEN35)
    assert untyped["tool_calls"][0]["function"]["arguments"] == '{"zip": 32286, "count": 3}'


@pytest.mark.parametrize(
    "model, leaked",
    [
        (_QWEN35, _LEAKED_CALL),
        (
            "zai-org/GLM-5.3",
            "<tool_call>lookup<arg_key>zip</arg_key><arg_value>32286</arg_value>"
            "<arg_key>count</arg_key><arg_value>3</arg_value></tool_call>",
        ),
    ],
    ids=["qwen3.5", "glm-5"],
)
def test_recovery_uses_the_model_codec(model, leaked):
    msg = _response_to_message(_FakeChoiceMessage(content=leaked), [_LOOKUP], model)
    assert msg["tool_calls"] == [_EXPECTED_CALL]


def test_glm_reasoning_only_call_stays_reasoning():
    """GLM-5.3 always opens `<think>`, so reasoning with no content never left the think block: a call
    written there is deliberation, as it is when training parses the same turn."""
    leaked = "<tool_call>lookup<arg_key>zip</arg_key><arg_value>32286</arg_value></tool_call>"
    msg = _response_to_message(_FakeChoiceMessage(content="", reasoning=leaked), [_LOOKUP], "zai-org/GLM-5.3")
    assert "tool_calls" not in msg
    assert msg["reasoning_content"] == leaked


def test_unknown_model_leaves_leaked_call_unparsed():
    choice = _FakeChoiceMessage(content="", reasoning=_LEAKED_CALL)
    assert "tool_calls" not in _response_to_message(choice, [_LOOKUP], "meta-llama/Llama-3-8B")


@pytest.mark.parametrize(
    "content, reasoning",
    [
        ("", f"<think>\nmaybe {_LEAKED_CALL}\n</think>\nstill deciding"),
        ("", "<tool_call>\n<function=lookup>\n<parameter=zip>\n32286\n"),
    ],
    ids=["deliberating in a closed think block", "truncated block"],
)
def test_reasoning_not_mined(content, reasoning):
    choice = _FakeChoiceMessage(content=content, reasoning=reasoning)
    assert "tool_calls" not in _response_to_message(choice, [_LOOKUP], _QWEN35)


def test_unclosed_think_is_mined():
    """The exclusion above keys off the closing tag, so an unterminated think block leaks
    through. No endpoint splitting reasoning into its own field has produced that shape;
    recorded so the asymmetry reads as a known choice rather than a surprise."""
    unclosed = f"<think>\nmaybe {_LEAKED_CALL}"
    assert _response_to_message(_FakeChoiceMessage(content="", reasoning=unclosed), [_LOOKUP], _QWEN35)["tool_calls"]


def test_tool_result_referencing_no_call_raises():
    with pytest.raises(ValueError, match=r"messages\[0\] is a tool result"):
        _to_api_messages([Message(role="tool", content="orphan")])


def test_stale_generated_ids_do_not_span_assistant_turns():
    messages = [
        _assistant_call("f", "{}", call_id=None),
        _assistant_call("g", "{}", call_id=None),
        _assistant_call("h", "{}", call_id=None),
        Message(role="tool", content="for h"),
    ]
    api = _to_api_messages(messages)

    assert api[2]["tool_calls"][0]["id"] == "call_2"
    assert api[3]["tool_call_id"] == "call_2"


def test_tool_result_forwards_name_when_present():
    messages = [
        _assistant_call("lookup", "{}"),
        Message(role="tool", content="ok", tool_call_id="orig", name="lookup"),
    ]
    assert _to_api_messages(messages)[1]["name"] == "lookup"


@pytest.mark.asyncio
async def test_complete_turn_forwards_tools_and_sampling():
    create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=_FakeChoiceMessage(content="ok"), finish_reason="stop")]
        )
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    tools = [ToolSpec(name="lookup", description="look it up", parameters={"type": "object"})]
    await complete_turn(
        client,
        "m",
        [Message(role="user", content="hi")],
        tools,
        temperature=0.2,
        max_tokens=16,
        enable_thinking=True,
    )

    kwargs = create.await_args.kwargs
    assert kwargs["tools"] == _to_tools_param(tools)
    assert kwargs["max_tokens"] == 16
    assert kwargs["chat_template_kwargs"] == {"enable_thinking": True}
