"""OpenAI-compatible chat-completions adapter: structured messages in, a Message out.

Converts our ``Message``/``ToolSpec`` ledger into an OpenAI-style request, calls a
hosted chat-completions endpoint (e.g. Together serverless), and converts the
structured response back into our ``Message`` shape.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

from post_training_cookbook.codecs import parse_completion
from post_training_cookbook.types import Message, ToolCall, ToolCallFunction, ToolSpec

if TYPE_CHECKING:
    from together import AsyncTogether
    from together.types.chat.chat_completion_message import ChatCompletionMessage


def _to_tools_param(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    """Map our ToolSpecs to OpenAI-compatible request ``tools`` entries."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            },
        }
        for tool in tools
    ]


def _to_api_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert our message ledger to OpenAI-compatible request messages.

    The API requires every ``tool`` message to cite an ``assistant.tool_calls``
    id. We replay ids the endpoint already issued. When a call has none, we
    generate ``call_{n}``, put it on that call, and keep it in
    ``pending_generated_ids`` until the matching id-less tool result in this
    turn claims it.

    A tool result with neither a stored id nor a generated id raises an exception.

    Generated ids are unique across the whole request. The pending list is per
    assistant turn, so an unused generated id cannot attach to a later turn.

    Prior-turn ``reasoning_content`` is intentionally dropped (not resent).
    """
    api_messages: list[dict[str, Any]] = []
    pending_generated_ids: list[str] = []
    n_generated = 0

    for index, message in enumerate(messages):
        role = message["role"]
        tool_calls = message.get("tool_calls")

        if role == "assistant" and tool_calls:
            pending_generated_ids.clear()
            entries: list[dict[str, Any]] = []
            for position, tool_call in enumerate(tool_calls):
                call_id = tool_call.get("id")
                if not call_id:
                    call_id = f"call_{n_generated}"
                    n_generated += 1
                    pending_generated_ids.append(call_id)
                entries.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "index": position,
                        "function": {
                            "name": tool_call["function"]["name"],
                            "arguments": tool_call["function"]["arguments"],
                        },
                    }
                )
            api_messages.append({"role": "assistant", "content": message.get("content"), "tool_calls": entries})
        elif role == "tool":
            tool_call_id = message.get("tool_call_id")
            if not tool_call_id:
                if not pending_generated_ids:
                    raise ValueError(f"messages[{index}] is a tool result referencing no preceding tool call")
                tool_call_id = pending_generated_ids.pop(0)
            payload: dict[str, Any] = {
                "role": "tool",
                "content": message.get("content", ""),
                "tool_call_id": tool_call_id,
            }
            if name := message.get("name"):
                payload["name"] = name
            api_messages.append(payload)
        else:
            api_messages.append({"role": role, "content": message.get("content", "")})

    return api_messages


def _endpoint_message(choice: ChatCompletionMessage) -> Message:
    message = Message(role="assistant")
    if choice.content is not None:
        message["content"] = choice.content
    if choice.reasoning is not None:
        message["reasoning_content"] = choice.reasoning
    if choice.tool_calls:
        message["tool_calls"] = [
            ToolCall(
                id=tool_call.id or None,
                function=ToolCallFunction(name=tool_call.function.name, arguments=tool_call.function.arguments),
            )
            for tool_call in choice.tool_calls
        ]
    return message


def _recovered_message(choice: ChatCompletionMessage, tools: list[ToolSpec], model: str) -> Message | None:
    try:
        parsed, _ = parse_completion(
            model,
            content=choice.content,
            reasoning=choice.reasoning,
            tools=tools,
        )
    except ValueError:
        return None
    return parsed


def _response_to_message(
    choice: ChatCompletionMessage,
    tools: list[ToolSpec],
    model: str,
    *,
    append_unparsed_tool_calls: bool = False,
) -> Message:
    """Build our assistant Message from an endpoint's ``ChatCompletionMessage``.

    Structured ``tool_calls`` win by default. Pass ``append_unparsed_tool_calls``
    to also recover leaked ``<tool_call>`` blocks and append them.
    """
    endpoint = _endpoint_message(choice)
    structured = endpoint.get("tool_calls")
    if structured and not append_unparsed_tool_calls:
        return endpoint

    recovered = _recovered_message(choice, tools, model)
    extra = recovered.get("tool_calls") if recovered else None
    if structured and extra:
        return Message(**{**endpoint, "tool_calls": structured + extra})
    return recovered if recovered and not structured else endpoint


class Turn(NamedTuple):
    """One assistant turn plus why the endpoint stopped generating it."""

    message: Message
    finish_reason: str | None


async def complete_turn(
    client: AsyncTogether,
    model: str,
    messages: list[Message],
    tools: list[ToolSpec],
    *,
    temperature: float,
    max_tokens: int | None,
    enable_thinking: bool | None,
    append_unparsed_tool_calls: bool = False,
) -> Turn:
    """Request one assistant turn from an OpenAI-compatible chat-completions endpoint."""
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": _to_api_messages(messages),
        "temperature": temperature,
    }
    if tools:
        kwargs["tools"] = _to_tools_param(tools)
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if enable_thinking is not None:
        kwargs["chat_template_kwargs"] = {"enable_thinking": enable_thinking}

    resp = await client.chat.completions.create(**kwargs)
    choice = resp.choices[0]
    return Turn(
        _response_to_message(choice.message, tools, model, append_unparsed_tool_calls=append_unparsed_tool_calls),
        choice.finish_reason,
    )
