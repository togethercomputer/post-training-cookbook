from __future__ import annotations

from typing import Literal, Required, TypedDict

Role = Literal["system", "user", "assistant", "tool"]


class ToolCallFunction(TypedDict):
    """Function payload attached to a tool call."""

    name: str
    arguments: str


class ToolCall(TypedDict):
    """Structured assistant tool call in OpenAI/HF chat-template shape."""

    id: str | None
    function: ToolCallFunction


class Message(TypedDict, total=False):
    """Structured chat message shared across codecs, envs, and training code."""

    role: Required[Role]
    content: str
    reasoning_content: str
    tool_calls: list[ToolCall]
    tool_call_id: str
    name: str


class ToolSpec(TypedDict):
    """JSON-schema tool definition passed into chat templates."""

    name: str
    description: str
    parameters: dict
