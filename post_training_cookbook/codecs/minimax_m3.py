"""MiniMax-M3 codec: ``<mm:think>`` reasoning and namespaced-XML tool calls.

The checkpoint's Jinja template renders an assistant turn as::

    ]~b]ai
    <mm:think>...reasoning...</mm:think>...content...]<]minimax[>[<tool_call>
    ]<]minimax[>[<invoke name="NAME">]<]minimax[>[<KEY>VALUE]<]minimax[>[</KEY>]<]minimax[>[</invoke>
    ]<]minimax[>[</tool_call>[e~[

A turn without reasoning opens with a bare ``</mm:think>``. ``[e~[`` is the generation stop and sits
inside every rendered turn, so ``turn_terminator`` is empty. The template takes a ``thinking_mode``
of ``enabled`` (the generation prompt ends inside an open ``<mm:think>``), ``disabled`` (it ends
after ``</mm:think>``) or ``adaptive``, its default, where the model opens either one itself.

The template renders the reasoning of every assistant turn it is given, so re-rendering history is
prefix-stable and ``preserve_thinking=True`` changes nothing; ``False`` drops reasoning from turns
before the last user message first, as a client that does not echo it back would. Reasoning and
content are rendered verbatim, and the parser returns them verbatim.

Argument values carry no type: strings and numbers are written raw, booleans as ``true``/``false``,
objects and arrays as nested elements (``<item>`` per array entry), and ``None`` is omitted. Each
value is coerced against its schema, following ``properties`` and ``items`` into nested values
(``$ref`` is not resolved); a value with no declared type gets best-effort JSON. Only ``[e~[`` and
``]~b]`` are special tokens, so ``decode`` keeps the ``]<]minimax[>[`` namespace marker and the
think tags.
"""

from __future__ import annotations

import json
import re

from transformers import PreTrainedTokenizerBase

from post_training_cookbook.codecs.base import (
    check_thinking_effort,
    load_tokenizer,
    messages_for_template,
    resolve_enable_thinking,
    strip_history_reasoning,
)
from post_training_cookbook.codecs.schema_coercion import (
    coerce_by_schema,
    declared_type,
    non_null_branch,
    param_schemas_by_tool,
)
from post_training_cookbook.types import Message, Role, ToolCall, ToolCallFunction, ToolSpec

_NS = "]<]minimax[>["
_EOS = "[e~["
_THINK_OPEN = "<mm:think>"
_THINK_CLOSE = "</mm:think>"
_TOOL_CALL_OPEN = f"{_NS}<tool_call>"
_TOOL_CALL_CLOSE = f"{_NS}</tool_call>"

_INVOKE_RE = re.compile(
    rf'{re.escape(_NS)}<invoke name="(?P<name>[^"]*)">(?P<body>.*?){re.escape(_NS)}</invoke>', re.DOTALL
)
_TAG_RE = re.compile(rf"{re.escape(_NS)}<(?P<close>/?)(?P<tag>[^<>]*)>")

_THINKING_MODES: dict[bool | None, str | None] = {None: None, True: "enabled", False: "disabled"}


class MiniMaxM3Codec:
    """Encode and parse MiniMax-M3 turns through the checkpoint's own chat template."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        hf_token: str | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        enable_thinking: bool | None = None,
        preserve_thinking: bool | None = None,
        thinking_effort: str | None = None,
    ) -> None:
        check_thinking_effort(thinking_effort, supported=(), family="MiniMax-M3")
        self._strip_history = preserve_thinking is False
        self._tokenizer = load_tokenizer(model_name, tokenizer=tokenizer, hf_token=hf_token)
        enable = resolve_enable_thinking(enable_thinking=enable_thinking, preserve_thinking=preserve_thinking)
        self._thinking_mode = _THINKING_MODES[enable]

    def encode_messages(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        add_generation_prompt: bool = False,
    ) -> list[int]:
        kwargs: dict = {"tokenize": True, "add_generation_prompt": add_generation_prompt}
        if tools:
            kwargs["tools"] = [{"type": "function", "function": t} for t in tools]
        if self._thinking_mode is not None:
            kwargs["thinking_mode"] = self._thinking_mode
        if self._strip_history:
            messages = strip_history_reasoning(messages)
        return self._tokenizer.apply_chat_template(messages_for_template(messages), **kwargs)["input_ids"]

    @staticmethod
    def parse_completion(
        *, content: str | None, reasoning: str | None, tools: list[ToolSpec] | None = None
    ) -> tuple[Message, bool]:
        """Parse an endpoint completion whose reasoning the server already split off.

        Serving never runs anything written in reasoning, so it is kept as reasoning and only
        ``content`` is parsed, as text that starts outside any think block.
        """
        msg, ok = _parse_turn(content or "", param_schemas_by_tool(tools), in_think=False)
        if reasoning:
            msg["reasoning_content"] = reasoning + msg.get("reasoning_content", "")
        return msg, ok

    def parse_response(
        self, response_tokens: list[int], *, tools: list[ToolSpec] | None = None
    ) -> tuple[Message, bool]:
        # Verbatim, so the parser sees where `[e~[` ends the turn: the template writes a newline after it.
        text = str(self._tokenizer.decode(response_tokens))
        return _parse_turn(text, param_schemas_by_tool(tools), in_think=self._thinking_mode == "enabled")

    def turn_terminator(self, next_role: Role | None) -> list[int]:
        return []

    def encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=False))

    def decode(self, tokens: list[int]) -> str:
        return str(self._tokenizer.decode(tokens, skip_special_tokens=True))


class _MalformedValueError(ValueError):
    """An argument value whose nested elements do not balance."""


def _parse_turn(text: str, param_schemas: dict[str, dict], *, in_think: bool) -> tuple[Message, bool]:
    """Split one assistant turn into reasoning, content and tool calls.

    ``in_think`` says the text starts inside an open think block. A turn cut off mid-reasoning or
    mid-content keeps what it produced and parses ok; a tool-call block that never closes, or a
    namespace marker outside any complete block, is a truncation and fails. Text after ``[e~[`` is
    template framing, not part of the turn.
    """
    reasoning, rest = _split_reasoning(text.partition(_EOS)[0], in_think=in_think)
    content: list[str] = []
    calls: list[ToolCall] = []
    ok = True
    pos = 0
    while (start := rest.find(_TOOL_CALL_OPEN, pos)) >= 0:
        content.append(rest[pos:start])
        end = rest.find(_TOOL_CALL_CLOSE, start)
        if end < 0:
            ok = False
            pos = len(rest)
            break
        calls.extend(_parse_calls(rest[start + len(_TOOL_CALL_OPEN) : end], param_schemas))
        pos = end + len(_TOOL_CALL_CLOSE)
    content.append(rest[pos:])
    content_text = "".join(content)
    # A lone marker is a tool-call block cut before it opened.
    if _NS in content_text:
        ok = False

    msg = Message(role="assistant")
    if content_text:
        msg["content"] = content_text
    if reasoning:
        msg["reasoning_content"] = reasoning
    if calls:
        msg["tool_calls"] = calls
    return msg, ok


def _split_reasoning(text: str, *, in_think: bool) -> tuple[str, str]:
    """``(reasoning, rest)``; a think block that never closes is all reasoning, as the serving parser reads it."""
    if text.startswith(_THINK_OPEN):
        text, in_think = text[len(_THINK_OPEN) :], True
    elif text.startswith(_THINK_CLOSE):
        return "", text[len(_THINK_CLOSE) :]
    if not in_think:
        return "", text
    reasoning, _, rest = text.partition(_THINK_CLOSE)
    return reasoning, rest


def _parse_calls(block: str, param_schemas: dict[str, dict]) -> list[ToolCall]:
    """Parse the invokes inside one closed tool-call block.

    As the serving parsers do, an invoke with no name or unbalanced arguments is dropped and text
    around the invokes is ignored; the block closed, so none of that is a truncation.
    """
    calls: list[ToolCall] = []
    for match in _INVOKE_RE.finditer(block):
        name = match.group("name")
        schemas = param_schemas.get(name, {})
        try:
            arguments = {key: _value(raw, schemas.get(key)) for key, raw in _elements(match.group("body"))}
        except _MalformedValueError:
            continue
        if name:
            calls.append(ToolCall(id=None, function=ToolCallFunction(name=name, arguments=json.dumps(arguments))))
    return calls


def _elements(text: str) -> list[tuple[str, str]]:
    """The top-level ``]<]minimax[>[<TAG>…]<]minimax[>[</TAG>`` elements of ``text`` as ``(tag, inner)``.

    Raises ``_MalformedValueError`` when the elements do not balance or text sits between them.
    """
    elements: list[tuple[str, str]] = []
    depth = 0
    tag = ""
    inner_start = last_end = 0
    for match in _TAG_RE.finditer(text):
        closing = bool(match.group("close"))
        if depth == 0:
            if closing or text[last_end : match.start()].strip():
                raise _MalformedValueError(text)
            tag, inner_start, depth = match.group("tag"), match.end(), 1
        elif match.group("tag") == tag:
            depth += -1 if closing else 1
            if depth == 0:
                elements.append((tag, text[inner_start : match.start()]))
                last_end = match.end()
    if depth or text[last_end:].strip():
        raise _MalformedValueError(text)
    return elements


def _value(raw: str, schema: object) -> object:
    """Decode one argument value against its JSON schema; anything but a dict declares nothing."""
    spec = schema if isinstance(schema, dict) else {}
    declared = declared_type(spec)
    if _NS not in raw or declared not in (None, "array", "object"):
        if not raw and declared in ("array", "object"):
            return [] if declared == "array" else {}
        typed = coerce_by_schema(raw, declared)
        try:  # JSON cannot carry NaN or Infinity, which float() and json.loads accept, even nested.
            json.dumps(typed, allow_nan=False)
        except ValueError:
            return raw
        return typed
    children = _elements(raw)
    branch = non_null_branch(spec)
    if declared == "array" or (declared is None and all(tag == "item" for tag, _ in children)):
        return [_value(inner, branch.get("items")) for _, inner in children]
    properties = branch.get("properties") or {}
    return {tag: _value(inner, properties.get(tag)) for tag, inner in children}
