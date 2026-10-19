"""Muse Spark codec: the recipient-channel format of ``meta-models/Muse-Spark-Open-816B-A42B``'s chat template.

An assistant turn is a run of channels, one tool call per channel::

    <|start|>assistant to=self<|message|>reasoning<|eom|>
    <|start|>assistant to=commentary<|message|>visible update<|eom|>
    <|start|>assistant to=NAME<|message|><atem:function_calls>
    <atem:invoke name="NAME">
    <atem:parameter name="K">V</atem:parameter>
    </atem:invoke>
    </atem:function_calls><|eot|>

A final answer is addressed ``to=user``. The generation prompt stops at ``<|start|>assistant``, so a sampled turn
starts mid-header. Sampling runs across ``<|eom|>`` and stops on the ``<|eot|>`` the template writes inside the
turn, so ``turn_terminator`` is empty. The one exception is a turn with reasoning only, which the template ends on
``<|eom|>``; rendering one as the last message raises, so it is never trained without a stop. The markers are special
tokens, so responses are decoded verbatim.

Argument values carry no type on the wire, and the template writes a string verbatim. A value the tool schema
declares a (nullable) string is therefore kept as text; any other is read JSON-or-raw, as the serving parser reads
every value.

The parsed Message is the serving parsers' reading: calls come from tool channels only (commentary is content), a
malformed call is dropped, and names are kept as written. The verdict is strict: a turn the template could not have
written fails, so it is never rewarded as a well-formed action. That is a header other than ``to=RECIPIENT``, a message
that runs into the next header without its ``<|eom|>``, a tool channel that is not one well-formed
``<atem:function_calls>`` block, text outside any message, or a cut-off turn. An endpoint completion is exempt where
the server already did part of the work: its content may be unwrapped prose, and its last tool channel lost
``<|eot|>`` as the stop token.

The template renders the reasoning of every assistant turn it is given, so ``preserve_thinking=True`` changes
nothing and ``False`` drops reasoning before the last user message first. It has no thinking switch, so
``enable_thinking=False`` is rejected. It stamps a date into the system prompt: the codec's construction date by
default, fixed so a run that crosses midnight keeps re-rendering the same prefix.

The checkpoint repo is gated on the Hugging Face Hub: loading the tokenizer needs an ``hf_token`` whose account has
been granted access, or an explicit ``tokenizer``.
"""

from __future__ import annotations

import datetime
import json
import re

from transformers import PreTrainedTokenizerBase

from post_training_cookbook.codecs.base import (
    check_thinking_effort,
    load_tokenizer,
    messages_for_template,
    resolve_enable_thinking,
    strip_history_reasoning,
    with_turn_call_ids,
)
from post_training_cookbook.codecs.schema_coercion import coerce_by_schema, param_types_by_tool
from post_training_cookbook.types import Message, Role, ToolCall, ToolCallFunction, ToolSpec

# Every Muse Spark Open catalog name (e.g. the NVFP4 export) shares this tokenizer and template.
_TOKENIZER_REPO = "meta-models/Muse-Spark-Open-816B-A42B"
_TOKENIZER_REVISION = "0fb02478b02565b2ce7fc57e6051c91172774ac9"
# The template's reasoning_effort_map keys; it raises on any other value and defaults to xhigh.
_THINKING_EFFORTS = ("low", "medium", "high", "xhigh")

_START = "<|start|>"
_MESSAGE = "<|message|>"
_EOM = "<|eom|>"
_EOT = "<|eot|>"
_END_OF_TEXT = "<|end_of_text|>"
_TERMINATOR_RE = re.compile("|".join(map(re.escape, (_EOM, _EOT, _END_OF_TEXT))))
_ROLE_RE = re.compile(r"\s*<\|start\|>(\w+)")
# A channel header as the template writes it; the first one of a sampled turn has no `<|start|>assistant`.
_HEADER_RE = re.compile(r"\s*(?:<\|start\|>assistant)?\s*to=[^\s<]+")
# The recipient anywhere in a channel header, for reading a malformed one; a header without one addresses the user.
_RECIPIENT_RE = re.compile(r"to=(?P<recipient>[^\s<]+)")
# The first header of a sampled turn, cut off inside ``to=NAME``.
_CUT_HEADER_RE = re.compile(r"\s*(?:t|to|to=[^\s<]*)")
_BLOCK_RE = re.compile(r"\s*<atem:function_calls>(?P<invokes>.*)</atem:function_calls>\s*", re.DOTALL)
_INVOKE_RE = re.compile(r'<atem:invoke\b[^>]*?\bname="(?P<name>[^"]+)"[^>]*>(?P<body>.*?)</atem:invoke>', re.DOTALL)
_PARAM_RE = re.compile(
    r'<atem:parameter\b[^>]*?\bname="(?P<key>[^"]+)"[^>]*>(?P<value>.*?)</atem:parameter>', re.DOTALL
)
# The template's attribute escapes, undone with `&amp;` last.
_ENTITIES = (("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"'), ("&#39;", "'"), ("&amp;", "&"))


class MuseSparkCodec:
    """Renders and parses Muse Spark's recipient channels through its release chat template."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        hf_token: str | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        enable_thinking: bool | None = None,
        preserve_thinking: bool | None = None,
        thinking_effort: str | None = None,
        current_date: datetime.date | None = None,
    ) -> None:
        if resolve_enable_thinking(enable_thinking=enable_thinking, preserve_thinking=preserve_thinking) is False:
            raise ValueError("Muse Spark's chat template has no thinking switch; enable_thinking=False is unsupported")
        self._thinking_effort = check_thinking_effort(
            thinking_effort, supported=_THINKING_EFFORTS, family="Muse Spark"
        )
        self._strip_history = preserve_thinking is False
        self._current_date = (current_date if current_date is not None else datetime.date.today()).isoformat()
        # `model_name` only routes to this codec; the tokenizer always comes from the pinned repo.
        self._tokenizer = load_tokenizer(
            _TOKENIZER_REPO, tokenizer=tokenizer, hf_token=hf_token, revision=_TOKENIZER_REVISION
        )

    def encode_messages(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        add_generation_prompt: bool = False,
    ) -> list[int]:
        """Render through the release template at the codec's date and thinking effort.

        Raises:
            ValueError: The last message is a reasoning-only assistant turn and no generation prompt follows (the
                template gives such a turn no stop token), or a tool result has neither a name nor a tool_call_id.
        """
        if not add_generation_prompt and messages and _is_reasoning_only(messages[-1]):
            raise ValueError(
                "Muse Spark's template ends a reasoning-only assistant turn on <|eom|>, so it has no stop token; "
                "merge it into the next assistant turn or drop it."
            )
        messages = _name_tool_results(messages)
        if self._strip_history:
            messages = strip_history_reasoning(messages)
        kwargs: dict[str, object] = {
            "tokenize": True,
            "add_generation_prompt": add_generation_prompt,
            "current_date": self._current_date,
        }
        if self._thinking_effort is not None:
            kwargs["reasoning_effort"] = self._thinking_effort
        if tools:
            kwargs["tools"] = [{"type": "function", "function": t} for t in tools]
        return self._tokenizer.apply_chat_template(messages_for_template(messages), **kwargs)["input_ids"]

    @staticmethod
    def parse_completion(
        *, content: str | None, reasoning: str | None, tools: list[ToolSpec] | None = None
    ) -> tuple[Message, bool]:
        """Parse an endpoint completion whose reasoning the server already split off.

        ``reasoning`` is kept as reasoning and only ``content`` is parsed. The server trims the final ``<|eot|>`` as
        the stop token, which closes the last tool channel too.
        """
        msg, ok = _parse_turn(content or "", tools, endpoint=True)
        if joined := "\n".join(filter(None, (reasoning, msg.get("reasoning_content")))):
            msg["reasoning_content"] = joined
        return msg, ok

    def parse_response(
        self, response_tokens: list[int], *, tools: list[ToolSpec] | None = None
    ) -> tuple[Message, bool]:
        return _parse_turn(str(self._tokenizer.decode(response_tokens, skip_special_tokens=False)), tools)

    def turn_terminator(self, next_role: Role | None) -> list[int]:
        return []

    def encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=False))

    def decode(self, tokens: list[int]) -> str:
        return str(self._tokenizer.decode(tokens, skip_special_tokens=True))


def _is_reasoning_only(msg: Message) -> bool:
    return (
        msg["role"] == "assistant"
        and bool(msg.get("reasoning_content") or msg.get("reasoning"))
        and not (msg.get("content") or msg.get("tool_calls"))
    )


def _name_tool_results(messages: list[Message]) -> list[Message]:
    """Name each unnamed tool result after its own call.

    Parsed call ids restart at ``call_1`` every turn, and the template names a result by the last call in the whole
    conversation with its id. A result whose id matches no call keeps the template's reading, the id as its name.

    Raises:
        ValueError: A tool result has neither a name nor an id, which the template cannot render.
    """
    names: dict[str, str] = {}
    named: list[Message] = []
    for index, msg in enumerate(messages):
        if msg["role"] == "assistant":
            calls = msg.get("tool_calls") or []  # `datasets` rows carry `tool_calls: None`
            names.update((call["id"], call["function"]["name"]) for call in calls if call.get("id"))
        elif msg["role"] == "tool" and not msg.get("name"):
            if not msg.get("tool_call_id"):
                raise ValueError(f"tool message {index} has neither a name nor a tool_call_id")
            if msg["tool_call_id"] in names:
                msg = {**msg, "name": names[msg["tool_call_id"]]}
        named.append(msg)
    return named


def _parse_turn(text: str, tools: list[ToolSpec] | None = None, *, endpoint: bool = False) -> tuple[Message, bool]:
    """Walk one assistant turn's channels into a Message.

    ``to=self`` is reasoning, ``to=user``/``to=commentary`` content, any other recipient a tool call; text outside
    any header is content. The verdict follows the module docstring, with ``endpoint`` granting its exemptions. A
    turn that fails is still read, and a turn cut inside reasoning or prose keeps what it produced.
    """
    param_types = param_types_by_tool(tools)
    reasoning: list[str] = []
    content: list[str] = []
    calls: list[ToolCall] = []
    ok = True
    pos = 0
    while pos < len(text):
        terminator = _TERMINATOR_RE.search(text, pos)
        stop = terminator.start() if terminator is not None else len(text)
        message_at = text.find(_MESSAGE, pos, stop)
        if message_at < 0:
            chunk = text[pos:stop]
            if chunk.lstrip().startswith(_START) or (
                terminator is None and _CUT_HEADER_RE.fullmatch(chunk) is not None
            ):
                ok = False
                break
            if chunk.strip():
                content.append(chunk)
                ok = ok and endpoint
        else:
            ok = ok and _HEADER_RE.fullmatch(text, pos, message_at) is not None
            role_match = _ROLE_RE.match(text, pos, message_at)
            if role_match is not None and role_match.group(1) != "assistant":
                break  # another role's message: the turn is over
            recipient_match = _RECIPIENT_RE.search(text, pos, message_at)
            recipient = recipient_match.group("recipient") if recipient_match is not None else "user"
            body = text[message_at + len(_MESSAGE) : stop]
            # The message ran into the next one without its `<|eom|>`.
            ok = ok and _START not in body and _MESSAGE not in body
            if recipient == "self":
                reasoning.append(body)
            elif recipient in ("user", "commentary"):
                content.append(body)
            else:
                channel_calls, channel_ok = _parse_calls(body, param_types)
                calls.extend(channel_calls)
                # `<|eot|>` is the stop token, which endpoints trim; a sampled tool channel always keeps it.
                ok = ok and channel_ok and (terminator is not None or endpoint)
        if terminator is None or terminator.group() != _EOM:
            break
        pos = terminator.end()
    # `<|eom|>` always continues the turn, so text ending on it after a call was cut before its remaining calls.
    if calls and text.rstrip().endswith(_EOM):
        ok = False

    msg = Message(role="assistant")
    if content_text := "".join(content):
        msg["content"] = content_text
    if reasoning_text := "\n".join(reasoning):
        msg["reasoning_content"] = reasoning_text
    if calls:
        msg["tool_calls"] = with_turn_call_ids(calls)
    return msg, ok


def _parse_calls(body: str, param_types: dict[str, dict[str, str]]) -> tuple[list[ToolCall], bool]:
    """Parse the ``<atem:invoke>`` calls of one tool channel; ids are assigned by the caller.

    The calls are the serving parsers' matching: an invoke or parameter with no name is skipped, and one left
    unclosed runs on to the next closing tag of its kind, or is dropped when none follows. The flag is whether the
    body is exactly one ``<atem:function_calls>`` block of named invokes holding named parameters and nothing else.
    """
    block = _BLOCK_RE.fullmatch(body)
    invokes = block.group("invokes") if block is not None else body
    ok = block is not None and not _INVOKE_RE.sub("", invokes).strip()
    calls: list[ToolCall] = []
    for invoke in _INVOKE_RE.finditer(invokes):
        name = _unescape(invoke.group("name"))
        declared = param_types.get(name, {})
        arguments: dict[str, object] = {}
        for param in _PARAM_RE.finditer(invoke.group("body")):
            key = _unescape(param.group("key"))
            arguments[key] = _decode_value(param.group("value"), declared.get(key))
        ok = ok and not _PARAM_RE.sub("", invoke.group("body")).strip()
        calls.append(ToolCall(id=None, function=ToolCallFunction(name=name, arguments=json.dumps(arguments))))
    return calls, ok and bool(calls)


def _decode_value(raw: str, declared_type: str | None) -> object:
    """Decode a declared (nullable) string as ``coerce_by_schema`` reads it, anything else as an undeclared value.

    Serving reads every value JSON-or-raw, ignoring the schema, so it would read a string-typed ``1068`` as an int;
    the env gets the declared string instead, as with the other untyped formats. Other declared types are not
    applied: ``coerce_by_schema`` would read a boolean-typed ``1`` or ``True`` as ``true``, which serving does not
    and which re-renders differently.
    """
    return coerce_by_schema(raw, declared_type if declared_type in ("string", "string|null") else None)


def _unescape(attribute: str) -> str:
    for entity, char in _ENTITIES:
        attribute = attribute.replace(entity, char)
    return attribute
