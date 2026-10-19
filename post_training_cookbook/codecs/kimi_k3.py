"""Kimi-K3 codec: the XTML channel format rendered by the checkpoint's own encoder.

Kimi-K3 ships no Jinja chat template: ``moonshotai/Kimi-K3``'s ``tokenization_kimi.py`` renders in
Python (``encoding_k3.build_chat_segments``), so the codec loads that tokenizer with
``trust_remote_code`` at a pinned revision and lets it render. Every message is an XTML element::

    <|open|>message role="assistant"<|sep|>
      <|open|>think<|sep|>...reasoning...<|close|>think<|sep|>
      <|open|>response<|sep|>...visible answer...<|close|>response<|sep|>
      <|open|>tools<|sep|>
        <|open|>call tool="NAME" index="1"<|sep|>
          <|open|>argument key="K" type="string|number|boolean|object|array|null"<|sep|>VALUE<|close|>argument<|sep|>
        <|close|>call<|sep|>
      <|close|>tools<|sep|>
    <|close|>message<|sep|><|end_of_msg|>

The generation prompt ends inside the open ``think`` channel (thinking on, the model default) or the
open ``response`` channel (thinking off), so a sampled turn starts mid-channel. ``<|end_of_msg|>``
is the generation stop and sits inside every rendered turn, so ``turn_terminator`` is empty.

With thinking on, the encoder renders the reasoning of every assistant turn it is given, so
re-rendering history is prefix-stable and ``preserve_thinking=True`` changes nothing; ``False`` drops
reasoning from turns before the last user message first, as a client that does not echo it back
would. With thinking off the encoder drops every ``think`` channel itself. A well-formed
parsed turn re-renders to the text it was sampled as: content and reasoning come back verbatim, and
each tool-call argument keeps its sampled literal, which the encoder renders back unchanged. Shapes
the encoder normalizes re-render differently, and ``split_turns`` starts a new sequence there — e.g.
a ``think`` that never closes, a non-string argument that is not valid JSON, padded non-string
literals, a dropped malformed call, or duplicate argument keys.

Arguments are typed by the model's own ``type`` attribute, not the tool schema, because that is what
the serving parsers (smg, sglang, vLLM) execute; ``tools`` is accepted for the protocol and ignored.

Only ``<|open|>``/``<|close|>``/``<|sep|>``/``<|end_of_msg|>`` are control tokens, and the encoder
writes message text with control tokens disallowed. Decoded text cannot tell a structural marker from
one the model spelled out, though, so parsing follows the channel grammar: inside ``think`` only its
own close or the end of the message is a transition, so anything written in an open ``think`` is
reasoning; inside ``response`` a ``tools`` open is one too.
"""

from __future__ import annotations

import json
import re

from transformers import PreTrainedTokenizerBase

from post_training_cookbook.codecs.base import (
    check_thinking_effort,
    load_tokenizer,
    resolve_enable_thinking,
    strip_history_reasoning,
    with_turn_call_ids,
)
from post_training_cookbook.codecs.schema_coercion import parse_json
from post_training_cookbook.types import Message, Role, ToolCall, ToolCallFunction, ToolSpec

_OPEN = "<|open|>"
_CLOSE = "<|close|>"
_SEP = "<|sep|>"
_END_OF_MSG = "<|end_of_msg|>"

# The tokenizer is checkpoint code, so the repo and revision that run are pinned together (bumping
# them is deliberate). Every Kimi-K3 catalog variant (e.g. "Kimi-K3-262k") shares this tokenizer.
_TOKENIZER_REPO = "moonshotai/Kimi-K3"
_TOKENIZER_REVISION = "f831ab66814297da540d832a5235f8e904f29d06"
_THINKING_EFFORTS = ("low", "high", "max")

_TOOLS_CLOSE = f"{_CLOSE}tools{_SEP}"
# Where a text channel ends: its own close is consumed, while a lookahead leaves the end of the message
# (and, for ``response``, a ``tools`` open) for the walker. A ``think`` ends nowhere else: whatever the
# model writes inside an open think, markers included, is reasoning.
_TURN_END = r"<\|close\|>message<\|sep\|>|<\|end_of_msg\|>"
_CHANNEL_END_RE = {
    "think": re.compile(rf"<\|close\|>think<\|sep\|>|(?={_TURN_END})"),
    "response": re.compile(rf"<\|close\|>response<\|sep\|>|(?=<\|open\|>tools<\|sep\|>|{_TURN_END})"),
}

_ATTRS = r"(?P<attrs>(?: \w+=\"[^\"]*\")*)"
_MARKER_RE = re.compile(
    rf"<\|open\|>(?P<open>\w+){_ATTRS}<\|sep\|>|<\|close\|>(?P<close>\w+)<\|sep\|>|<\|end_of_msg\|>"
)
_ATTR_RE = re.compile(r'(\w+)="([^"]*)"')
_CALL_RE = re.compile(rf"<\|open\|>call{_ATTRS}<\|sep\|>(?P<body>.*?)<\|close\|>call<\|sep\|>", re.DOTALL)
_ARG_RE = re.compile(rf"<\|open\|>argument{_ATTRS}<\|sep\|>(?P<value>.*?)<\|close\|>argument<\|sep\|>", re.DOTALL)
_JSON_RE = re.compile(rf"<\|open\|>json{_ATTRS}<\|sep\|>(?P<value>.*?)<\|close\|>json<\|sep\|>", re.DOTALL)


class KimiK3Codec:
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
        self._thinking_effort = check_thinking_effort(thinking_effort, supported=_THINKING_EFFORTS, family="Kimi-K3")
        self._strip_history = preserve_thinking is False
        enable = resolve_enable_thinking(enable_thinking=enable_thinking, preserve_thinking=preserve_thinking)
        self._thinking = enable is not False
        # `model_name` only routes to this codec; the tokenizer always comes from the pinned repo.
        self._tokenizer = load_tokenizer(
            _TOKENIZER_REPO,
            tokenizer=tokenizer,
            hf_token=hf_token,
            revision=_TOKENIZER_REVISION,
            trust_remote_code=True,
        )

    def encode_messages(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        add_generation_prompt: bool = False,
    ) -> list[int]:
        # Canonical messages go in as-is: the encoder parses JSON-string arguments itself, keeping
        # each value's original literal, and renders non-JSON arguments in its raw `json` element.
        kwargs: dict[str, object] = {
            "tokenize": True,
            "add_generation_prompt": add_generation_prompt,
            "thinking": self._thinking,
        }
        if self._thinking_effort is not None:
            kwargs["thinking_effort"] = self._thinking_effort
        if tools:
            kwargs["tools"] = [{"type": "function", "function": t} for t in tools]
        if self._strip_history:
            messages = strip_history_reasoning(messages)
        return list(self._tokenizer.apply_chat_template(messages, **kwargs))

    @staticmethod
    def parse_completion(
        *, content: str | None, reasoning: str | None, tools: list[ToolSpec] | None = None
    ) -> tuple[Message, bool]:
        """Parse an endpoint completion whose reasoning the server already split off.

        Serving never runs anything written in reasoning, so it is kept as reasoning and only
        ``content`` is parsed: from a channel boundary when it opens with a marker, else mid-response.
        """
        content = content or ""
        msg, ok = _parse_turn(content, channel=None if content.startswith((_OPEN, _CLOSE)) else "response")
        if reasoning:
            msg["reasoning_content"] = reasoning + msg.get("reasoning_content", "")
        return msg, ok

    def parse_response(
        self, response_tokens: list[int], *, tools: list[ToolSpec] | None = None
    ) -> tuple[Message, bool]:
        channel = "think" if self._thinking else "response"
        return _parse_turn(self.decode(response_tokens), channel=channel)

    def turn_terminator(self, next_role: Role | None) -> list[int]:
        return []

    def encode(self, text: str) -> list[int]:
        # Like the other families' tokenizers, marker strings map to their control ids. No kwargs:
        # any kwarg sends the checkpoint tokenizer down the generic slow path with a warning.
        return list(self._tokenizer.encode(text))

    def decode(self, tokens: list[int]) -> str:
        # Verbatim: the parser needs the XTML control tokens, and rendered and sampled K3 turns carry
        # no BOS/EOS to skip. Any kwarg (e.g. skip_special_tokens) would also take the slow path.
        return str(self._tokenizer.decode(tokens))


def _parse_turn(text: str, *, channel: str | None) -> tuple[Message, bool]:
    """Walk one assistant turn's channels into a Message.

    ``channel`` is the channel the text starts inside (``think`` or ``response``), or ``None`` at a
    channel boundary. A turn cut off inside a text channel keeps what it produced and parses ok, unless the cut lands
    inside a multi-token marker (even the channel's own close): that half-marker is a truncation. A
    ``tools`` channel or call that never closes keeps the complete calls and fails; a complete but
    malformed call is dropped, as serving drops it, and the turn still parses ok.
    """
    reasoning: list[str] = []
    content: list[str] = []
    calls: list[ToolCall] = []
    ok = True
    pos = 0
    while pos < len(text):
        if channel in ("think", "response"):
            end = _CHANNEL_END_RE[channel].search(text, pos)
            (reasoning if channel == "think" else content).append(text[pos : end.start() if end else None])
            pos, channel = (end.end() if end else len(text)), None
        elif channel == "tools":
            close = text.find(_TOOLS_CLOSE, pos)
            section_calls, section_ok = _parse_calls(text[pos:] if close < 0 else text[pos:close])
            calls.extend(section_calls)
            ok = ok and section_ok
            if close < 0:
                break
            pos, channel = close + len(_TOOLS_CLOSE), None
        else:
            marker = _MARKER_RE.search(text, pos)
            if (
                marker is None
                or marker.group(0) == _END_OF_MSG
                or "message" in (marker.group("open"), marker.group("close"))
            ):
                break
            pos = marker.end()
            # Text between channels, and any element other than a channel, is not part of the turn.
            if marker.group("open") in ("think", "response", "tools"):
                channel = marker.group("open")

    # A turn cut right after its tools open, or inside a multi-token marker, is truncated too.
    last_marker = max(text.rfind(_OPEN), text.rfind(_CLOSE))
    if channel == "tools" or (last_marker >= 0 and _SEP not in text[last_marker:]):
        ok = False

    msg = Message(role="assistant")
    if content_text := "".join(content):
        msg["content"] = content_text
    if reasoning_text := "".join(reasoning):
        msg["reasoning_content"] = reasoning_text
    if calls:
        # The encoder re-derives each call's index from its position, so ids follow suit.
        msg["tool_calls"] = with_turn_call_ids(calls)
    return msg, ok


def _parse_calls(section: str) -> tuple[list[ToolCall], bool]:
    """Parse the body of one ``tools`` channel; ids are assigned by the caller.

    As the serving parsers do, a complete call with no tool name or malformed arguments is dropped
    without failing the turn; only a call that never closed (a truncation) fails it.
    """
    calls: list[ToolCall] = []
    for match in _CALL_RE.finditer(section):
        name = _parse_attrs(match.group("attrs")).get("tool", "")
        arguments = _parse_arguments(match.group("body"))
        if name and arguments is not None:
            calls.append(ToolCall(id=None, function=ToolCallFunction(name=name, arguments=arguments)))
    # A call opened after the last call close never closed: the turn was cut inside it.
    return calls, section.rfind(f"{_OPEN}call") <= section.rfind(f"{_CLOSE}call{_SEP}")


def _parse_arguments(body: str) -> str | None:
    """A call body's arguments as a JSON object string, or ``None`` when they are malformed.

    The encoder writes typed ``argument`` elements, or one raw ``json`` element when the history's
    argument string was not a JSON object.
    """
    json_block = _JSON_RE.search(body)
    if json_block is not None:
        raw = json_block.group("value")
        try:
            parsed = parse_json(raw)
        except ValueError:
            return None
        return raw if isinstance(parsed, dict) else None
    members: dict[str, str] = {}
    for arg in _ARG_RE.finditer(body):
        attrs = _parse_attrs(arg.group("attrs"))
        members[attrs.get("key", "")] = _argument_json(arg.group("value"), attrs.get("type"))
    return "{" + ", ".join(f"{json.dumps(key)}: {text}" for key, text in members.items()) + "}"


def _argument_json(value: str, wire_type: str | None) -> str:
    """One argument value as JSON text, typed by the model's own ``type`` attribute.

    The serving parsers type by the wire attribute alone and ignore the tool schema, so training
    rewards the call that would actually run. A non-string value that is valid JSON keeps its sampled
    literal (``2.50``, ``1e3``, ``{"k":1}``), which the encoder renders back unchanged; one that is not
    (``True``, ``NaN``, ``1_000``) falls back to the raw text as a string.
    """
    if wire_type not in (None, "string"):
        try:
            parse_json(value)
        except ValueError:
            pass
        else:
            return value
    return json.dumps(value)


def _parse_attrs(attrs: str) -> dict[str, str]:
    # The encoder escapes exactly these two in attribute values.
    return {key: value.replace("&quot;", '"').replace("&amp;", "&") for key, value in _ATTR_RE.findall(attrs)}
