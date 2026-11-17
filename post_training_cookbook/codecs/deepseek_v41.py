"""DeepSeek-V4.1 Flash codec backed by the release encoder (see ``vendor/SOURCES.md``).

Thinking is on by default, at effort ``high`` (75). ``thinking_effort`` also accepts
``low`` (50), ``max`` (100), or a string integer from 1 to 100. Declared tools keep
reasoning throughout a tool loop, so sampled turns remain prefix-stable. Without
tools the release drops reasoning before the last user message unless preserved.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from transformers import PreTrainedTokenizerBase

from post_training_cookbook.codecs.base import load_tokenizer, resolve_enable_thinking, strip_history_reasoning
from post_training_cookbook.codecs.schema_coercion import coerce_by_schema, param_types_by_tool, parse_json
from post_training_cookbook.codecs.vendor import deepseek_v41_encoding as encoding
from post_training_cookbook.types import Message, Role, ToolCall, ToolCallFunction, ToolSpec

DEEPSEEK_V41_REPO = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
DEEPSEEK_V41_REVISION = "08c8657f02291e11b6f6b63e0e16014f8e748518"
ENCODER_SHA256 = "13231c592e958fb39e2af6001dae69d90e7a36b265702b9b390236fb387599e2"

_MARKER_IDS = {
    encoding.bos_token: 0,
    encoding.eos_token: 1,
    encoding.SYSTEM_SP_TOKEN: 128799,
    encoding.USER_SP_TOKEN: 128803,
    encoding.ASSISTANT_SP_TOKEN: 128804,
    encoding.thinking_start_token: 128821,
    encoding.thinking_end_token: 128822,
    encoding.dsml_token: 128825,
}
_DSML = encoding.dsml_token
_THINK_OPEN = encoding.thinking_start_token
_THINK_CLOSE = encoding.thinking_end_token
_CALLS_OPEN = f"<{_DSML}{encoding.tool_calls_block_name}>"
_CALLS_CLOSE = f"</{_DSML}{encoding.tool_calls_block_name}>"
_CALLS_PREFIXES = tuple(_CALLS_OPEN[:i] for i in range(1, len(_CALLS_OPEN)))
_INVOKE = f"{_DSML}{encoding.tool_call_tag_name}"
_PARAMETER = f"{_DSML}{encoding.tool_parameter_tag_name}"
# Match the serving parser's spacing and zero-argument self-closing calls.
_INVOKE_RE = re.compile(rf'<{_INVOKE}\s+name="(?P<name>[^"]*)"\s*(?:/>|>(?P<body>.*?)</{_INVOKE}>)', re.DOTALL)
_PARAMETER_RE = re.compile(
    rf'<{_PARAMETER}\s+name="(?P<key>[^"]*)"\s+string="(?P<string>[^"]*)"\s*>(?P<value>.*?)</{_PARAMETER}>',
    re.DOTALL,
)
_SCHEMA_TYPES: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "string|null": (str, type(None)),
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}


def _parse_thinking_effort(raw: str | None) -> str | int | None:
    if raw is None or raw in encoding.REASONING_EFFORT_MAPPINGS:
        return raw
    if isinstance(raw, str) and raw.isascii() and raw.isdigit() and 1 <= int(raw) <= 100:
        return int(raw)
    raise ValueError(f"DeepSeek-V4.1 thinking_effort must be low/high/max or a string integer 1-100; got {raw!r}")


class DeepSeekV41Codec:
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
        self._thinking = (
            resolve_enable_thinking(enable_thinking=enable_thinking, preserve_thinking=preserve_thinking) is not False
        )
        self._preserve_thinking = preserve_thinking
        self._reasoning_effort = _parse_thinking_effort(thinking_effort)
        local = model_name is not None and Path(model_name).is_dir()
        self._tokenizer = load_tokenizer(
            model_name if local else DEEPSEEK_V41_REPO,
            tokenizer=tokenizer,
            hf_token=hf_token,
            revision=None if local else DEEPSEEK_V41_REVISION,
        )
        for marker, token_id in _MARKER_IDS.items():
            if self.encode(marker) != [token_id]:
                raise ValueError(f"not a DeepSeek-V4.1 tokenizer: {marker} does not encode to [{token_id}]")

    @property
    def reasoning_budget(self) -> int:
        """Numeric reasoning effort written into thinking prompts (75 by default)."""
        effort = self._reasoning_effort
        if effort is None:
            effort = encoding.DEFAULT_REASONING_EFFORT
        return effort if isinstance(effort, int) else encoding.REASONING_EFFORT_MAPPINGS[effort]

    def render(self, messages: list[Message], *, tools: list[ToolSpec] | None = None) -> str:
        """Render through the release encoder, including its automatic generation header."""
        if self._preserve_thinking is False:
            messages = strip_history_reasoning(messages)
        return encoding.encode_messages(
            _encoder_messages(messages, tools),
            thinking_mode="thinking" if self._thinking else "chat",
            drop_thinking=self._preserve_thinking is not True,
            reasoning_effort=self._reasoning_effort,
        )

    def encode_messages(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        add_generation_prompt: bool = False,
    ) -> list[int]:
        text = self.render(messages, tools=tools)
        if not add_generation_prompt and messages and messages[-1]["role"] in ("user", "tool"):
            header = encoding.ASSISTANT_SP_TOKEN + (_THINK_OPEN if self._thinking else _THINK_CLOSE)
            text = text.removesuffix(header)
        return self.encode(text)

    @staticmethod
    def parse_text(text: str, *, tools: list[ToolSpec] | None = None) -> tuple[Message, bool]:
        """Parse text with an optional explicit thinking fence."""
        if _THINK_CLOSE not in text:
            return _parse_turn(text, tools, thinking=False)
        fenced = text.lstrip()
        if fenced.startswith(_THINK_OPEN):
            text = fenced[len(_THINK_OPEN) :]
        return _parse_turn(text, tools, thinking=True)

    @staticmethod
    def parse_completion(
        *, content: str | None, reasoning: str | None, tools: list[ToolSpec] | None = None
    ) -> tuple[Message, bool]:
        """Parse only the content channel; calls in endpoint-split reasoning never execute."""
        msg, ok = _parse_turn(content or "", tools, thinking=False)
        if reasoning:
            msg["reasoning_content"] = reasoning
        return msg, ok

    def parse_response(
        self, response_tokens: list[int], *, tools: list[ToolSpec] | None = None
    ) -> tuple[Message, bool]:
        return _parse_turn(self.decode(response_tokens), tools, thinking=self._thinking)

    def turn_terminator(self, next_role: Role | None) -> list[int]:
        return []

    def encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=False))

    def decode(self, tokens: list[int]) -> str:
        return str(self._tokenizer.decode(tokens, skip_special_tokens=True))


def _encoder_messages(messages: list[Message], tools: list[ToolSpec] | None) -> list[dict]:
    out: list[dict] = [dict(msg) for msg in messages]
    if tools:
        declared = [{"type": "function", "function": dict(tool)} for tool in tools]
        if out and out[0].get("role") == "system":
            out[0]["tools"] = declared
        else:
            out.insert(0, {"role": "system", "content": "", "tools": declared})
    return out


def _parse_turn(text: str, tools: list[ToolSpec] | None, *, thinking: bool) -> tuple[Message, bool]:
    reasoning: str | None = None
    rest = text
    if thinking:
        head, closed, tail = text.partition(_THINK_CLOSE)
        if closed:
            reasoning, rest = head, tail
        else:
            return Message(role="assistant", reasoning_content=text), _DSML not in text

    calls: list[ToolCall] = []
    start = rest.find(_CALLS_OPEN)
    if start < 0:
        content = rest
        ok = _DSML not in content and not content.endswith(_CALLS_PREFIXES)
    else:
        # Preserve text whitespace, removing only the encoder's blank-line separator.
        content = rest[:start].removesuffix("\n\n")
        end = rest.find(_CALLS_CLOSE, start)
        if end < 0:
            ok = False
        else:
            calls, ok = _parse_calls(rest[start + len(_CALLS_OPEN) : end], param_types_by_tool(tools))
            ok = ok and not rest[end + len(_CALLS_CLOSE) :].strip()
        ok = ok and _DSML not in content

    msg = Message(role="assistant")
    if content:
        msg["content"] = content
    if reasoning:
        msg["reasoning_content"] = reasoning
    if calls:
        msg["tool_calls"] = calls
    return msg, ok


def _parse_calls(block: str, param_types: dict[str, dict[str, str]]) -> tuple[list[ToolCall], bool]:
    calls: list[ToolCall] = []
    ok = True
    pos = 0
    for invoke in _INVOKE_RE.finditer(block):
        ok = ok and not block[pos : invoke.start()].strip()
        pos = invoke.end()
        name = invoke.group("name").strip()
        arguments, arguments_ok = _parse_arguments(invoke.group("body") or "", param_types.get(name, {}))
        if not name or name.count("::") > 1 or not arguments_ok:
            ok = False
            continue
        calls.append(
            ToolCall(
                id=f"call_{len(calls) + 1}", function=ToolCallFunction(name=name, arguments=json.dumps(arguments))
            )
        )
    return calls, ok and not block[pos:].strip() and bool(calls)


def _parse_arguments(body: str, declared: dict[str, str]) -> tuple[dict[str, object], bool]:
    stripped = body.strip()
    if stripped.startswith("{"):
        try:
            arguments = parse_json(stripped)
        except ValueError:
            return {}, False
        return (arguments, True) if isinstance(arguments, dict) else ({}, False)

    arguments: dict[str, object] = {}
    ok = True
    leftover: list[str] = []
    pos = 0
    for param in _PARAMETER_RE.finditer(body):
        leftover.append(body[pos : param.start()])
        pos = param.end()
        key = param.group("key")
        if not key or key in arguments:
            ok = False
            continue
        arguments[key] = _typed(
            param.group("value"), wire_string=param.group("string") == "true", declared=declared.get(key)
        )
    leftover.append(body[pos:])
    rest = "".join(leftover)
    if _DSML in rest or (not arguments and rest.strip()):
        ok = False
    return arguments, ok


def _typed(raw: str, *, wire_string: bool, declared: str | None) -> object:
    """Keep wire types unless schema coercion can repair a mismatch."""
    if wire_string:
        value: object = raw
    else:
        try:
            value = parse_json(raw.strip())
        except ValueError:
            value = raw.strip()
    if declared is None or _has_type(value, declared):
        return value
    coerced = coerce_by_schema(raw, declared)
    return coerced if _has_type(coerced, declared) else value


def _has_type(value: object, declared: str) -> bool:
    expected = _SCHEMA_TYPES.get(declared)
    if expected is None:
        return True
    # bool is an int subclass, but is not a JSON number.
    return isinstance(value, expected) and not (isinstance(value, bool) and declared != "boolean")
