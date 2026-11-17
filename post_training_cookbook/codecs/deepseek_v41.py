"""DeepSeek-V4.1 Flash text encoding and DSML tool-call parsing.

Thinking is on by default, at effort ``high`` (75). ``thinking_effort`` also accepts
``low`` (50), ``max`` (100), or a string integer from 1 to 100. Declared tools keep
reasoning throughout a tool loop, so sampled turns remain prefix-stable. Without
tools the release drops reasoning before the last user message unless preserved.
"""

# Text rendering adapted from encoding/encoding.py in DEEPSEEK_V41_REPO at DEEPSEEK_V41_REVISION.
# MIT License
#
# Copyright (c) 2023 DeepSeek
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import json
import re
from pathlib import Path

from transformers import PreTrainedTokenizerBase

from post_training_cookbook.codecs.base import load_tokenizer, resolve_enable_thinking, strip_history_reasoning
from post_training_cookbook.codecs.schema_coercion import coerce_by_schema, param_types_by_tool, parse_json
from post_training_cookbook.types import Message, Role, ToolCall, ToolCallFunction, ToolSpec

DEEPSEEK_V41_REPO = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
DEEPSEEK_V41_REVISION = "08c8657f02291e11b6f6b63e0e16014f8e748518"

_BOS = "<｜begin▁of▁sentence｜>"
_EOS = "<｜end▁of▁sentence｜>"
_SYSTEM = "<｜System｜>"
_USER = "<｜User｜>"
_ASSISTANT = "<｜Assistant｜>"
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_DSML = "｜DSML｜"
_REASONING_EFFORTS = {"low": 50, "high": 75, "max": 100}

_MARKER_IDS = {
    _BOS: 0,
    _EOS: 1,
    _SYSTEM: 128799,
    _USER: 128803,
    _ASSISTANT: 128804,
    _THINK_OPEN: 128821,
    _THINK_CLOSE: 128822,
    _DSML: 128825,
}
_CALLS_OPEN = f"<{_DSML} calls>"
_CALLS_CLOSE = f"</{_DSML} calls>"
_CALLS_PREFIXES = tuple(_CALLS_OPEN[:i] for i in range(1, len(_CALLS_OPEN)))
_INVOKE = f"{_DSML} invoke"
_PARAMETER = f"{_DSML} parameter"
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


_TOOLS_TEMPLATE = (
    "## Tools\n"
    "\n"
    "You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "
    '"<{dsml_token}{tc_block_name}>" block like the following:\n'
    "\n"
    "<{dsml_token}{tc_block_name}>\n"
    '<{dsml_token}{tool_call_tag_name} name="$TOOL_NAME">\n'
    '<{dsml_token}{tool_parameter_tag_name} name="$PARAMETER_NAME" '
    'string="true|false">$PARAMETER_VALUE</{dsml_token}{tool_parameter_tag_name}>\n'
    "...\n"
    "</{dsml_token}{tool_call_tag_name}>\n"
    '<{dsml_token}{tool_call_tag_name} name="$TOOL_NAME2">\n'
    "...\n"
    "</{dsml_token}{tool_call_tag_name}>\n"
    "</{dsml_token}{tc_block_name}>\n"
    "\n"
    'String parameters should be specified as is and set `string="true"`. For all other types (numbers, '
    'booleans, arrays, objects), pass the value in JSON format and set `string="false"`.\n'
    "\n"
    "If thinking_mode is enabled (triggered by {thinking_start_token}), you MUST output your complete "
    "reasoning inside {thinking_start_token}...{thinking_end_token} BEFORE any tool calls or final response.\n"
    "\n"
    "Otherwise, output directly after {thinking_end_token} with tool calls or final response.\n"
    "\n"
    "### Available Tool Schemas\n"
    "\n"
    "{tool_schemas}\n"
    "\n"
    "You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n"
)


def _parse_thinking_effort(raw: str | None) -> str | int | None:
    if raw is None or raw in _REASONING_EFFORTS:
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
            effort = "high"
        return effort if isinstance(effort, int) else _REASONING_EFFORTS[effort]

    def render(self, messages: list[Message], *, tools: list[ToolSpec] | None = None) -> str:
        """Render through the release encoder, including its automatic generation header."""
        if self._preserve_thinking is False:
            messages = strip_history_reasoning(messages)
        return _render_messages(
            _encoder_messages(messages, tools),
            thinking=self._thinking,
            preserve_thinking=self._preserve_thinking is True,
            budget=self.reasoning_budget,
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
            header = _ASSISTANT + (_THINK_OPEN if self._thinking else _THINK_CLOSE)
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


def _tool_name(name: str, namespace: str | dict | None = None) -> str:
    if isinstance(namespace, dict):
        namespace = namespace["name"]
    prefix, separator, bare = name.partition("::")
    if separator:
        if namespace not in (None, prefix):
            raise ValueError(f"Conflicting tool namespaces: {namespace} != {prefix}")
        namespace, name = prefix, bare
    if "::" in name or (namespace is not None and "::" in namespace):
        raise ValueError(f"Invalid tool name: {name}")
    return name if namespace is None else f"{namespace}::{name}"


def _render_tools(tools: list[dict]) -> str:
    functions = []
    for tool in tools:
        function = dict(tool["function"])
        if tool.get("namespace") is not None:
            function["namespace"] = tool["namespace"]
        function["name"] = _tool_name(function["name"], function.get("namespace"))
        namespace = function.pop("namespace", None)
        if isinstance(namespace, dict) and namespace.get("description"):
            function["description"] = namespace["description"] + "\n" + (function.get("description") or "")
        functions.append(json.dumps(function, ensure_ascii=False))
    return _TOOLS_TEMPLATE.format(
        tool_schemas="\n".join(functions),
        dsml_token=_DSML,
        tc_block_name=" calls",
        tool_call_tag_name=" invoke",
        tool_parameter_tag_name=" parameter",
        thinking_start_token=_THINK_OPEN,
        thinking_end_token=_THINK_CLOSE,
    )


def _render_call(call: ToolCall) -> str:
    function = call["function"]
    name = _tool_name(function["name"], call.get("namespace") or function.get("namespace"))
    arguments = function["arguments"]
    for _ in range(2):
        if not isinstance(arguments, str):
            break
        try:
            arguments = json.loads(arguments)
        except ValueError:
            break
    if not isinstance(arguments, dict):
        arguments = {"arguments": function["arguments"]}
    parameters = []
    for key, value in arguments.items():
        string = isinstance(value, str)
        text = value if string else json.dumps(value, ensure_ascii=False)
        parameters.append(f'<{_PARAMETER} name="{key}" string="{str(string).lower()}">{text}</{_PARAMETER}>')
    return f'<{_INVOKE} name="{name}">\n' + "\n".join(parameters) + f"\n</{_INVOKE}>"


def _merge_tool_results(messages: list[dict]) -> list[dict]:
    merged: list[dict] = []
    for msg in messages:
        role = msg["role"]
        if role == "tool":
            blocks = [
                {"type": "tool_result", "content": msg.get("content", ""), "tool_use_id": msg.get("tool_call_id")}
            ]
        elif role == "user":
            blocks = msg.get("content_blocks")
            if blocks is None:
                blocks = [{"type": "text", "text": msg.get("content", "")}]
        else:
            merged.append(msg)
            continue
        if merged and merged[-1]["role"] == "user":
            merged[-1]["content_blocks"].extend(blocks)
        else:
            merged.append({**msg, "role": "user", "content_blocks": list(blocks)})
    return merged


def _render_user(msg: dict, call_order: dict[str, int]) -> str:
    blocks = list(msg["content_blocks"])
    tool_results = [block for block in blocks if block["type"] == "tool_result"]
    tool_results.sort(key=lambda block: call_order.get(block.get("tool_use_id"), 0))
    ordered = iter(tool_results)
    parts = []
    for block in blocks:
        if block["type"] == "tool_result":
            block = next(ordered)
            parts.append(f"<tool_result>{block.get('content', '')}</tool_result>")
        elif block["type"] == "text":
            parts.append(block.get("text", ""))
        else:
            raise ValueError(f"Unsupported DeepSeek-V4.1 content block: {block['type']}")
    return _USER + "\n\n".join(parts)


def _render_messages(messages: list[dict], *, thinking: bool, preserve_thinking: bool, budget: int) -> str:
    messages = _merge_tool_results(messages)
    keep_reasoning = preserve_thinking or any(msg.get("tools") for msg in messages)
    last_user = max(
        (
            i
            for i, msg in enumerate(messages)
            if (msg["role"] == "system" and i > 0)
            or (msg["role"] == "user" and any(block["type"] == "text" for block in msg["content_blocks"]))
        ),
        default=-1,
    )
    parts = [_BOS]
    call_order: dict[str, int] = {}
    for i, msg in enumerate(messages):
        role = msg["role"]
        if i == 0:
            if thinking or role == "system":
                parts.append(_SYSTEM)
            if thinking:
                parts.append(
                    f"Reasoning Effort: {budget} "
                    "(range 1-100, the higher the value, the more thorough the reasoning)\n\n"
                )
        if role == "system":
            if i > 0:
                parts.append(_SYSTEM)
            parts.append(msg.get("content") or "")
            if msg.get("tools"):
                parts.append("\n\n" + _render_tools(msg["tools"]))
        elif role == "user":
            parts.append(_render_user(msg, call_order))
        elif role == "assistant":
            if thinking and (keep_reasoning or i > last_user):
                parts.append((msg.get("reasoning_content") or "") + _THINK_CLOSE)
            parts.append(msg.get("content") or "")
            if calls := msg.get("tool_calls"):
                parts.append(
                    "\n\n" + _CALLS_OPEN + "\n" + "\n".join(_render_call(call) for call in calls) + "\n" + _CALLS_CLOSE
                )
                call_order = {
                    call_id: j
                    for j, call in enumerate(calls)
                    if (call_id := call.get("id") or call["function"].get("id", ""))
                }
            parts.append(_EOS)
        else:
            raise ValueError(f"Unsupported DeepSeek-V4.1 message role: {role}")
        if (i == len(messages) - 1 or messages[i + 1]["role"] == "assistant") and (
            role == "user" or (role == "system" and i > 0)
        ):
            parts.append(
                _ASSISTANT + (_THINK_OPEN if thinking and (keep_reasoning or i >= last_user) else _THINK_CLOSE)
            )
    return "".join(parts)
