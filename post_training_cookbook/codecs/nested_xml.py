r"""Shared parsing for chat formats with nested-XML tool calls.

Qwen3.5 (and 3.6, 3.8) and Nemotron 3.5 Super VL both emit a ``<think>`` block followed by tool calls of
the form::

    <tool_call>
    <function=NAME>
    <parameter=KEY>
    VALUE
    </parameter>
    </function>
    </tool_call>

and close every assistant turn with ``<|im_end|>``. ``NestedXmlCodec`` holds the parsing and token-level
methods those codecs share; each subclass sets ``_tokenizer`` and implements ``encode_messages`` for its
own template flags, then calls ``_detect_gen_prompt_think``.
"""

from __future__ import annotations

import json
import re
from functools import partial

from transformers import PreTrainedTokenizerBase

from post_training_cookbook.codecs.response_blocks import (
    parse_after_open_think,
    parse_response_blocks,
    restore_response_text,
)
from post_training_cookbook.codecs.schema_coercion import coerce_by_schema, param_types_by_tool
from post_training_cookbook.types import Message, Role, ToolCall, ToolCallFunction, ToolSpec

_FUNCTION_NAME_RE = re.compile(r"<function=([^>]*)>", re.DOTALL)
_PARAMETER_RE = re.compile(r"<parameter=([^>]*)>(.*?)</parameter>", re.DOTALL)


class NestedXmlCodec:
    """Base for codecs whose tool calls use the nested ``<function=…><parameter=…>`` grammar."""

    _tokenizer: PreTrainedTokenizerBase
    _gen_prompt_opens_think: bool

    def encode_messages(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        add_generation_prompt: bool = False,
    ) -> list[int]:
        raise NotImplementedError

    def _detect_gen_prompt_think(self) -> None:
        # Whether a response starts inside `<think>` depends on the template and its flags, so ask the
        # rendered generation prompt itself.
        gen_prompt = self.decode(self.encode_messages([{"role": "user", "content": ""}], add_generation_prompt=True))
        self._gen_prompt_opens_think = gen_prompt.rstrip().endswith("<think>")

    @staticmethod
    def parse_text(text: str, *, tools: list[ToolSpec] | None = None) -> tuple[Message, bool]:
        param_types = param_types_by_tool(tools)
        return parse_response_blocks(text, lambda raw: _parse_tool_body_xml(raw, param_types))

    @classmethod
    def parse_completion(
        cls, *, content: str | None, reasoning: str | None, tools: list[ToolSpec] | None = None
    ) -> tuple[Message, bool]:
        return cls.parse_text(restore_response_text(content=content, reasoning=reasoning), tools=tools)

    def parse_response(
        self, response_tokens: list[int], *, tools: list[ToolSpec] | None = None
    ) -> tuple[Message, bool]:
        parse_text = partial(self.parse_text, tools=tools)
        text = self.decode(response_tokens)
        return parse_after_open_think(text, parse_text) if self._gen_prompt_opens_think else parse_text(text)

    def turn_terminator(self, next_role: Role | None) -> list[int]:
        # The template already closes every assistant turn with `<|im_end|>`.
        return []

    def encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=False))

    def decode(self, tokens: list[int]) -> str:
        return str(self._tokenizer.decode(tokens, skip_special_tokens=True))


def _parse_tool_body_xml(raw: str, param_types: dict[str, dict[str, str]]) -> tuple[ToolCall | None, bool]:
    """Parse the nested-XML body of a <tool_call> block.

    Each `<parameter=KEY>VALUE</parameter>` value is rendered **unquoted** and
    newline-framed, so the wire form carries no type information. We strip the
    template's framing newline, then coerce each value against the param's declared
    JSON-schema type (``param_types[name][key]``) — passing the schema is what keeps
    a string-typed numeric arg a string rather than an int (the mismatch that fails
    downstream schema validation, i.e. the serverless-vs-generator reward gap).
    Coercion never raises.
    """
    name_match = _FUNCTION_NAME_RE.search(raw)
    if name_match is None:
        return None, False
    name = name_match.group(1).strip()
    if not name:
        return None, False

    types_for_tool = param_types.get(name, {})
    arguments: dict[str, object] = {}
    for param_match in _PARAMETER_RE.finditer(raw):
        key = param_match.group(1).strip()
        if not key:
            return None, False
        value = _strip_framing_newline(param_match.group(2))
        arguments[key] = coerce_by_schema(value, types_for_tool.get(key))

    return ToolCall(
        id=None,
        function=ToolCallFunction(name=name, arguments=json.dumps(arguments)),
    ), True


def _strip_framing_newline(value: str) -> str:
    r"""Peel the template's `<parameter=x>\n … \n</parameter>` framing.

    Removes exactly one leading and one trailing newline, so interior and intentional
    edge whitespace (e.g. code indentation) in string-typed values survives.
    """
    if value.startswith("\n"):
        value = value[1:]
    if value.endswith("\n"):
        value = value[:-1]
    return value
