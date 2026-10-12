from __future__ import annotations

import json
import re
import warnings
from functools import partial
from importlib.resources import files

from jinja2.meta import find_undeclared_variables
from transformers import PreTrainedTokenizerBase
from transformers.utils.chat_template_utils import _compile_jinja_template

from post_training_cookbook.codecs.base import (
    PreserveThinkingForkedWarning,
    check_thinking_effort,
    load_tokenizer,
    messages_for_template,
    resolve_enable_thinking,
)
from post_training_cookbook.codecs.response_blocks import (
    parse_after_open_think,
    parse_response_blocks,
    restore_response_text,
)
from post_training_cookbook.codecs.schema_coercion import coerce_by_schema, param_types_by_tool
from post_training_cookbook.types import Message, Role, ToolCall, ToolCallFunction, ToolSpec

_FUNCTION_NAME_RE = re.compile(r"<function=([^>]*)>", re.DOTALL)
_PARAMETER_RE = re.compile(r"<parameter=([^>]*)>(.*?)</parameter>", re.DOTALL)


class Qwen35Codec:
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
        check_thinking_effort(thinking_effort, supported=(), family="Qwen3.5")
        self._tokenizer = load_tokenizer(model_name, tokenizer=tokenizer, hf_token=hf_token)
        self._enable_thinking = resolve_enable_thinking(
            enable_thinking=enable_thinking, preserve_thinking=preserve_thinking
        )

        # Qwen3.6's and Qwen3.8's templates take preserve_thinking natively (3.8 keeps
        # history reasoning when it is omitted), so an explicit True/False is forwarded
        # and None keeps the template default. Qwen3.5's template has no such flag.
        native = _chat_template_reads_preserve_thinking(self._tokenizer.chat_template)
        self._preserve_thinking = preserve_thinking if native else None
        self._chat_template: str | None = None
        if preserve_thinking and not native:
            self._chat_template = (
                files("post_training_cookbook.codecs.chat_templates")
                .joinpath("qwen3_5_preserve_thinking.jinja")
                .read_text()
            )
            warnings.warn(
                "Qwen35Codec: preserve_thinking=True is served by a forked chat template at "
                "post_training_cookbook/codecs/chat_templates/qwen3_5_preserve_thinking.jinja; the upstream model "
                "template does not support this option natively.",
                PreserveThinkingForkedWarning,
                stacklevel=2,
            )
        # Qwen3.8 thinks by default and Qwen3.5's small models do not, so ask the gen prompt itself.
        gen_prompt = self.decode(self.encode_messages([{"role": "user", "content": ""}], add_generation_prompt=True))
        self._gen_prompt_opens_think = gen_prompt.rstrip().endswith("<think>")

    def encode_messages(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        add_generation_prompt: bool = False,
    ) -> list[int]:
        kwargs: dict = {
            "tokenize": True,
            "add_generation_prompt": add_generation_prompt,
        }
        if tools:
            kwargs["tools"] = [{"type": "function", "function": t} for t in tools]
        if self._chat_template is not None:
            kwargs["chat_template"] = self._chat_template
        if self._enable_thinking is not None:
            kwargs["enable_thinking"] = self._enable_thinking
        if self._preserve_thinking is not None:
            kwargs["preserve_thinking"] = self._preserve_thinking
        return self._tokenizer.apply_chat_template(messages_for_template(messages), **kwargs)["input_ids"]

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


def _chat_template_reads_preserve_thinking(chat_template: str | dict | None) -> bool:
    """Whether the chat template reads `preserve_thinking` as a variable.

    A mention in a comment or string does not count. Parsed in Hugging Face's own Jinja
    environment, so any template it renders parses here; a missing template or a dict of
    named templates is not native.
    """
    if not isinstance(chat_template, str):
        return False
    env = _compile_jinja_template(chat_template).environment
    return "preserve_thinking" in find_undeclared_variables(env.parse(chat_template))


def _parse_tool_body_xml(raw: str, param_types: dict[str, dict[str, str]]) -> tuple[ToolCall | None, bool]:
    """Parse the nested-XML body of a Qwen3.5 <tool_call> block.

    Qwen3.5 renders each `<parameter=KEY>VALUE</parameter>` value **unquoted** and
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
