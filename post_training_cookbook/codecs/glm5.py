"""GLM-5.3 codec.

GLM-5.3 always thinks: the generation prompt opens ``<think>`` whatever ``enable_thinking`` says, so
the codec rejects ``enable_thinking=False``. The template prepends ``Reasoning Effort: Low|High|Max``
(``Max`` by default) and keeps history reasoning unless ``clear_thinking`` is set, which is how
``preserve_thinking=False`` is served.
"""

from __future__ import annotations

import json
import re
from functools import partial

from transformers import PreTrainedTokenizerBase

from post_training_cookbook.codecs.base import (
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

_ARG_RE = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL)

# GLM's template closes an assistant turn with the *next* message's role marker, so the
# token that stops a turn depends on what follows it. These are exactly the three stops
# generation_config.json declares: end the conversation, yield to the user, hand off to a
# tool. Any other continuation (a back-to-back assistant turn, an interleaved system
# message) has no stop token, so `turn_terminator` refuses it rather than training a
# marker that does not stop generation.
_TURN_TERMINATORS: dict[Role | None, str] = {
    None: "<|endoftext|>",
    "user": "<|user|>",
    "tool": "<|observation|>",
}


# The template renders `Reasoning Effort: Low`/`High` for those levels and `Max` for anything else.
_THINKING_EFFORTS = ("low", "high", "max")


class Glm5Codec:
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
        self._thinking_effort = check_thinking_effort(thinking_effort, supported=_THINKING_EFFORTS, family="GLM-5")
        if resolve_enable_thinking(enable_thinking=enable_thinking, preserve_thinking=preserve_thinking) is False:
            raise ValueError("GLM-5.3 always thinks; its template has no enable_thinking=False mode")
        self._preserve_thinking = preserve_thinking
        self._tokenizer = load_tokenizer(model_name, tokenizer=tokenizer, hf_token=hf_token)

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
        if self._preserve_thinking is not None:
            # The template preserves history reasoning unless clear_thinking is set.
            kwargs["clear_thinking"] = not self._preserve_thinking
        if self._thinking_effort is not None:
            kwargs["reasoning_effort"] = self._thinking_effort
        return self._tokenizer.apply_chat_template(messages_for_template(messages), **kwargs)["input_ids"]

    @staticmethod
    def parse_text(text: str, *, tools: list[ToolSpec] | None = None) -> tuple[Message, bool]:
        param_types = param_types_by_tool(tools)
        return parse_response_blocks(text, lambda raw: _parse_tool_body_kv(raw, param_types))

    @classmethod
    def parse_completion(
        cls, *, content: str | None, reasoning: str | None, tools: list[ToolSpec] | None = None
    ) -> tuple[Message, bool]:
        # Content means the endpoint saw `</think>` close, so only a content-free completion can
        # still be inside the think block the gen prompt opened.
        if content and content.strip():
            return cls.parse_text(restore_response_text(content=content, reasoning=reasoning), tools=tools)
        return parse_after_open_think(reasoning or "", partial(cls.parse_text, tools=tools))

    def parse_response(
        self, response_tokens: list[int], *, tools: list[ToolSpec] | None = None
    ) -> tuple[Message, bool]:
        return parse_after_open_think(self.decode(response_tokens), partial(self.parse_text, tools=tools))

    def turn_terminator(self, next_role: Role | None) -> list[int]:
        if next_role not in _TURN_TERMINATORS:
            raise ValueError(
                f"GLM-5 has no stop token for an assistant turn followed by {next_role!r}; "
                "merge the turns, split the conversation, or drop the turn."
            )
        return self.encode(_TURN_TERMINATORS[next_role])

    def encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=False))

    def decode(self, tokens: list[int]) -> str:
        return str(self._tokenizer.decode(tokens, skip_special_tokens=True))


def _parse_tool_body_kv(raw: str, param_types: dict[str, dict[str, str]]) -> tuple[ToolCall | None, bool]:
    """Parse the arg-key/arg-value body of a GLM-5 <tool_call> block.

    The body is the function name followed by
    `<arg_key>K</arg_key><arg_value>V</arg_value>` pairs. The template renders a
    string value **raw** and everything else as JSON, so the wire form carries no
    type information; each value is coerced against the param's declared JSON-schema
    type (``param_types[name][key]``), which is what keeps a string-typed numeric arg
    a string rather than an int (the mismatch that fails downstream schema validation,
    i.e. the serverless-vs-generator reward gap). Coercion never raises.

    Values are coerced unstripped: GLM adds no framing whitespace inside `<arg_value>`,
    so any edge whitespace is part of the value (the serverless tool-call parser trims,
    so a padded string-typed arg parses differently there).
    """
    name = raw.split("<", 1)[0].strip()
    if not name:
        return None, False

    types_for_tool = param_types.get(name, {})
    arguments: dict[str, object] = {}
    for arg_match in _ARG_RE.finditer(raw):
        key = arg_match.group(1).strip()
        if not key:
            return None, False
        arguments[key] = coerce_by_schema(arg_match.group(2), types_for_tool.get(key))

    return ToolCall(
        id=None,
        function=ToolCallFunction(name=name, arguments=json.dumps(arguments)),
    ), True
