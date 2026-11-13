from __future__ import annotations

import warnings
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
from post_training_cookbook.codecs.nested_xml import NestedXmlCodec
from post_training_cookbook.types import Message, ToolSpec


class Qwen35Codec(NestedXmlCodec):
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
        # Qwen3.8 thinks by default and Qwen3.5's small models do not.
        self._detect_gen_prompt_think()

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
