r"""Nemotron 3.5 codec: the checkpoint's own chat template, with Qwen3.5's nested-XML tool-call grammar.

``nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16`` renders ChatML turns with a ``<think>`` block and
the same ``<tool_call><function=…><parameter=…>`` body as Qwen3.5, so parsing is shared with
``Qwen35Codec``. Its template differs in the flags it reads:

- Thinking is on by default: the generation prompt opens ``<think>\n``, and ``enable_thinking=False``
  closes it empty (``<think></think>``). A turn without reasoning also re-renders as ``<think></think>``,
  so with thinking on such a turn does not extend the prompt it was sampled from; ``split_turns`` starts a
  new sequence there.
- ``truncate_history_thinking`` (default on) drops reasoning from assistant turns before the last user
  message. ``preserve_thinking=True`` turns it off natively, and ``False`` sets it explicitly.
- ``medium_effort`` appends ``{reasoning effort: efficient}`` to the last user message, with thinking on
  or off. It is ``thinking_effort="medium"``; ``"high"`` is the template default. Since the hint moves to
  each new user message, a prompt rendered before a user turn does not extend the next one, so ``"medium"``
  splits a multi-turn episode at every user message, even with ``preserve_thinking=True``. That matches
  what inference sees.

The tokenizer and template are pinned to the revision Together's RL catalog serves, so every Super VL name
that routes here (BF16, FP8 and other precisions) renders with the same files. Other Nemotron 3.5 models
do not route here, since they ship their own tokenizers.
"""

from __future__ import annotations

from transformers import PreTrainedTokenizerBase

from post_training_cookbook.codecs.base import (
    check_thinking_effort,
    load_tokenizer,
    messages_for_template,
    resolve_enable_thinking,
)
from post_training_cookbook.codecs.qwen3_5 import Qwen35Codec
from post_training_cookbook.types import Message, ToolSpec

NEMOTRON_35_SUPER_VL = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"
NEMOTRON_35_SUPER_VL_REVISION = "fb0ee0bc6b3f70342afd8d84cb80dbadbaa8a1fe"
_THINKING_EFFORTS = ("medium", "high")


class Nemotron35Codec(Qwen35Codec):
    """Nemotron 3.5: renders with the checkpoint's template, parses Qwen3.5's XML tool calls."""

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
        self._thinking_effort = check_thinking_effort(
            thinking_effort, supported=_THINKING_EFFORTS, family="Nemotron 3.5"
        )
        self._tokenizer = load_tokenizer(
            NEMOTRON_35_SUPER_VL, tokenizer=tokenizer, hf_token=hf_token, revision=NEMOTRON_35_SUPER_VL_REVISION
        )
        self._enable_thinking = resolve_enable_thinking(
            enable_thinking=enable_thinking, preserve_thinking=preserve_thinking
        )
        self._preserve_thinking = preserve_thinking
        gen_prompt = self.decode(self.encode_messages([{"role": "user", "content": ""}], add_generation_prompt=True))
        self._gen_prompt_opens_think = gen_prompt.rstrip().endswith("<think>")

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
        if self._enable_thinking is not None:
            kwargs["enable_thinking"] = self._enable_thinking
        if self._preserve_thinking is not None:
            kwargs["truncate_history_thinking"] = not self._preserve_thinking
        # The template adds the effort hint with thinking off too, so it is passed through as serving does.
        if self._thinking_effort is not None:
            kwargs["medium_effort"] = self._thinking_effort == "medium"
        return self._tokenizer.apply_chat_template(messages_for_template(messages), **kwargs)["input_ids"]
