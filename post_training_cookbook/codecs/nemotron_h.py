r"""Nemotron 3.5 codecs: the checkpoint's own chat template, with the nested-XML tool-call grammar.

``nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16`` (``Nemotron35Codec``) and
``nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`` (``Nemotron35LightningCodec``) render ChatML turns
with a ``<think>`` block and the same ``<tool_call><function=…><parameter=…>`` body as Qwen3.5, so parsing
comes from ``NestedXmlCodec``, as ``Qwen35Codec``'s does. Their templates differ from Qwen's in the flags
they read:

- Thinking is on by default: the generation prompt opens ``<think>\n``, and ``enable_thinking=False``
  closes it empty (``<think></think>``). A turn without reasoning also re-renders as ``<think></think>``,
  so with thinking on such a turn does not extend the prompt it was sampled from; ``split_turns`` starts a
  new sequence there.
- ``truncate_history_thinking`` (default on) drops reasoning from assistant turns before the last user
  message. ``preserve_thinking=True`` turns it off natively, and ``False`` sets it explicitly.
- Super VL only: ``medium_effort`` appends ``{reasoning effort: efficient}`` to the last user message, with thinking on
  or off. It is ``thinking_effort="medium"``; ``"high"`` is the template default. Since the hint moves to
  each new user message, a prompt rendered before a user turn does not extend the next one, so ``"medium"``
  splits a multi-turn episode at every user message, even with ``preserve_thinking=True``. To serve a
  model trained this way with the same prompts, chat-completions requests must pass
  ``chat_template_kwargs={"medium_effort": true}``: SGLang does not map ``reasoning_effort="medium"`` to
  this flag for this template, and the template ignores ``reasoning_effort``.

Lightning's template has no effort flag, so its codec rejects any ``thinking_effort``. It also concatenates
content as a plain string where Super VL's ``render_content`` coerces it: a ``None`` system message raises
``TypeError`` and a ``None`` user or tool message renders the word ``None``, so the Lightning codec passes
``None`` content as ``""``, which is how Super VL renders it. List-of-parts content is not supported.
Its tokenizer names Super VL's vision tokens (ids 18-29: ``<image>``, ``<box>``…) ``<SPECIAL_n>``, so those
literal strings tokenize as ordinary text; all other text tokenizes the same.

Each tokenizer and template is pinned to the revision Together's RL catalog serves, so every name that
routes to a codec (BF16, FP8 and other precisions) renders with the same files. Other Nemotron 3.5 models
(Nano) route to neither, since they ship their own tokenizers.
"""

from __future__ import annotations

from transformers import PreTrainedTokenizerBase

from post_training_cookbook.codecs.base import (
    check_thinking_effort,
    load_tokenizer,
    messages_for_template,
    resolve_enable_thinking,
)
from post_training_cookbook.codecs.nested_xml import NestedXmlCodec
from post_training_cookbook.types import Message, ToolSpec

NEMOTRON_35_SUPER_VL = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"
NEMOTRON_35_SUPER_VL_REVISION = "fb0ee0bc6b3f70342afd8d84cb80dbadbaa8a1fe"
NEMOTRON_35_LIGHTNING = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
NEMOTRON_35_LIGHTNING_REVISION = "a9904d24bcc1d289a1950fa9d2b978c47cf903b9"


class Nemotron35Codec(NestedXmlCodec):
    """Nemotron 3.5 Super VL: renders with the checkpoint's template, parses nested-XML tool calls."""

    _CHECKPOINT = NEMOTRON_35_SUPER_VL
    _REVISION = NEMOTRON_35_SUPER_VL_REVISION
    _THINKING_EFFORTS: tuple[str, ...] = ("medium", "high")
    _FAMILY = "Nemotron 3.5 Super VL"

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
            thinking_effort, supported=self._THINKING_EFFORTS, family=self._FAMILY
        )
        self._tokenizer = load_tokenizer(
            self._CHECKPOINT, tokenizer=tokenizer, hf_token=hf_token, revision=self._REVISION
        )
        self._enable_thinking = resolve_enable_thinking(
            enable_thinking=enable_thinking, preserve_thinking=preserve_thinking
        )
        self._preserve_thinking = preserve_thinking
        self._detect_gen_prompt_think()

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
        # The template adds the effort hint with thinking off too, so it is passed regardless of enable_thinking.
        if self._thinking_effort is not None:
            kwargs["medium_effort"] = self._thinking_effort == "medium"
        return self._tokenizer.apply_chat_template(self._template_messages(messages), **kwargs)["input_ids"]

    def _template_messages(self, messages: list[Message]) -> list[dict]:
        return messages_for_template(messages)


class Nemotron35LightningCodec(Nemotron35Codec):
    """Nemotron 3.5 Lightning: Super VL's thinking and history flags and tool-call grammar, without an effort flag."""

    _CHECKPOINT = NEMOTRON_35_LIGHTNING
    _REVISION = NEMOTRON_35_LIGHTNING_REVISION
    _THINKING_EFFORTS = ()
    _FAMILY = "Nemotron 3.5 Lightning"

    def _template_messages(self, messages: list[Message]) -> list[dict]:
        # The template concatenates content with `+` and `| string`, so None would crash or render "None".
        return [
            {**m, "content": ""} if m.get("content", "") is None else m for m in super()._template_messages(messages)
        ]
