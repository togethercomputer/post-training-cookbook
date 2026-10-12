"""Codec interface, shared encode-path normalization, and reasoning-in-history strategy.

Some chat templates strip ``<think>…</think>`` traces from earlier assistant turns once
the next turn begins. Re-rendering history is then not prefix-stable, which
matters whenever we splice trained tokens into rendered context (SFT bodies, GRPO
actions). There are two ways to cope, and they make opposite trade-offs:

- **Default (``preserve_thinking=None``):** keep the stock template. Qwen3.5 and
  Qwen3.6 strip history reasoning; the other families keep it. ``split_turns`` — reached via
  ``render_conversation`` (SFT) and ``flatten_episode`` (GRPO) — then breaks a
  conversation into one sequence per prefix-stable run. Each turn trains against
  the history the model actually sees at inference — faithful, at the cost of more
  (and partly redundant) sequences per conversation.
- **``preserve_thinking=False``:** strip history reasoning. Same render as ``None``
  where the stock template already strips. GLM-5 and Qwen3.8 forward it to their
  template flag (``clear_thinking`` / ``preserve_thinking``), the only way those
  templates drop history the default keeps; Kimi-K3 and MiniMax-M3 strip it from
  the messages before rendering.
- **``preserve_thinking=True``:** keep thinking in every turn, yielding a single
  prefix-stable sequence. Cheaper and simpler, but it trains on history the
  default-stripping inference path will not reproduce — use it only when inference
  also preserves thinking, or accept the train/inference mismatch. Templates that
  expose a flag for this (GLM-5's ``clear_thinking``, the ``preserve_thinking``
  of Qwen3.6 and Qwen3.8) are driven natively. Qwen3.5 needs a forked
  template and says so via ``PreserveThinkingForkedWarning``. Kimi-K3's encoder
  and MiniMax-M3's template render reasoning in every turn they are given, so
  ``None`` and ``True`` render identically, and ``False`` drops it from the
  messages first (``strip_history_reasoning``).

``enable_thinking`` is the orthogonal on/off switch for thinking at all;
``preserve_thinking=True`` implies it, and a family whose template cannot turn
thinking off (GLM-5) rejects ``False``. ``thinking_effort`` picks a family's
reasoning-budget level where the template has one (GLM-5, Kimi-K3); ``None``
keeps the template default, and families without the setting reject any value.
"""

from __future__ import annotations

import json
import warnings
from typing import Protocol, runtime_checkable

from transformers import PreTrainedTokenizerBase
from transformers.models.auto.tokenization_auto import AutoTokenizer

from post_training_cookbook.types import Message, Role, ToolCall, ToolSpec


class PreserveThinkingForkedWarning(UserWarning):
    """Emitted at construction when a codec serves preserve_thinking=True via a forked Jinja template."""


def load_tokenizer(
    model_name: str | None,
    *,
    tokenizer: PreTrainedTokenizerBase | None,
    hf_token: str | None,
    revision: str | None = None,
    trust_remote_code: bool = False,
) -> PreTrainedTokenizerBase:
    """Resolve the tokenizer a codec was constructed with; `tokenizer` wins, else `model_name` is downloaded.

    ``revision`` pins the checkpoint; pass it with ``trust_remote_code`` so the remote code that runs is
    fixed.
    """
    if tokenizer is not None:
        return tokenizer
    if model_name is None:
        raise ValueError("Provide model_name or tokenizer")
    return AutoTokenizer.from_pretrained(
        model_name, use_fast=True, token=hf_token, revision=revision, trust_remote_code=trust_remote_code
    )


def resolve_enable_thinking(*, enable_thinking: bool | None, preserve_thinking: bool | None) -> bool | None:
    """Apply the universal rule that preserving thinking requires thinking to be on."""
    if not preserve_thinking:
        return enable_thinking
    if enable_thinking is False:
        raise ValueError("preserve_thinking=True requires enable_thinking != False (nothing to preserve)")
    return True


def strip_history_reasoning(messages: list[Message]) -> list[Message]:
    """Drop ``reasoning_content`` from assistant turns before the last user message.

    Turns after it, the ones answering it, keep theirs.
    """
    last_user = max((i for i, msg in enumerate(messages) if msg["role"] == "user"), default=-1)
    return [
        {key: value for key, value in msg.items() if key != "reasoning_content"}
        if i < last_user and msg["role"] == "assistant"
        else msg
        for i, msg in enumerate(messages)
    ]


def check_thinking_effort(thinking_effort: str | None, *, supported: tuple[str, ...], family: str) -> str | None:
    """Validate a thinking-effort level against what the family's template accepts.

    ``None`` keeps the template's own default; a family with no effort setting passes
    ``supported=()`` and so rejects every other value.
    """
    if thinking_effort is None or thinking_effort in supported:
        return thinking_effort
    if not supported:
        raise ValueError(f"{family} has no thinking-effort setting; got thinking_effort={thinking_effort!r}")
    raise ValueError(f"{family} thinking_effort must be one of {supported}; got {thinking_effort!r}")


def messages_for_template(messages: list[Message]) -> list[dict]:
    """Adapt canonical messages for templates that do `tool_call.arguments | items`.

    Canonical Messages store arguments as a JSON string (OpenAI/HF convention), so the
    returned dicts are template input, not Messages — `arguments` is deliberately a dict.
    """
    out: list[dict] = []
    for msg in messages:
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            # `map`, not a comprehension: it adds no Python frame, so the `stacklevel` of a
            # warning raised below stays the same on every supported Python version.
            out.append({**msg, "tool_calls": list(map(_tool_call_with_dict_args, tool_calls))})
        else:
            out.append({**msg})
    return out


def _tool_call_with_dict_args(tool_call: ToolCall) -> dict:
    """Return a template-shaped tool call whose `arguments` is a dict, not a string."""
    name = tool_call["function"]["name"]
    args = tool_call["function"]["arguments"]
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            # stacklevel=3 points past this module at the codec's `encode_messages`, so the
            # warning identifies which family's render produced it.
            warnings.warn(
                f"tool call {name!r} has non-JSON arguments {args!r}; encoding with empty arguments.",
                stacklevel=3,
            )
            args = {}
    return {"id": tool_call.get("id"), "function": {"name": name, "arguments": args}}


@runtime_checkable
class MessageCodec(Protocol):
    """Model-specific bridge between structured messages and token sequences."""

    def encode_messages(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        add_generation_prompt: bool = False,
    ) -> list[int]:
        """Encode a conversation with the model's chat template."""
        ...

    @staticmethod
    def parse_completion(
        *, content: str | None, reasoning: str | None, tools: list[ToolSpec] | None = None
    ) -> tuple[Message, bool]:
        """Parse an endpoint completion whose server split reasoning from content. Tokenizer-independent.

        Each family decides what may be recovered from the split: the Qwen codecs re-join the channels
        and mine them for leaked tool calls; GLM-5 does so only when content is present, since a
        content-free completion may never have left the think block its gen prompt opened; families
        whose serving parser never runs a call written in reasoning keep ``reasoning`` as reasoning and
        parse only ``content``.
        """
        ...

    def parse_response(
        self, response_tokens: list[int], *, tools: list[ToolSpec] | None = None
    ) -> tuple[Message, bool]:
        """Decode sampled tokens into a structured Message with tool_calls extracted.

        ``tools`` supplies the declared parameter schema so each tool-call argument
        is parsed to its declared type. This matters for formats that serialize
        argument values without type information (e.g. the Qwen3.5 XML form renders
        them unquoted, so a string-typed zip code "32286" is otherwise coerced to
        int 32286 and fails string-typed validation downstream). Pass the same tool
        list given to ``encode_messages``; codecs whose wire format is self-describing
        ignore it.

        Returns (message, parse_success). parse_success is False if the tool call
        was malformed or tags were incomplete; when the gen prompt opens a think block
        (always for GLM-5, by default for Qwen3.8), a turn cut off inside it parses as
        reasoning, successfully.
        """
        ...

    def turn_terminator(self, next_role: Role | None) -> list[int]:
        """Tokens a trained assistant turn needs in order to stop, given what follows it.

        Empty when the template already renders the turn's own stop token inside the
        turn (Qwen's universal ``<|im_end|>``). Templates that instead close an assistant
        turn with the *next* message's role marker (GLM-5) return that marker, so each
        trained turn learns the stop that actually belongs to it: yield to the user, hand
        off to a tool, or end the conversation (``next_role=None``).

        Raises ``ValueError`` when the family declares no stop token for ``next_role``
        (e.g. GLM-5 followed by another assistant turn); the caller must merge, split,
        or drop the turn rather than train a marker that does not stop generation.
        """
        ...

    def encode(self, text: str) -> list[int]:
        """Encode raw text without chat-template wrapping."""
        ...

    def decode(self, tokens: list[int]) -> str:
        """Decode raw tokens, skipping special tokens other than the markers the codec's parser needs."""
        ...
