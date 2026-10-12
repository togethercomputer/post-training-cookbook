from __future__ import annotations

import re
from collections.abc import Callable

from post_training_cookbook.types import Message, ToolCall

_BLOCK_RE = re.compile(r"<think>(.*?)</think>|<tool_call>(.*?)</tool_call>", re.DOTALL)

_ToolBodyParser = Callable[[str], tuple[ToolCall | None, bool]]


def restore_response_text(*, content: str | None, reasoning: str | None) -> str:
    """Reconstruct text from endpoint channels using the shared think-fence grammar.

    Non-whitespace content proves the endpoint saw a closed think block, so untagged
    reasoning is wrapped before content is appended. Without content, reasoning stays
    unfenced because there is no evidence that the think block closed.
    """
    if not content or not content.strip():
        return reasoning or ""
    if not reasoning:
        return content
    if "<think>" in reasoning or "</think>" in reasoning:
        return reasoning + content
    return f"<think>{reasoning}</think>{content}"


def parse_after_open_think(text: str, parse_text: Callable[[str], tuple[Message, bool]]) -> tuple[Message, bool]:
    """Parse a turn sampled after a gen prompt that opened `<think>`.

    A turn cut off before `</think>` never left the think block, so it is all reasoning,
    including any tool call written inside it.
    """
    if text and "</think>" not in text:
        return Message(role="assistant", reasoning_content=text), True
    return parse_text(text)


def parse_response_blocks(text: str, parse_tool_body: _ToolBodyParser) -> tuple[Message, bool]:
    r"""Walk decoded response text, extracting reasoning and tool calls.

    The block grammar (`<think>` and `<tool_call>` wrappers) is shared by the Qwen
    and GLM families; only the tool-call body grammar differs, which the caller
    injects via `parse_tool_body`.

    Templates open-prime the generation prompt with `<think>` when thinking is
    enabled, so the response begins mid-think with only a closing `</think>`. We
    restore a bare opening tag in that case so the block regex matches. The
    whitespace some templates prime after the tag (`<think>\n`) is prompt framing,
    not reasoning, and each template re-frames reasoning on its own when
    re-rendering — so it must not be folded into `reasoning_content`, or GLM-5
    (which re-renders `reasoning_content` verbatim) loses prefix stability.
    """
    if "</think>" in text and "<think>" not in text:
        text = "<think>" + text

    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    parse_success = True
    last_end = 0

    for match in _BLOCK_RE.finditer(text):
        if match.start() > last_end:
            segment = text[last_end : match.start()].strip()
            if segment:
                text_parts.append(segment)

        thinking_content, tool_call_content = match.groups()
        if thinking_content is not None:
            reasoning_parts.append(thinking_content)
        else:
            tc, ok = parse_tool_body(tool_call_content)
            if ok and tc is not None:
                tool_calls.append(tc)
            else:
                parse_success = False
        last_end = match.end()

    if last_end < len(text):
        segment = text[last_end:].strip()
        if segment:
            text_parts.append(segment)

    if "<tool_call>" in text and "</tool_call>" not in text:
        parse_success = False

    msg = Message(role="assistant")
    if text_parts:
        msg["content"] = " ".join(text_parts)
    if reasoning_parts:
        msg["reasoning_content"] = "\n\n".join(reasoning_parts)
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg, parse_success
