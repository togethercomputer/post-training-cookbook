"""Unit tests for `parse_response_blocks`.

The roundtrip tests in test_codec.py cover the end-to-end invariant via the
real tokenizer. These tests pin down the specific text shapes the helper must
accept — including the case where the chat template primes the gen prompt with
an open `<think>`, so the decoded response begins mid-think with only the
closing tag.
"""

from __future__ import annotations

from post_training_cookbook.codecs.response_blocks import parse_response_blocks


def _no_tools(_raw: str) -> tuple[None, bool]:
    return None, True


class TestParseResponseBlocks:
    def test_full_think_block(self):
        """Both tags present: the model emits its own opening and closing tags."""
        text = "<think>reasoning here</think>\n\nthe answer"
        msg, ok = parse_response_blocks(text, _no_tools)
        assert ok
        assert msg["reasoning_content"] == "reasoning here"
        assert msg["content"] == "the answer"

    def test_primed_think_block_no_opening_tag(self):
        """Qwen3.5 with thinking on: the `<think>\\n` opener is in the prompt,
        so the response only contains the closing `</think>`. The helper must
        treat the prefix as reasoning anyway."""
        text = "reasoning here\n</think>\n\nthe answer"
        msg, ok = parse_response_blocks(text, _no_tools)
        assert ok
        # Exact: the restored opening tag must be bare, or templates that re-render
        # `reasoning_content` verbatim (GLM-5) gain a stray leading newline and stop
        # reproducing the sampled tokens.
        assert msg["reasoning_content"] == "reasoning here\n"
        assert msg["content"] == "the answer"
        assert "</think>" not in msg["content"]

    def test_no_think_markers(self):
        """No thinking at all — entire text is content."""
        text = "just a plain answer"
        msg, ok = parse_response_blocks(text, _no_tools)
        assert ok
        assert msg["content"] == "just a plain answer"
        assert "reasoning_content" not in msg

    def test_empty_primed_think_block(self):
        """A closed-empty think block carries no reasoning."""
        text = "</think>\n\nthe answer"
        msg, ok = parse_response_blocks(text, _no_tools)
        assert ok
        assert msg.get("reasoning_content", "").strip() == ""
        assert msg["content"] == "the answer"

    def test_primed_think_followed_by_tool_call(self):
        """Qwen3.5 case with a tool call after reasoning. Tool body parser is
        stubbed to return a sentinel so we can verify it was invoked."""
        captured: list[str] = []

        def parse_tool(raw: str):
            captured.append(raw)
            return {"id": "x"}, True  # type: ignore[return-value]

        text = "I need to call a tool\n</think>\n\n<tool_call>BODY</tool_call>"
        msg, ok = parse_response_blocks(text, parse_tool)
        assert ok
        assert "I need to call a tool" in msg["reasoning_content"]
        assert captured == ["BODY"]
        assert len(msg["tool_calls"]) == 1

    def test_literal_close_tag_with_no_open_is_treated_as_primed(self):
        """Defensive: even on a degenerate input with no reasoning text before
        `</think>`, the helper should not blow up and should yield empty
        reasoning + everything-after-the-tag as content."""
        text = "</think>answer"
        msg, ok = parse_response_blocks(text, _no_tools)
        assert ok
        assert msg["content"] == "answer"
