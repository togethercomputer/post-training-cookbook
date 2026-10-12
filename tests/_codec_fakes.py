"""Shared codec fakes for tests that drive rendering through the ``MessageCodec`` protocol.

``DummyCodec`` is the single source of truth for the SFT and DPO rendering tests, so a new
protocol method is implemented once rather than once per test module.
"""

from __future__ import annotations

from typing import Any


class DummyCodec:
    """Predictable codec: encodes ``"role: content\\n"`` per message as ord() per char.

    With ``add_generation_prompt=True`` the ``"assistant: "`` header is appended, so
    rendering ``messages[:i+1]`` extends rendering ``messages[:i]`` exactly.
    """

    def encode_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: Any = None,
        add_generation_prompt: bool = False,
    ) -> list[int]:
        tokens: list[int] = []
        for msg in messages:
            tokens.extend(ord(ch) for ch in f"{msg['role']}: ")
            tokens.extend(ord(ch) for ch in f"{msg['content']}\n")
        if add_generation_prompt:
            tokens.extend(ord(ch) for ch in "assistant: ")
        return tokens

    def encode(self, text: str) -> list[int]:
        return [ord(ch) for ch in text]

    def turn_terminator(self, next_role: str | None) -> list[int]:
        return []  # "\n" already closes every turn
