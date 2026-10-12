"""GLM-5 codec: the arg-key/arg-value parser, role-dependent turn terminators, and re-encode
fidelity of sampled thinking turns."""

from __future__ import annotations

import json

import pytest

from post_training_cookbook.codecs import get_codec
from post_training_cookbook.codecs.base import MessageCodec
from post_training_cookbook.codecs.glm5 import _parse_tool_body_kv
from post_training_cookbook.sft.train import TRAIN_ON_ALL, render_conversation
from post_training_cookbook.types import Message, Role, ToolCall, ToolCallFunction
from tests.codecs import ARG_SCHEMA, REFERENCE_MODELS, TWO_TURN_CONVERSATION, single_token, trained_runs

_GLM5 = REFERENCE_MODELS["glm-5"]


@pytest.fixture(scope="session")
def glm5() -> MessageCodec:
    return get_codec(_GLM5)


def test_rejects_thinking_off():
    with pytest.raises(ValueError, match="always thinks"):
        get_codec(_GLM5, enable_thinking=False)


def test_unclosed_think_is_reasoning(glm5: MessageCodec):
    """The gen prompt always opens `<think>`, so a turn cut off before `</think>` is all reasoning, and a
    tool call written inside it is not executed."""
    text = "still thinking <tool_call>lookup<arg_key>zip</arg_key><arg_value>1</arg_value></tool_call>"
    assert glm5.parse_response(glm5.encode(text)) == (Message(role="assistant", reasoning_content=text), True)


class TestGlm5KvArgParsing:
    """Direct, format-specific tests of the GLM-5 arg-key/arg-value parser
    (deterministic, not subject to template rendering)."""

    def test_declared_types_respected(self):
        """GLM renders a string value raw and everything else via `tojson`, so the
        literals are JSON-cased (`true`/`null`). Coercion keys off the declared
        schema type, not the literal's casing."""
        raw = (
            "lookup"
            "<arg_key>zip</arg_key><arg_value>32286</arg_value>"
            "<arg_key>count</arg_key><arg_value>3</arg_value>"
            "<arg_key>flag</arg_key><arg_value>true</arg_value>"
            "<arg_key>opt</arg_key><arg_value>null</arg_value>"
            '<arg_key>tags</arg_key><arg_value>["a", "b"]</arg_value>'
            '<arg_key>meta</arg_key><arg_value>{"k": 1}</arg_value>'
        )
        tc, ok = _parse_tool_body_kv(raw, ARG_SCHEMA)
        assert ok
        assert tc["function"]["name"] == "lookup"
        args = json.loads(tc["function"]["arguments"])
        assert args["zip"] == "32286" and isinstance(args["zip"], str)
        assert args["count"] == 3 and isinstance(args["count"], int)
        assert args["flag"] is True
        assert args["opt"] is None
        assert args["tags"] == ["a", "b"]
        assert args["meta"] == {"k": 1}

    def test_whitespace_significant_string_preserved_verbatim(self):
        """GLM adds no framing inside `<arg_value>`, so edge whitespace and interior
        newlines in a string-typed value are content and must survive untouched."""
        body = "\n  indented line\n\nblank above  "
        tc, ok = _parse_tool_body_kv(f"lookup<arg_key>note</arg_key><arg_value>{body}</arg_value>", ARG_SCHEMA)
        assert ok
        assert json.loads(tc["function"]["arguments"])["note"] == body

    def test_value_containing_angle_bracket(self):
        tc, ok = _parse_tool_body_kv("lookup<arg_key>note</arg_key><arg_value>a < b</arg_value>", ARG_SCHEMA)
        assert ok
        assert json.loads(tc["function"]["arguments"])["note"] == "a < b"

    def test_unknown_param_falls_back_to_coercion(self):
        tc, ok = _parse_tool_body_kv("lookup<arg_key>surprise</arg_key><arg_value>7</arg_value>", ARG_SCHEMA)
        assert ok
        assert json.loads(tc["function"]["arguments"])["surprise"] == 7

    def test_malformed_body_does_not_raise(self):
        """A name-only call is legal (no args); an unterminated value simply yields no
        args; a missing name fails the parse rather than inventing one."""
        tc, ok = _parse_tool_body_kv("get_time", ARG_SCHEMA)
        assert ok
        assert tc["function"]["name"] == "get_time"
        assert json.loads(tc["function"]["arguments"]) == {}

        tc, ok = _parse_tool_body_kv("lookup<arg_key>zip</arg_key><arg_value>32286", ARG_SCHEMA)
        assert ok
        assert json.loads(tc["function"]["arguments"]) == {}

        tc, ok = _parse_tool_body_kv("<arg_key>zip</arg_key><arg_value>32286</arg_value>", ARG_SCHEMA)
        assert not ok and tc is None

        tc, ok = _parse_tool_body_kv("lookup<arg_key></arg_key><arg_value>32286</arg_value>", ARG_SCHEMA)
        assert not ok and tc is None


class TestGlm5TurnTerminators:
    """GLM closes an assistant turn with the next message's role marker, so unlike Qwen's
    universal `<|im_end|>` the stop token a turn must learn is role-dependent. Training
    `<|endoftext|>` on an intermediate turn teaches the model to end the conversation where
    it should have yielded to the user or called a tool."""

    @pytest.mark.parametrize(
        "next_role, expected",
        [(None, "<|endoftext|>"), ("user", "<|user|>"), ("tool", "<|observation|>")],
    )
    def test_terminator_follows_the_next_role(self, glm5: MessageCodec, next_role: Role | None, expected: str):
        assert glm5.turn_terminator(next_role) == glm5.encode(expected)

    def test_intermediate_turn_yields_instead_of_ending(self, glm5: MessageCodec):
        sequences = render_conversation(TWO_TURN_CONVERSATION, codec=glm5, train_on_what=TRAIN_ON_ALL, max_length=4096)
        runs = trained_runs(sequences)
        end_of_text = single_token(glm5, "<|endoftext|>")
        assert runs[0][-1] == single_token(glm5, "<|user|>")
        assert end_of_text not in runs[0]
        assert runs[-1][-1] == end_of_text

    def test_appended_terminator_coincides_with_the_template_boundary(self, glm5: MessageCodec):
        """The terminator is synthetic, so it must land exactly where the template's own turn
        boundary is. If it ever drifts (upstream renders `\\n<|user|>`, or the tokenizer changes),
        `split_turns` silently emits a second sequence — re-encoding turn 1 as context and
        training a boundary the model never sees at inference. Every per-turn assertion above
        survives that regression, so pin the whole render."""
        sequences = render_conversation(TWO_TURN_CONVERSATION, codec=glm5, train_on_what=TRAIN_ON_ALL, max_length=4096)
        assert len(sequences) == 1
        assert sequences[0].tokens == glm5.encode_messages(TWO_TURN_CONVERSATION) + glm5.turn_terminator(None)

    def test_tool_handoff_turn_trains_observation(self, glm5: MessageCodec):
        messages: list[Message] = [
            Message(role="user", content="weather?"),
            Message(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        function=ToolCallFunction(name="get_weather", arguments=json.dumps({"location": "Paris"})),
                    )
                ],
            ),
            Message(role="tool", content='{"temp": 20}'),
            Message(role="assistant", content="it is sunny"),
        ]
        runs = trained_runs(render_conversation(messages, codec=glm5, train_on_what=TRAIN_ON_ALL, max_length=4096))
        assert runs[0][-1] == single_token(glm5, "<|observation|>")
        assert runs[-1][-1] == single_token(glm5, "<|endoftext|>")

    @pytest.mark.parametrize("next_role", ["assistant", "system"])
    def test_continuation_without_a_stop_token_is_refused(self, glm5: MessageCodec, next_role: Role):
        """`<|assistant|>` and `<|system|>` are role markers, not declared stops, so a turn
        followed by one has no stop to train. Refusing beats training a marker that does not
        stop generation — the runaway-generation bug `turn_terminator` exists to close."""
        with pytest.raises(ValueError, match="no stop token"):
            glm5.turn_terminator(next_role)

    def test_back_to_back_assistant_turns_refused(self, glm5: MessageCodec):
        """The refusal must reach the caller, not just the codec method: consecutive assistant
        turns occur in real datasets (multi-part answers)."""
        messages: list[Message] = [
            Message(role="user", content="u1"),
            Message(role="assistant", content="a1"),
            Message(role="assistant", content="a2"),
        ]
        with pytest.raises(ValueError, match="no stop token"):
            render_conversation(messages, codec=glm5, train_on_what=TRAIN_ON_ALL, max_length=4096)


class TestGlm5ReencodeFidelity:
    r"""GLM's template renders `reasoning_content` verbatim, so `parse_response` must not
    invent framing around it: the shared parser restores a *bare* `<think>` when the gen
    prompt primed one. Restoring `<think>\n` instead drifts a token on every thinking turn,
    silently corrupting multi-turn GRPO, which re-encodes each parsed action."""

    @pytest.mark.parametrize("preserve_thinking", [False, True])
    @pytest.mark.parametrize(
        "sampled_text, byte_identical",
        [
            # GLM primes `<think>` with no newline, so a sample starts mid-think.
            ("reasoning</think>text", True),
            # The boundary: GLM's template emits assistant content as `content.strip()`, so
            # whitespace after `</think>` cannot be reproduced by any re-encode. The render
            # stays a prefix, but the turn is no longer byte-identical — which is exactly the
            # single-sequence collapse `preserve_thinking=True` exists for, so GLM only gets
            # it when the sample has no whitespace there. Fixing it needs a forked template;
            # pinning it here means such a fork trips this test instead of passing silently.
            ("reasoning</think>\n\ntext", False),
        ],
    )
    def test_sampled_response_survives_reencode(
        self, preserve_thinking: bool, sampled_text: str, byte_identical: bool
    ):
        codec = get_codec(_GLM5, preserve_thinking=preserve_thinking)
        prompt: list[Message] = [Message(role="user", content="q")]
        prompt_tokens = codec.encode_messages(prompt, add_generation_prompt=True)
        sampled = codec.encode(sampled_text)

        msg, ok = codec.parse_response(sampled)
        assert ok

        full = codec.encode_messages(prompt + [msg])
        assert full[: len(prompt_tokens)] == prompt_tokens, "history render is not prefix-stable"
        assert (full[len(prompt_tokens) :] == sampled) is byte_identical, (
            "re-encode fidelity changed for this sample shape"
        )
