"""Contract tests for the MessageCodec interface.

Parametrized over all registered codecs via conftest.py.
Round-trip tests verify correctness without hardcoding format details.
"""

from __future__ import annotations

import json
import re

import pytest

from post_training_cookbook.codecs import (
    Glm5Codec,
    KimiK3Codec,
    MiniMaxM3Codec,
    Qwen35Codec,
    get_codec,
    parse_completion,
    resolve_codec_class,
)
from post_training_cookbook.codecs.base import MessageCodec, messages_for_template
from post_training_cookbook.sft.train import TRAIN_ON_ALL, render_conversation
from post_training_cookbook.types import Message, ToolCall, ToolCallFunction, ToolSpec
from tests.codecs import REFERENCE_MODELS, THINKING_OFF_MODELS, TWO_TURN_CONVERSATION, single_token, trained_runs

_GLM5 = REFERENCE_MODELS["glm-5"]
_MODELS = list(REFERENCE_MODELS.values())

# Whether a family's template renders history reasoning without preserve_thinking. Either way
# split_turns copes; preserve_thinking must keep it for every family.
_KEEPS_HISTORY_REASONING: dict[str, bool] = {
    REFERENCE_MODELS["qwen3.5"]: False,
    _GLM5: True,
    REFERENCE_MODELS["kimi-k3"]: True,
    REFERENCE_MODELS["minimax-m3"]: True,
}


def _extract_response_tokens(codec: MessageCodec, prompt: list[Message], assistant_msg: Message) -> list[int]:
    """Get just the assistant response tokens by diffing full vs prompt-only encodings."""
    prompt_tokens = codec.encode_messages(prompt, add_generation_prompt=True)
    full_tokens = codec.encode_messages(prompt + [assistant_msg])
    assert full_tokens[: len(prompt_tokens)] == prompt_tokens, "history render is not prefix-stable"
    return full_tokens[len(prompt_tokens) :]


class TestEncodeDecodeRoundTrip:
    def test_text_survives_round_trip(self, codec: MessageCodec):
        original = "The answer is 42."
        assert codec.decode(codec.encode(original)) == original

    def test_empty_string(self, codec: MessageCodec):
        assert codec.encode("") == []


class TestEncodeMessagesRoundTrip:
    def test_message_content_preserved(self, codec: MessageCodec):
        messages: list[Message] = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="What is 2+2?"),
            Message(role="assistant", content="It is 4."),
            Message(role="user", content="Thanks!"),
        ]
        decoded = codec.decode(codec.encode_messages(messages))
        for msg in messages:
            assert msg["content"] in decoded

    def test_tools_produce_tokens(self, codec: MessageCodec):
        messages: list[Message] = [Message(role="user", content="weather?")]
        tools: list[ToolSpec] = [
            ToolSpec(
                name="get_weather",
                description="Get weather",
                parameters={
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            )
        ]
        tokens = codec.encode_messages(messages, tools=tools)
        assert len(tokens) > 0


class TestParseResponseRoundTrip:
    def test_plain_text_recovered(self, codec_no_thinking: MessageCodec):
        """Thinking off, so unmarked text is the answer; a prompt that leaves a think block open
        (Kimi-K3's default) would make the same text reasoning."""
        codec = codec_no_thinking
        text = "The answer is 42."
        msg, ok = codec.parse_response(codec.encode(text))
        assert ok
        assert msg["role"] == "assistant"
        assert text in msg.get("content", "")

    def test_reasoning_content_recovered(self, codec_preserve_thinking: MessageCodec):
        """Template-encode an assistant turn with reasoning, then parse_response
        should split it back into reasoning_content vs content.

        Exercises the case where the gen prompt is primed with an open `<think>` so
        response tokens contain only the closing `</think>` — the parser must still
        recover the reasoning rather than dumping it into `content`.
        """
        prompt: list[Message] = [Message(role="user", content="q")]
        assistant_msg = Message(
            role="assistant",
            reasoning_content="step one then step two",
            content="final answer is 4",
        )
        response_tokens = _extract_response_tokens(codec_preserve_thinking, prompt, assistant_msg)
        msg, ok = codec_preserve_thinking.parse_response(response_tokens)
        assert ok
        assert "step one then step two" in msg.get("reasoning_content", "")
        assert "final answer is 4" in msg.get("content", "")
        assert "step one then step two" not in msg.get("content", "")
        assert "</think>" not in msg.get("content", "")

    def test_reasoning_with_tool_call_recovered(self, codec_preserve_thinking: MessageCodec):
        """Reasoning followed by a tool call: both fields should be recovered,
        content should be empty/absent since the model only emitted a tool call."""
        prompt: list[Message] = [Message(role="user", content="weather?")]
        assistant_msg = Message(
            role="assistant",
            reasoning_content="I should call the weather tool",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    function=ToolCallFunction(
                        name="get_weather",
                        arguments=json.dumps({"location": "Paris"}),
                    ),
                )
            ],
        )
        response_tokens = _extract_response_tokens(codec_preserve_thinking, prompt, assistant_msg)
        msg, ok = codec_preserve_thinking.parse_response(response_tokens)
        assert ok
        assert "I should call the weather tool" in msg.get("reasoning_content", "")
        assert len(msg.get("tool_calls", [])) == 1
        assert msg["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_tool_call_recovered(self, codec: MessageCodec):
        """Template-encode an assistant tool call, then parse_response should recover it."""
        prompt: list[Message] = [Message(role="user", content="What is the weather?")]
        assistant_msg = Message(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    function=ToolCallFunction(
                        name="get_weather",
                        arguments=json.dumps({"location": "Paris"}),
                    ),
                )
            ],
        )
        response_tokens = _extract_response_tokens(codec, prompt, assistant_msg)
        msg, ok = codec.parse_response(response_tokens)
        assert ok
        assert len(msg.get("tool_calls", [])) == 1
        tc = msg["tool_calls"][0]
        assert tc["function"]["name"] == "get_weather"
        assert json.loads(tc["function"]["arguments"])["location"] == "Paris"

    def test_multiple_tool_calls_recovered(self, codec: MessageCodec):
        prompt: list[Message] = [Message(role="user", content="Weather in two cities?")]
        assistant_msg = Message(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    function=ToolCallFunction(
                        name="get_weather",
                        arguments=json.dumps({"location": "Paris"}),
                    ),
                ),
                ToolCall(
                    id="call_2",
                    function=ToolCallFunction(
                        name="get_weather",
                        arguments=json.dumps({"location": "London"}),
                    ),
                ),
            ],
        )
        response_tokens = _extract_response_tokens(codec, prompt, assistant_msg)
        msg, ok = codec.parse_response(response_tokens)
        assert ok
        assert len(msg.get("tool_calls", [])) == 2

    def test_parsed_then_reencoded_reproduces_tokens(self, codec: MessageCodec):
        """`flatten_episode` re-encodes parsed assistant turns; the round-trip
        `encode(parse(encode(msg))) == encode(msg)` must hold so a parsed turn's
        tokens reproduce exactly. Exercises a multi-turn history with a tool call so
        the XML wire format (Qwen3.5) is on the path."""
        tools: list[ToolSpec] = [
            ToolSpec(
                name="get_weather",
                description="Get weather",
                parameters={"type": "object", "properties": {"location": {"type": "string"}}},
            )
        ]
        prompt: list[Message] = [
            Message(role="user", content="weather in Paris?"),
        ]
        assistant_msg = Message(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    function=ToolCallFunction(
                        name="get_weather",
                        arguments=json.dumps({"location": "Paris"}),
                    ),
                )
            ],
        )
        original = _extract_response_tokens(codec, prompt, assistant_msg)
        parsed, ok = codec.parse_response(original, tools=tools)
        assert ok
        reencoded = _extract_response_tokens(codec, prompt, parsed)
        assert reencoded == original

    def test_tool_call_cut_short_never_parses_ok(self, codec: MessageCodec):
        """Multi-turn GRPO detects a length-truncated turn only by `ok`, so a turn cut anywhere
        after it starts diverging from a plain answer parses ok only once the call is complete."""
        prompt: list[Message] = [Message(role="user", content="weather?")]
        call = ToolCall(id="call_1", function=ToolCallFunction(name="get_weather", arguments='{"location": "Paris"}'))
        with_call = _extract_response_tokens(codec, prompt, Message(role="assistant", tool_calls=[call]))
        plain = _extract_response_tokens(codec, prompt, Message(role="assistant", content=""))
        assert codec.parse_response(with_call)[1]
        diverge = next((i for i, (a, b) in enumerate(zip(with_call, plain)) if a != b), len(plain))
        for cut in range(diverge + 1, len(with_call) + 1):
            msg, ok = codec.parse_response(with_call[:cut])
            calls = [
                (c["function"]["name"], json.loads(c["function"]["arguments"])) for c in msg.get("tool_calls", [])
            ]
            assert not ok or calls == [("get_weather", {"location": "Paris"})], codec.decode(with_call[:cut])


class TestToolArgTypePreservation:
    """Codec contract: parse_response returns tool-call arguments whose Python types
    match the tool's declared JSON-schema types, regardless of wire format (Qwen3.5
    XML renders values unquoted and recovers types from the passed schema). Numeric
    strings such as zip codes must remain strings when the schema declares them so."""

    _TOOL = ToolSpec(
        name="lookup_user",
        description="look up a user record",
        parameters={
            "type": "object",
            "properties": {
                "zip": {"type": "string"},  # numeric-looking STRING — the bug case
                "count": {"type": "integer"},
                "ratio": {"type": "number"},
                "flag": {"type": "boolean"},
                "opt": {"type": "null"},
                "tags": {"type": "array"},
                "meta": {"type": "object"},
                # Pydantic's Optional[int] / Optional[str] shapes.
                "maybe_count": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                "maybe_zip": {"type": ["string", "null"]},
            },
        },
    )

    def test_arg_types_match_declared_schema(self, codec: MessageCodec):
        """Full-template round-trip: scalars, bool, null, and nested array/object
        recover their declared Python types. The bool/null cases guard fix #1 —
        the template emits capitalized `True`/`False`/`None`, not JSON literals."""
        prompt: list[Message] = [Message(role="user", content="look me up")]
        original_args = {
            "zip": "32286",
            "count": 3,
            "ratio": 0.5,
            "flag": True,
            "opt": None,
            "tags": ["a", "b"],
            "meta": {"k": 1},
            "maybe_count": None,
            "maybe_zip": "32286",
        }
        assistant_msg = Message(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    function=ToolCallFunction(
                        name="lookup_user",
                        arguments=json.dumps(original_args),
                    ),
                )
            ],
        )
        response_tokens = _extract_response_tokens(codec, prompt, assistant_msg)
        msg, ok = codec.parse_response(response_tokens, tools=[self._TOOL])
        assert ok
        args = json.loads(msg["tool_calls"][0]["function"]["arguments"])
        assert isinstance(args["zip"], str) and args["zip"] == "32286"  # not int 32286
        assert isinstance(args["count"], int) and args["count"] == 3
        assert isinstance(args["ratio"], float) and args["ratio"] == 0.5
        assert args["flag"] is True
        # MiniMax-M3's template omits a null argument rather than rendering it.
        assert args.get("opt") is None
        assert args["tags"] == ["a", "b"]
        assert args["meta"] == {"k": 1}
        assert args.get("maybe_count") is None  # not the string "null"
        assert isinstance(args["maybe_zip"], str) and args["maybe_zip"] == "32286"

    def test_unknown_param_round_trips_through_template(self, codec: MessageCodec):
        """A param the model emits that isn't in the declared schema survives the
        full-template round-trip via best-effort coercion (full-template coverage
        for the unit-level fallback case)."""
        prompt: list[Message] = [Message(role="user", content="look me up")]
        assistant_msg = Message(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    function=ToolCallFunction(
                        name="lookup_user",
                        arguments=json.dumps({"surprise": "hi"}),
                    ),
                )
            ],
        )
        response_tokens = _extract_response_tokens(codec, prompt, assistant_msg)
        msg, ok = codec.parse_response(response_tokens, tools=[self._TOOL])
        assert ok
        assert json.loads(msg["tool_calls"][0]["function"]["arguments"])["surprise"] == "hi"

    def test_tool_call_only_response_has_no_content_key(self, codec: MessageCodec):
        """A pure tool-call turn must not emit a stray empty `content` key."""
        prompt: list[Message] = [Message(role="user", content="look me up")]
        assistant_msg = Message(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    function=ToolCallFunction(
                        name="lookup_user",
                        arguments=json.dumps({"zip": "32286"}),
                    ),
                )
            ],
        )
        response_tokens = _extract_response_tokens(codec, prompt, assistant_msg)
        msg, ok = codec.parse_response(response_tokens, tools=[self._TOOL])
        assert ok
        assert "content" not in msg
        assert len(msg.get("tool_calls", [])) == 1


class TestPreserveThinking:
    _MESSAGES: list[Message] = [
        Message(role="user", content="solve this"),
        Message(role="assistant", reasoning_content="careful chain of thought", content="answer one"),
        Message(role="user", content="follow up"),
    ]

    @pytest.mark.parametrize("enable_thinking", [None, True])
    @pytest.mark.parametrize("model", _MODELS)
    def test_without_preserve_history_follows_the_template(self, model: str, enable_thinking: bool | None):
        codec = get_codec(model, enable_thinking=enable_thinking)
        rendered = codec.decode(codec.encode_messages(self._MESSAGES, add_generation_prompt=True))
        assert ("careful chain of thought" in rendered) is _KEEPS_HISTORY_REASONING[model]

    def test_preserve_thinking_keeps_history_reasoning(self, codec_preserve_thinking: MessageCodec):
        rendered = codec_preserve_thinking.decode(
            codec_preserve_thinking.encode_messages(self._MESSAGES, add_generation_prompt=True)
        )
        assert "careful chain of thought" in rendered

    @pytest.mark.parametrize("model", _MODELS)
    def test_preserve_false_strips_history_reasoning(self, model: str):
        """Reasoning before the last user message is dropped; the tool loop answering it keeps its own."""
        codec = get_codec(model, preserve_thinking=False)
        call = ToolCall(id="call_1", function=ToolCallFunction(name="lookup", arguments='{"q": "x"}'))
        messages = self._MESSAGES + [
            Message(role="assistant", reasoning_content="loop reasoning", tool_calls=[call]),
            Message(role="tool", content="sunny", tool_call_id="call_1", name="lookup"),
            Message(role="assistant", reasoning_content="final reasoning", content="two"),
        ]
        rendered = codec.decode(codec.encode_messages(messages))
        assert "careful chain of thought" not in rendered
        assert "loop reasoning" in rendered
        assert "final reasoning" in rendered

    def test_history_render_is_prefix_of_next_turn(self, codec_preserve_thinking: MessageCodec):
        """`flatten_episode` keeps an episode in one sample only when each turn's
        gen-prompt render extends the previous turn's. Without `enable_thinking=True`
        the closed empty `<think></think>` in turn N's render gets rewritten on turn
        N+1, breaking the prefix invariant and fragmenting the episode."""
        messages: list[Message] = [
            Message(role="system", content="be brief"),
            Message(role="user", content="q1"),
            Message(role="assistant", reasoning_content="reasoning one", content="a1"),
            Message(role="user", content="q2"),
            Message(
                role="assistant",
                reasoning_content="reasoning two",
                tool_calls=[ToolCall(id="call_1", function=ToolCallFunction(name="lookup", arguments='{"q": "x"}'))],
            ),
            Message(role="tool", content="sunny", tool_call_id="call_1", name="lookup"),
            Message(role="assistant", reasoning_content="reasoning three", content="a2"),
            Message(role="user", content="q3"),
        ]
        renders = [
            codec_preserve_thinking.encode_messages(messages[:i], add_generation_prompt=True)
            for i, m in enumerate(messages)
            if m["role"] == "assistant"
        ]
        for prev, curr in zip(renders, renders[1:]):
            assert curr[: len(prev)] == prev

    def test_no_close_think_leaks_into_rendered_content(self, codec_preserve_thinking: MessageCodec):
        """Re-rendering a multi-turn history with preserved reasoning must wrap
        each turn's thinking in `<think>…</think>` and
        leave no stray `</think>` inside the rendered answer content."""
        messages: list[Message] = [
            Message(role="user", content="q1"),
            Message(role="assistant", reasoning_content="reasoning one", content="answer one"),
            Message(role="user", content="q2"),
            Message(role="assistant", reasoning_content="reasoning two", content="answer two"),
            Message(role="user", content="q3"),
        ]
        rendered = codec_preserve_thinking.decode(
            codec_preserve_thinking.encode_messages(messages, add_generation_prompt=True)
        )
        # No orphan `</think>` floating in content: at every prefix, closers never
        # outnumber openers (the trailing gen-prompt opener may be unmatched, which
        # is fine — that's the primed next turn).
        opened = 0
        for token in re.findall(r"<think>|</think>", rendered):
            opened += 1 if token == "<think>" else -1
            assert opened >= 0, "stray </think> leaked into rendered content"


def _gen_prompt(model: str, **kwargs) -> list[int]:
    return get_codec(model, **kwargs).encode_messages([Message(role="user", content="q")], add_generation_prompt=True)


@pytest.mark.parametrize("model", _MODELS)
class TestThinkingFlagCombos:
    """`enable_thinking` controls the gen prompt; `preserve_thinking` controls
    re-rendering of past turns. They're independent except that preserve requires
    enable (nothing to preserve otherwise), and GLM-5 cannot turn thinking off at all."""

    def test_preserve_implies_enable(self, model: str):
        """preserve_thinking only changes history, so its gen prompt is the thinking one."""
        assert _gen_prompt(model, preserve_thinking=True) == _gen_prompt(model, enable_thinking=True)

    def test_preserve_with_enable_false_rejected(self, model: str):
        with pytest.raises(ValueError, match="preserve_thinking"):
            get_codec(model, enable_thinking=False, preserve_thinking=True)


@pytest.mark.parametrize("model", THINKING_OFF_MODELS)
def test_instruct_mode_changes_the_gen_prompt(model: str):
    assert _gen_prompt(model, enable_thinking=False) != _gen_prompt(model, enable_thinking=True)


# Tokens that stop generation, per family. GLM-5.3's generation_config.json declares
# three — `<|endoftext|>` ends the conversation, `<|user|>` yields back to the user,
# `<|observation|>` hands off to a tool — where Qwen declares one universal `<|im_end|>`.
_STOP_TOKENS: dict[str, tuple[str, ...]] = {
    REFERENCE_MODELS["qwen3.5"]: ("<|im_end|>",),
    _GLM5: ("<|endoftext|>", "<|user|>", "<|observation|>"),
    # Kimi-K3 ends every message, assistant turns included, with `<|end_of_msg|>`.
    REFERENCE_MODELS["kimi-k3"]: ("<|end_of_msg|>",),
    REFERENCE_MODELS["minimax-m3"]: ("[e~[",),
}


@pytest.mark.parametrize("model", _MODELS)
def test_every_trained_turn_ends_on_a_stop_token(model: str):
    """SFT and DPO train exactly the tokens an assistant turn adds, so a trained turn with
    no stop token teaches the model to generate until the token budget runs out. GLM needs
    `turn_terminator` for this: its template closes a turn with the *next* message's role
    marker, so nothing inside the turn stops it."""
    codec = get_codec(model)
    stop_ids = {single_token(codec, token) for token in _STOP_TOKENS[model]}
    runs = trained_runs(
        render_conversation(TWO_TURN_CONVERSATION, codec=codec, train_on_what=TRAIN_ON_ALL, max_length=4096)
    )
    assert len(runs) == 2
    for run in runs:
        # Qwen's template writes a newline after `<|im_end|>`, so the stop token closes the
        # turn without always being the very last token.
        assert stop_ids & set(run[-2:]), f"trained turn does not end on a stop token: {run[-4:]}"


# The effort levels each family's template renders, and the one it renders by default. Pinned so a
# template bump cannot silently change how much of the token budget every rollout spends.
_THINKING_EFFORTS: dict[str, tuple[tuple[str, ...], str | None]] = {
    REFERENCE_MODELS["qwen3.5"]: ((), None),
    _GLM5: (("low", "high", "max"), "max"),
    REFERENCE_MODELS["kimi-k3"]: (("low", "high", "max"), "max"),
    REFERENCE_MODELS["minimax-m3"]: ((), None),
}


@pytest.mark.parametrize("model", _MODELS)
def test_thinking_effort(model: str):
    """Each supported level renders distinctly, `None` keeps the template default, and a level the
    template cannot render is rejected at construction."""
    supported, default = _THINKING_EFFORTS[model]
    for level in {"low", "medium", "high", "max"} - set(supported):
        with pytest.raises(ValueError, match="thinking"):
            get_codec(model, thinking_effort=level)
    renders = {level: _gen_prompt(model, thinking_effort=level) for level in supported}
    assert len({tuple(render) for render in renders.values()}) == len(supported)
    if default is not None:
        assert _gen_prompt(model) == renders[default]


@pytest.mark.parametrize("model", THINKING_OFF_MODELS)
def test_thinking_off_ignores_effort(model: str):
    for level in _THINKING_EFFORTS[model][0]:
        assert _gen_prompt(model, enable_thinking=False, thinking_effort=level) == _gen_prompt(
            model, enable_thinking=False
        )


def test_non_json_tool_call_arguments_warn_and_encode_empty():
    """Canonical arguments are a JSON string; anything else is dropped with a warning
    rather than raising mid-render."""
    messages: list[Message] = [
        Message(
            role="assistant",
            tool_calls=[ToolCall(id=None, function=ToolCallFunction(name="f", arguments="not json"))],
        )
    ]
    with pytest.warns(UserWarning, match="non-JSON arguments"):
        adapted = messages_for_template(messages)
    assert adapted[0]["tool_calls"][0]["function"]["arguments"] == {}


class TestCodecRouting:
    """Longest prefix wins, and unsupported families are refused rather than
    falling through to another family's parser."""

    @pytest.mark.parametrize(
        "model, expected",
        [
            ("Qwen/Qwen3.5-35B-A3B", Qwen35Codec),
            ("Qwen/Qwen3.6-35B-A3B", Qwen35Codec),
            ("Qwen/Qwen3.8-27B", Qwen35Codec),
            (_GLM5, Glm5Codec),
            ("moonshotai/Kimi-K3", KimiK3Codec),
            ("moonshotai/Kimi-K3-262k", KimiK3Codec),
            ("MiniMaxAI/MiniMax-M3", MiniMaxM3Codec),
            ("MiniMaxAI/MiniMax-M3-MXFP8", MiniMaxM3Codec),
        ],
    )
    def test_resolves_expected_codec(self, model: str, expected: type[MessageCodec]):
        assert resolve_codec_class(model) is expected

    @pytest.mark.parametrize(
        "model",
        [
            "meta-llama/Llama-3-8B",
            # GLM-4 uses a different tool-call body grammar, and GLM-5.2 a template Glm5Codec
            # does not model, so neither may route to it.
            "zai-org/GLM-4.6",
            "zai-org/GLM-5.2",
        ],
    )
    def test_unknown_model_rejected(self, model: str):
        with pytest.raises(ValueError, match="No codec"):
            resolve_codec_class(model)

    @pytest.mark.parametrize(
        "model, tool_call",
        [
            ("Qwen/Qwen3.5-0.8B", "<tool_call>\n<function=lookup>\n</function>\n</tool_call>"),
            ("Qwen/Qwen3.8-27B", "<tool_call>\n<function=lookup>\n</function>\n</tool_call>"),
            ("zai-org/GLM-5.3", "<tool_call>lookup</tool_call>"),
        ],
        ids=["qwen3.5", "qwen3.8", "glm-5"],
    )
    @pytest.mark.parametrize("channel", ["content", "reasoning", "closed-think", "truncated"])
    def test_parse_completion_matches_full_response(self, model: str, tool_call: str, channel: str):
        if channel == "content":
            full_text = tool_call
            content, reasoning = tool_call, None
        elif channel == "reasoning":
            full_text = tool_call
            content, reasoning = "", tool_call
        elif channel == "closed-think":
            full_text = f"<think>{tool_call}</think>answer"
            content, reasoning = "answer", tool_call
        else:
            full_text = tool_call.rsplit("</tool_call>", 1)[0]
            content, reasoning = "", full_text

        # The Qwen codecs re-join the channels. GLM-5's gen prompt always opens a think block, so a
        # content-free completion may never have left it and must parse as training parses the sampled turn.
        codec = get_codec(model)
        parses_as_sampled = isinstance(codec, Glm5Codec) and not content
        expected = codec.parse_response(codec.encode(full_text)) if parses_as_sampled else codec.parse_text(full_text)
        actual = parse_completion(model, content=content, reasoning=reasoning)
        assert actual == expected
        if channel == "truncated":
            assert actual[1] is parses_as_sampled
            assert "tool_calls" not in actual[0]
