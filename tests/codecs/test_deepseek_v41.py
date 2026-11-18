"""DeepSeek-V4.1 codec contracts."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import pytest
from huggingface_hub import hf_hub_download
from jinja2 import TemplateError

from post_training_cookbook.codecs import DeepSeekV41Codec, get_codec, resolve_codec_class
from post_training_cookbook.codecs.deepseek_v41 import (
    DEEPSEEK_V41_REPO,
    DEEPSEEK_V41_REVISION,
)
from post_training_cookbook.types import Message, ToolCall, ToolCallFunction, ToolSpec
from tests.codecs import REFERENCE_MODELS

_MODEL = REFERENCE_MODELS["deepseek-v4.1-flash"]
_EOS = 1

_TOKENIZER_JSON_SHA256 = "c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b"

_LOOKUP = ToolSpec(
    name="lookup",
    description="look up a record",
    parameters={
        "type": "object",
        "properties": {
            "zip": {"type": "string"},
            "count": {"type": "integer"},
            "ratio": {"type": "number"},
            "flag": {"type": "boolean"},
            "tags": {"type": "array"},
        },
    },
)


def _call(call_id: str, name: str, arguments: dict) -> ToolCall:
    return ToolCall(id=call_id, function=ToolCallFunction(name=name, arguments=json.dumps(arguments)))


@lru_cache
def _release_tokenizer():
    return DeepSeekV41Codec(_MODEL)._tokenizer


def _codec(effort: str | int | None = None, **kwargs) -> DeepSeekV41Codec:
    return DeepSeekV41Codec(
        _MODEL,
        tokenizer=_release_tokenizer(),
        thinking_effort=str(effort) if effort is not None else None,
        **kwargs,
    )


@pytest.fixture(scope="module")
def codec() -> DeepSeekV41Codec:
    return _codec()


@pytest.fixture(scope="module")
def tokenizer(codec: DeepSeekV41Codec):
    return codec._tokenizer


def _sampled(codec: DeepSeekV41Codec, prompt: list[Message], turn: Message, tools: list[ToolSpec] | None) -> list[int]:
    """The tokens a turn adds to its prompt, as the model would sample them (EOS last)."""
    prompt_tokens = codec.encode_messages(prompt, tools=tools, add_generation_prompt=True)
    full = codec.encode_messages([*prompt, turn], tools=tools)
    assert full[: len(prompt_tokens)] == prompt_tokens, "history render is not prefix-stable"
    return full[len(prompt_tokens) :]


class TestReleaseFiles:
    def test_tokenizer_is_the_pinned_release_file(self):
        path = hf_hub_download(DEEPSEEK_V41_REPO, "tokenizer.json", revision=DEEPSEEK_V41_REVISION)
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == _TOKENIZER_JSON_SHA256


def _param(key: str, value: str, string: str = "true") -> str:
    return f'<｜DSML｜ parameter name="{key}" string="{string}">{value}</｜DSML｜ parameter>'


def _invoke(name: str, *params: str) -> str:
    return f'<｜DSML｜ invoke name="{name}">\n' + "\n".join(params) + "\n</｜DSML｜ invoke>"


def _calls(*invokes: str) -> str:
    """A calls block as the encoder writes it, with the blank line before it."""
    return "\n\n<｜DSML｜ calls>\n" + "\n".join(invokes) + "\n</｜DSML｜ calls>"


class TestParse:
    @pytest.mark.parametrize("reasoning", ["plan", _calls(_invoke("lookup"))])
    def test_endpoint_channels(self, reasoning: str):
        content = "done" + _calls(_invoke("lookup", _param("zip", "0")))
        msg, ok = DeepSeekV41Codec.parse_completion(content=content, reasoning=reasoning, tools=[_LOOKUP])
        assert ok
        assert msg["reasoning_content"] == reasoning
        assert msg["content"] == "done"
        assert len(msg["tool_calls"]) == 1
        msg, ok = DeepSeekV41Codec.parse_completion(content=None, reasoning=reasoning, tools=[_LOOKUP])
        assert ok
        assert msg == Message(role="assistant", reasoning_content=reasoning)

    @pytest.mark.parametrize("raw", ["1e999", "NaN", "[1e999]", '{"x": NaN}'])
    def test_nonfinite_arguments_stay_text(self, raw: str):
        msg, ok = DeepSeekV41Codec.parse_completion(
            content=_calls(_invoke("lookup", _param("ratio", raw, "false"))), reasoning=None, tools=[_LOOKUP]
        )
        assert ok
        arguments = json.loads(msg["tool_calls"][0]["function"]["arguments"])
        assert arguments == {"ratio": raw}
        json.dumps(arguments, allow_nan=False)

    @pytest.mark.parametrize("raw", ['{"ratio": 1e999}', '{"ratio": [NaN]}'])
    def test_nonfinite_json_body_fails(self, raw: str):
        msg, ok = DeepSeekV41Codec.parse_completion(content=_calls(_invoke("lookup", raw)), reasoning=None)
        assert not ok
        assert "tool_calls" not in msg

    def test_nullable_string_coercion(self):
        tools = [ToolSpec(name="lookup", parameters={"properties": {"zip": {"type": ["string", "null"]}}})]
        msg, ok = DeepSeekV41Codec.parse_completion(
            content=_calls(_invoke("lookup", _param("zip", "32286", "false"))), reasoning=None, tools=tools
        )
        assert ok
        assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"zip": "32286"}

    def test_stop_and_cut_parse_the_same(self, codec: DeepSeekV41Codec):
        tokens = codec.encode("r</think>c" + _calls(_invoke("lookup", _param("zip", "0"))))
        assert codec.parse_response(tokens + [_EOS]) == codec.parse_response(tokens)
        msg, ok = codec.parse_response(tokens, tools=[_LOOKUP])
        assert ok
        assert msg == Message(
            role="assistant", content="c", reasoning_content="r", tool_calls=[_call("call_1", "lookup", {"zip": "0"})]
        )

    def test_turn_cut_inside_reasoning_stays_reasoning(self, codec: DeepSeekV41Codec):
        assert codec.parse_response(codec.encode("still thinking about")) == (
            Message(role="assistant", reasoning_content="still thinking about"),
            True,
        )

    def test_calls_before_closing_reasoning_fail(self, codec: DeepSeekV41Codec):
        _, ok = codec.parse_response(codec.encode("plan" + _calls(_invoke("lookup"))))
        assert not ok

    def test_whitespace_is_kept_verbatim(self):
        """Re-rendering needs the exact text: a lone newline before the block stays in the content."""
        msg, ok = DeepSeekV41Codec.parse_text("\n\nr\n</think>\n" + _calls(_invoke("lookup")))
        assert ok
        assert msg["reasoning_content"] == "\n\nr\n"
        assert msg["content"] == "\n"

    @pytest.mark.parametrize(
        "value",
        ['say "hi"\nthen <b> & </b>', "café ≥ 110 %", "  padded  ", "", "{not json", "H-4821"],
    )
    def test_string_values_round_trip(self, codec: DeepSeekV41Codec, value: str):
        prompt = [Message(role="user", content="q")]
        turn = Message(role="assistant", reasoning_content="r", tool_calls=[_call("call_1", "lookup", {"zip": value})])
        sampled = _sampled(codec, prompt, turn, [_LOOKUP])
        parsed, ok = codec.parse_response(sampled, tools=[_LOOKUP])
        assert ok
        assert json.loads(parsed["tool_calls"][0]["function"]["arguments"]) == {"zip": value}
        assert _sampled(codec, prompt, parsed, [_LOOKUP]) == sampled

    def test_json_values_round_trip(self, codec: DeepSeekV41Codec):
        prompt = [Message(role="user", content="q")]
        arguments = {"count": 3, "ratio": 2.5, "flag": False, "tags": ["a", {"b": None}], "extra": 7}
        turn = Message(role="assistant", reasoning_content="r", tool_calls=[_call("call_1", "lookup", arguments)])
        sampled = _sampled(codec, prompt, turn, [_LOOKUP])
        parsed, ok = codec.parse_response(sampled, tools=[_LOOKUP])
        assert ok
        assert json.loads(parsed["tool_calls"][0]["function"]["arguments"]) == arguments
        assert _sampled(codec, prompt, parsed, [_LOOKUP]) == sampled

    @pytest.mark.parametrize(
        "parameter, expected",
        [
            # Mistyped on the wire: coerced to the declared type.
            (_param("count", "500"), {"count": 500}),
            (_param("zip", "32286", "false"), {"zip": "32286"}),
            (_param("flag", "true"), {"flag": True}),
            # Already well typed: kept as written (an int is a JSON number).
            (_param("ratio", "5", "false"), {"ratio": 5}),
            # Not coercible: the wire value stands, for the env to reject.
            (_param("count", "5.5", "false"), {"count": 5.5}),
            (_param("count", "many"), {"count": "many"}),
            # Undeclared: the wire type; non-JSON and NaN stay text.
            (_param("other", "[1,2]", "false"), {"other": [1, 2]}),
            (_param("other", " raw text ", "false"), {"other": "raw text"}),
            (_param("ratio", "NaN", "false"), {"ratio": "NaN"}),
        ],
    )
    def test_argument_types(self, parameter: str, expected: dict):
        msg, ok = DeepSeekV41Codec.parse_text("</think>" + _calls(_invoke("lookup", parameter)), tools=[_LOOKUP])
        assert ok
        assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == expected

    @pytest.mark.parametrize(
        "invoke",
        [
            # The serving parser's tolerances: spacing inside tags, a self-closing invoke, a JSON body, prose.
            '<｜DSML｜ invoke  name="lookup" >\n<｜DSML｜ parameter name="zip"  string="true" >1</｜DSML｜ parameter>'
            "\n</｜DSML｜ invoke>",
            '<｜DSML｜ invoke name="lookup"/>',
            _invoke("lookup", '{"zip": "1"}'),
            _invoke("lookup", "note " + _param("zip", "1")),
        ],
    )
    def test_tolerated_layouts(self, invoke: str):
        # A single newline before the block is tolerated too; it stays in the content.
        text = "</think>" + _calls(invoke).removeprefix("\n")
        msg, ok = DeepSeekV41Codec.parse_text(text, tools=[_LOOKUP])
        assert ok
        assert [call["function"]["name"] for call in msg["tool_calls"]] == ["lookup"]

    @pytest.mark.parametrize(
        "rest",
        [
            # V4-style unspaced tags, and a bare invoke with no calls block.
            '\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name="lookup">\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>',
            _invoke("lookup"),
            # A block that never closes, or an invoke that never closes.
            _calls(_invoke("lookup")).removesuffix("\n</｜DSML｜ calls>"),
            _calls('<｜DSML｜ invoke name="lookup">'),
            # Text after the block; an empty block.
            _calls(_invoke("lookup")) + "more",
            "\n\n<｜DSML｜ calls>\n</｜DSML｜ calls>",
            # A malformed parameter, a repeated parameter, an empty name, a doubly namespaced name.
            _calls(_invoke("lookup", '<｜DSML｜ parameter name="zip">1</｜DSML｜ parameter>')),
            _calls(_invoke("lookup", _param("zip", "1"), _param("zip", "2"))),
            _calls(_invoke("")),
            _calls(_invoke("a::b::c")),
        ],
    )
    def test_malformed_calls_fail(self, codec: DeepSeekV41Codec, rest: str):
        _, ok = codec.parse_response(codec.encode("r</think>" + rest), tools=[_LOOKUP])
        assert not ok

    def test_parse_text_reads_the_endpoint_fence(self):
        expected = Message(role="assistant", content="c", reasoning_content="r")
        assert DeepSeekV41Codec.parse_text("<think>r</think>c") == (expected, True)
        assert DeepSeekV41Codec.parse_text("r</think>c") == (expected, True)
        assert DeepSeekV41Codec.parse_text("c") == (Message(role="assistant", content="c"), True)

    def test_chat_mode_reads_everything_as_content(self):
        codec = _codec(enable_thinking=False)
        assert codec.parse_response(codec.encode("a </think> b")) == (
            Message(role="assistant", content="a </think> b"),
            True,
        )


class TestPrompt:
    _MESSAGES = [Message(role="system", content="s"), Message(role="user", content="q")]

    @pytest.mark.parametrize("preserve", [None, True, False])
    def test_tools_and_history_reasoning(self, preserve: bool | None):
        codec = _codec(preserve_thinking=preserve)
        messages = [
            Message(role="user", content="first"),
            Message(role="assistant", reasoning_content="old reasoning", content="answer"),
            Message(role="user", content="next"),
            Message(
                role="assistant", reasoning_content="current reasoning", tool_calls=[_call("call_1", "lookup", {})]
            ),
            Message(role="tool", content="ok", tool_call_id="call_1"),
        ]
        text = codec.render(messages, tools=[_LOOKUP])
        assert ("old reasoning" in text) is (preserve is not False)
        assert "current reasoning" in text

    @pytest.mark.parametrize(
        "effort, budget",
        [(None, 75), ("low", 50), ("high", 75), ("max", 100), ("1", 1), ("88", 88), ("100", 100)],
    )
    def test_reasoning_effort(self, effort: str | None, budget: int):
        codec = _codec(effort)
        assert codec.reasoning_budget == budget
        assert codec.render(self._MESSAGES) == (
            f"<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: {budget} (range 1-100, the higher the value, "
            "the more thorough the reasoning)\n\ns<｜User｜>q<｜Assistant｜><think>"
        )

    @pytest.mark.parametrize("effort", ["0", "101", "medium", "xhigh", "none", "minimal", "True", "1.5", " 75", "-5"])
    def test_rejects_efforts_the_encoder_rejects(self, effort: str):
        with pytest.raises(ValueError, match="thinking_effort"):
            get_codec(_MODEL, thinking_effort=effort)

    def test_chat_mode(self):
        codec = _codec("max", enable_thinking=False)
        assert codec.render(self._MESSAGES) == "<｜begin▁of▁sentence｜><｜System｜>s<｜User｜>q<｜Assistant｜></think>"
        assert codec.render(self._MESSAGES[1:]) == "<｜begin▁of▁sentence｜><｜User｜>q<｜Assistant｜></think>"

    def test_tools_attach_to_the_system_message_or_an_empty_one(self, codec: DeepSeekV41Codec):
        with_system = codec.render(self._MESSAGES, tools=[_LOOKUP])
        assert "the more thorough the reasoning)\n\ns\n\n## Tools\n\n" in with_system
        without = codec.render(self._MESSAGES[1:], tools=[_LOOKUP])
        assert "the more thorough the reasoning)\n\n\n\n## Tools\n\n" in without
        assert json.dumps(dict(_LOOKUP), ensure_ascii=False) in without

    def test_tool_results_follow_call_order(self, codec: DeepSeekV41Codec):
        calls = [_call("call_1", "lookup", {"zip": "1"}), _call("call_2", "lookup", {"zip": "2"})]
        messages = [
            Message(role="user", content="q"),
            Message(role="assistant", reasoning_content="r", tool_calls=calls),
            Message(role="tool", content="second", tool_call_id="call_2"),
            Message(role="tool", content="first", tool_call_id="call_1"),
        ]
        rendered = codec.render(messages, tools=[_LOOKUP])
        assert rendered.endswith(
            "<｜User｜><tool_result>first</tool_result>\n\n<tool_result>second</tool_result><｜Assistant｜><think>"
        )

    def test_generation_prompt_flag(self, codec: DeepSeekV41Codec):
        messages = [Message(role="user", content="q")]
        assert codec.decode(codec.encode_messages(messages, add_generation_prompt=True)).endswith(
            "<｜Assistant｜><think>"
        )
        assert not codec.decode(codec.encode_messages(messages)).endswith("<｜Assistant｜><think>")


class TestTokenizer:
    @pytest.mark.parametrize(
        "marker, token_id",
        [
            ("<｜begin▁of▁sentence｜>", 0),
            ("<｜end▁of▁sentence｜>", 1),
            ("<｜System｜>", 128799),
            ("<｜User｜>", 128803),
            ("<｜Assistant｜>", 128804),
            ("<think>", 128821),
            ("</think>", 128822),
            ("｜DSML｜", 128825),
        ],
    )
    def test_markers(self, codec: DeepSeekV41Codec, marker: str, token_id: int):
        assert codec.encode(marker) == [token_id]
        # Only BOS and EOS are special; the rest survive decoding, so the parser sees them.
        assert codec.decode([token_id]) == ("" if token_id in (0, 1) else marker)

    def test_no_bos_on_encode_and_spaces_survive_decode(self, codec: DeepSeekV41Codec, tokenizer):
        """transformers 5.3.0 and 5.6.x strip spaces when decoding some DeepSeek tokenizers (#44779); not this one."""
        assert (
            tokenizer.encode("How are you doing?") == codec.encode("How are you doing?") == [4117, 477, 440, 4843, 33]
        )
        assert codec.decode(codec.encode("How are you doing?")) == "How are you doing?"

    def test_rejects_another_tokenizer(self):
        class Other:
            def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
                return [7]

        with pytest.raises(ValueError, match="not a DeepSeek-V4.1 tokenizer"):
            DeepSeekV41Codec(tokenizer=Other())


class TestReleaseChatTemplate:
    """The release template agrees on tool loops and differs on effort aliases and argument types."""

    @staticmethod
    def _dict_arguments(messages: list[dict]) -> list[dict]:
        out = json.loads(json.dumps(messages))
        for msg in out:
            for call in msg.get("tool_calls") or []:
                call["function"]["arguments"] = json.loads(call["function"]["arguments"])
        return out

    def test_agrees_on_a_tool_loop(self, codec: DeepSeekV41Codec, tokenizer):
        messages = [
            Message(role="user", content="q"),
            Message(role="assistant", reasoning_content="r", tool_calls=[_call("call_1", "lookup", {"zip": "1"})]),
            Message(role="tool", content="ok", tool_call_id="call_1"),
        ]
        jinja = tokenizer.apply_chat_template(
            self._dict_arguments(messages),
            tools=[{"type": "function", "function": _LOOKUP}],
            add_generation_prompt=True,
            tokenize=False,
        )
        assert jinja == codec.render(messages, tools=[_LOOKUP])

    def test_deltas(self, codec: DeepSeekV41Codec, tokenizer):
        messages = [Message(role="user", content="q")]
        # Effort names only the jinja knows.
        assert "Reasoning Effort: 75 " in tokenizer.apply_chat_template(
            messages, reasoning_effort="medium", add_generation_prompt=True, tokenize=False
        )
        with pytest.raises(ValueError):
            get_codec(_MODEL, thinking_effort="medium")
        # JSON-string arguments: the jinja raises, the encoder parses them.
        turn = [*messages, Message(role="assistant", tool_calls=[_call("call_1", "lookup", {"zip": "1"})])]
        with pytest.raises(TemplateError, match="must be a dict"):
            tokenizer.apply_chat_template(turn, tokenize=False)
        assert 'string="true">1</｜DSML｜ parameter>' in codec.render(turn)
        # add_generation_prompt=False after a user turn: only the jinja drops the header.
        assert not tokenizer.apply_chat_template(messages, tokenize=False).endswith("<｜Assistant｜><think>")
        assert codec.render(messages).endswith("<｜Assistant｜><think>")


class TestRouting:
    @pytest.mark.parametrize(
        "model",
        [
            _MODEL,
            "deepseek-ai/DeepSeek-V4.1-Flash",
            "togethercomputer/DeepSeek-V4.1-Flash-1017",
            "DeepSeek-V4.1-Flash-1017-abl-x",
        ],
    )
    def test_v41_flash_names_route_here(self, model: str):
        assert resolve_codec_class(model) is DeepSeekV41Codec

    @pytest.mark.parametrize(
        "model",
        ["deepseek-ai/DeepSeek-V4-Flash", "deepseek-ai/DeepSeek-V4-Flash-0731", "deepseek-ai/DeepSeek-V4.1-Pro"],
    )
    def test_other_deepseek_encoders_do_not(self, model: str):
        with pytest.raises(ValueError, match="No codec"):
            resolve_codec_class(model)
