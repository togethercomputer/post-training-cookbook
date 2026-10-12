"""Qwen3.5-family codec: the XML tool-call body parser, tested directly on wire text, and the
Qwen3.6/Qwen3.8 templates that take preserve_thinking natively."""

from __future__ import annotations

import json
import re
import warnings

import pytest
from transformers import AutoTokenizer

from post_training_cookbook.codecs import PreserveThinkingForkedWarning, Qwen35Codec, get_codec
from post_training_cookbook.codecs.base import MessageCodec
from post_training_cookbook.codecs.qwen3_5 import _chat_template_reads_preserve_thinking, _parse_tool_body_xml
from post_training_cookbook.codecs.schema_coercion import param_types_by_tool
from post_training_cookbook.types import Message, ToolSpec
from tests.codecs import ARG_SCHEMA

_QWEN36 = "Qwen/Qwen3.6-27B"
_QWEN38 = "Qwen/Qwen3.8-27B"


class TestQwen35XmlArgParsing:
    """Direct, format-specific tests of the Qwen3.5 XML param parser (deterministic,
    not subject to template rendering)."""

    def test_declared_types_respected(self):
        """The template renders Python scalars via Jinja `| string`, so it emits the
        CAPITALIZED `True`/`False`/`None` — not JSON `true`/`false`/`null`. Coercion
        keys off the declared schema type, not the literal's casing."""
        raw = (
            "<function=lookup>\n"
            "<parameter=zip>\n32286\n</parameter>\n"
            "<parameter=count>\n3\n</parameter>\n"
            "<parameter=flag>\nTrue\n</parameter>\n"
            "<parameter=opt>\nNone\n</parameter>\n"
            "<parameter=note>\nhello\n</parameter>\n"
            "</function>"
        )
        tc, ok = _parse_tool_body_xml(raw, ARG_SCHEMA)
        assert ok
        args = json.loads(tc["function"]["arguments"])
        assert args["zip"] == "32286" and isinstance(args["zip"], str)
        assert args["count"] == 3 and isinstance(args["count"], int)
        assert args["flag"] is True
        assert args["opt"] is None
        assert args["note"] == "hello"

    def test_boolean_false_and_unrecognized_fall_back(self):
        raw = "<function=lookup>\n<parameter=flag>\nFalse\n</parameter>\n</function>"
        tc, _ = _parse_tool_body_xml(raw, ARG_SCHEMA)
        assert json.loads(tc["function"]["arguments"])["flag"] is False

        raw_bad = "<function=lookup>\n<parameter=flag>\nmaybe\n</parameter>\n</function>"
        tc, _ = _parse_tool_body_xml(raw_bad, ARG_SCHEMA)
        assert json.loads(tc["function"]["arguments"])["flag"] == "maybe"

    def test_whitespace_significant_string_preserved_verbatim(self):
        """A string-typed value keeps interior blank lines and leading indentation;
        only the single template framing newline is peeled (fix #2)."""
        body = "  indented line\n\nblank above"
        raw = f"<function=lookup>\n<parameter=note>\n{body}\n</parameter>\n</function>"
        tc, ok = _parse_tool_body_xml(raw, ARG_SCHEMA)
        assert ok
        assert json.loads(tc["function"]["arguments"])["note"] == body

    def test_array_and_object_args_round_trip(self):
        raw = (
            "<function=lookup>\n"
            '<parameter=tags>\n["a", "b"]\n</parameter>\n'
            '<parameter=meta>\n{"k": 1}\n</parameter>\n'
            "</function>"
        )
        tc, ok = _parse_tool_body_xml(raw, ARG_SCHEMA)
        assert ok
        args = json.loads(tc["function"]["arguments"])
        assert args["tags"] == ["a", "b"]
        assert args["meta"] == {"k": 1}

    def test_malformed_inner_xml_does_not_raise(self):
        """Unterminated `<function=...>` and a missing `</parameter>` must yield a
        failed/filtered parse, never raise."""
        no_close_fn = "<function=lookup\n  count = 3\n</function"
        tc, ok = _parse_tool_body_xml(no_close_fn, ARG_SCHEMA)
        assert not ok and tc is None  # no `>` to close the function tag

        no_close_param = "<function=lookup>\n<parameter=count>\n3\n</function>"
        tc, ok = _parse_tool_body_xml(no_close_param, ARG_SCHEMA)
        # name parses fine; the unterminated param simply yields no args — no raise.
        assert ok
        assert json.loads(tc["function"]["arguments"]) == {}

    def test_without_schema_falls_back_to_coercion(self):
        """An empty schema map — what `tools=None` yields — keeps the best-effort
        json.loads (documents why callers must pass tools to get string-typed args right)."""
        tc, ok = _parse_tool_body_xml("<function=lookup>\n<parameter=zip>32286</parameter>\n</function>", {})
        assert ok
        args = json.loads(tc["function"]["arguments"])
        assert args["zip"] == 32286 and isinstance(args["zip"], int)

    def test_unknown_param_falls_back_to_coercion(self):
        """A param absent from the schema keeps best-effort coercion."""
        tc, ok = _parse_tool_body_xml("<function=lookup>\n<parameter=count>3</parameter>\n</function>", ARG_SCHEMA)
        assert ok
        assert json.loads(tc["function"]["arguments"])["count"] == 3

    def test_union_typed_params_omitted_and_coerced_best_effort(self):
        """A param with no single top-level `type` (`anyOf`/`enum`) carries no scalar
        type to coerce against, so it's omitted from the schema map and falls through
        to best-effort json.loads — `7` becomes int 7, a bare word stays a string.
        Pins the deliberate union-schema limitation the schema map documents."""
        schema = param_types_by_tool(
            [
                ToolSpec(
                    name="lookup",
                    description="",
                    parameters={
                        "type": "object",
                        "properties": {
                            "either": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
                            "choice": {"enum": ["a", "b"]},
                            "zip": {"type": "string"},
                        },
                    },
                )
            ]
        )
        assert schema["lookup"] == {"zip": "string"}  # union/enum params absent

        raw = (
            "<function=lookup>\n<parameter=either>\n7\n</parameter>\n<parameter=choice>\na\n</parameter>\n</function>"
        )
        tc, ok = _parse_tool_body_xml(raw, schema)
        assert ok
        args = json.loads(tc["function"]["arguments"])
        assert args["either"] == 7 and isinstance(args["either"], int)  # best-effort json
        assert args["choice"] == "a"  # bare word isn't valid json → stays string


class TestNativePreserveThinkingTemplates:
    """Qwen3.6 and Qwen3.8 are Qwen35Codec, but their own templates take
    preserve_thinking, so an explicit value is forwarded and the 3.5 fork is not
    installed. Omitting it keeps the template default: 3.6 strips history reasoning,
    3.8 keeps it."""

    _HISTORY: list[Message] = [
        Message(role="user", content="solve this"),
        Message(role="assistant", reasoning_content="careful chain of thought", content="answer one"),
        Message(role="user", content="follow up"),
    ]

    def _render(self, codec: MessageCodec) -> str:
        assert isinstance(codec, Qwen35Codec)
        assert codec._chat_template is None
        return codec.decode(codec.encode_messages(self._HISTORY, add_generation_prompt=True))

    @staticmethod
    def _native_codec(model: str, **kwargs: object) -> MessageCodec:
        with warnings.catch_warnings():
            warnings.simplefilter("error", PreserveThinkingForkedWarning)
            return get_codec(model, **kwargs)

    @pytest.mark.parametrize(
        ("model", "preserve_thinking", "keeps_history"),
        [
            pytest.param(_QWEN36, None, False, id="qwen3.6-default"),
            pytest.param(_QWEN36, False, False, id="qwen3.6-false"),
            pytest.param(_QWEN36, True, True, id="qwen3.6-true"),
            pytest.param(_QWEN38, None, True, id="qwen3.8-default"),
            pytest.param(_QWEN38, False, False, id="qwen3.8-false"),
            pytest.param(_QWEN38, True, True, id="qwen3.8-true"),
        ],
    )
    def test_history_reasoning(self, model: str, preserve_thinking: bool | None, keeps_history: bool):
        rendered = self._render(self._native_codec(model, preserve_thinking=preserve_thinking))
        assert ("careful chain of thought" in rendered) is keeps_history

    @pytest.mark.parametrize(
        ("model", "preserve_thinking"),
        [
            pytest.param(_QWEN36, True, id="qwen3.6-true"),
            pytest.param(_QWEN38, None, id="qwen3.8-default"),
            pytest.param(_QWEN38, True, id="qwen3.8-true"),
        ],
    )
    def test_history_render_is_prefix_of_next_turn(self, model: str, preserve_thinking: bool | None):
        """Keeping history reasoning is what lets an episode stay one sequence: each
        turn's gen-prompt render must extend the previous one's."""
        codec = self._native_codec(model, preserve_thinking=preserve_thinking)
        messages: list[Message] = [
            Message(role="system", content="be brief"),
            Message(role="user", content="q1"),
            Message(role="assistant", reasoning_content="reasoning one", content="a1"),
            Message(role="user", content="q2"),
            Message(role="assistant", reasoning_content="reasoning two", content="a2"),
            Message(role="user", content="q3"),
        ]
        renders = [
            codec.encode_messages(messages[:i], add_generation_prompt=True)
            for i, m in enumerate(messages)
            if m["role"] == "assistant"
        ]
        for prev, curr in zip(renders, renders[1:]):
            assert curr[: len(prev)] == prev

    def test_qwen38_keeps_reasoning_effort_turn(self):
        rendered = self._render(self._native_codec(_QWEN38))
        assert "Reasoning effort is set to" in rendered
        assert not re.search(r"<think>\s*</think>", rendered)

    def test_detected_from_template_not_name(self):
        """A tokenizer loaded from a local checkpoint has no model id to go by."""
        tokenizer = AutoTokenizer.from_pretrained(_QWEN38, use_fast=True)
        tokenizer.name_or_path = "/checkpoints/run-7"
        with warnings.catch_warnings():
            warnings.simplefilter("error", PreserveThinkingForkedWarning)
            codec = Qwen35Codec(tokenizer=tokenizer, preserve_thinking=False)
        assert "careful chain of thought" not in self._render(codec)


@pytest.mark.parametrize(
    ("template", "native"),
    [
        ("{# a template could mention preserve_thinking here #}hi", False),
        ("{{ 'preserve_thinking' }}", False),
        ("{% if preserve_thinking %}yes{% endif %}", True),
        ("{% generation %}{{ preserve_thinking }}{% endgeneration %}", True),
        ({"default": "{{ preserve_thinking }}"}, False),
        (None, False),
    ],
)
def test_native_preserve_thinking_detection(template: str | dict | None, native: bool):
    """Only a template that reads the variable counts, not one that mentions the name."""
    assert _chat_template_reads_preserve_thinking(template) is native
