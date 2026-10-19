"""Muse Spark codec: sampled text parses to the right Message, and a parsed Message re-renders to the same tokens.

The parse tables run through ``parse_response`` on a stub tokenizer that decodes verbatim, so they need no hub
access. Tests taking the ``muse`` fixture need the gated hub repo and skip without it: they check that the release
template renders what the tables parse.
"""

from __future__ import annotations

import datetime
import json

import pytest

from post_training_cookbook.codecs import MuseSparkCodec
from post_training_cookbook.codecs.base import messages_for_template
from post_training_cookbook.types import Message, ToolCall, ToolCallFunction, ToolSpec
from tests.codecs import REFERENCE_MODELS, load_reference_codec


def _tool(name: str, **properties: str) -> ToolSpec:
    schema = {key: {"type": type_} for key, type_ in properties.items()}
    return ToolSpec(name=name, description=f"{name} tool.", parameters={"type": "object", "properties": schema})


MUSE_SPARK_TOOLS = [
    _tool("search_edicts", query="string"),
    _tool("fetch_holder", holder_id="string"),
    _tool(
        "file_allotment", holder_id="string", weave_id="string", requested_allotment_motes="integer", rush="boolean"
    ),
]
_OPENING: list[Message] = [
    Message(role="system", content="You are a filament-registry clerk."),
    Message(role="user", content="Holder H-6666 wants the Hearth weave at the highest allotment."),
]


def _channel(recipient: str, body: str, end: str = "<|eom|>", *, first: bool = False) -> str:
    """One channel as sampled; the first one's ``<|start|>assistant`` is the end of the generation prompt."""
    return f"{'' if first else '<|start|>assistant'} to={recipient}<|message|>{body}{end}"


def _atem(name: str, **params: str) -> str:
    rendered = "".join(f'<atem:parameter name="{key}">{value}</atem:parameter>\n' for key, value in params.items())
    return f'<atem:function_calls>\n<atem:invoke name="{name}">\n{rendered}</atem:invoke>\n</atem:function_calls>'


def _call(name: str, end: str = "<|eot|>", **params: str) -> str:
    return _channel(name, _atem(name, **params), end)


def _turn(*, reasoning: str | None = None, content: str | None = None, calls: list[tuple[str, dict]] = ()) -> Message:
    """The assistant Message a parse should produce; ``calls`` get the codec's per-turn ids ``call_1``, ``call_2``…

    Arguments compare as JSON text, which, unlike ``==``, tells ``1``, ``1.0`` and ``True`` apart.
    """
    msg = Message(role="assistant")
    if content:
        msg["content"] = content
    if reasoning:
        msg["reasoning_content"] = reasoning
    if calls:
        msg["tool_calls"] = [
            ToolCall(id=f"call_{i}", function=ToolCallFunction(name=name, arguments=json.dumps(args)))
            for i, (name, args) in enumerate(calls, start=1)
        ]
    return msg


class _StubTokenizer:
    """Stands in for the hub tokenizer: records what the codec hands the template, decodes to ``sampled``."""

    def __init__(self, sampled: str = "") -> None:
        self.sampled = sampled
        self.rendered: list[tuple[list[dict], dict]] = []

    def apply_chat_template(self, messages: list[dict], **kwargs) -> dict:
        self.rendered.append((messages, kwargs))
        return {"input_ids": [0]}

    def decode(self, tokens: list[int], skip_special_tokens: bool = False) -> str:
        return self.sampled


def _parse(text: str, tools: list[ToolSpec] | None = MUSE_SPARK_TOOLS) -> tuple[Message, bool]:
    """``parse_response`` on tokens that decode to ``text``."""
    return MuseSparkCodec(tokenizer=_StubTokenizer(text)).parse_response([0], tools=tools)


_REASONED = _channel("self", "r", first=True)
_CALL = _call("fetch_holder", holder_id="H-6666")
_FETCH_6666 = ("fetch_holder", {"holder_id": "H-6666"})
_FETCH_H1 = _atem("fetch_holder", holder_id="H-1")
_FILE_ARGS = {"holder_id": "H-6666", "weave_id": "loom_hearth", "requested_allotment_motes": 240000, "rush": True}

# Turns as the model samples them; the gated tests re-render each one.
_SAMPLED = {
    "reasoning+call": _channel("self", "Look the holder up first.", first=True) + _CALL,
    "commentary+int-and-bool-args": _channel("self", "The cap is 240000.", first=True)
    + _channel("commentary", "Filing now.")
    + _call(
        "file_allotment", holder_id="H-6666", weave_id="loom_hearth", requested_allotment_motes="240000", rush="true"
    ),
    "two-parallel-calls": _channel("self", "Two lookups.", first=True)
    + _call("search_edicts", "<|eom|>", query="EDICT-index")
    + _CALL,
    "string-typed-numbers": _channel("self", "Both look like numbers.", first=True)
    + _call("search_edicts", "<|eom|>", query="2026")
    + _call("fetch_holder", holder_id="1068"),
    "final-answer": _channel("self", "All filed.", first=True)
    + _channel("user", "Filed H-6666 on Hearth.", "<|eot|>"),
}
# For the shared round trip in test_codec.py; the template has no thinking switch.
MUSE_SPARK_SAMPLED_TURNS: dict[bool | None, dict[str, str]] = {None: _SAMPLED}

_COMPLETE = [
    pytest.param(
        _SAMPLED["reasoning+call"],
        _turn(reasoning="Look the holder up first.", calls=[_FETCH_6666]),
        True,
        id="reasoning+call",
    ),
    pytest.param(
        _SAMPLED["commentary+int-and-bool-args"],
        _turn(reasoning="The cap is 240000.", content="Filing now.", calls=[("file_allotment", _FILE_ARGS)]),
        True,
        id="commentary+int-and-bool-args",
    ),
    pytest.param(
        _SAMPLED["two-parallel-calls"],
        _turn(reasoning="Two lookups.", calls=[("search_edicts", {"query": "EDICT-index"}), _FETCH_6666]),
        True,
        id="two-parallel-calls",
    ),
    pytest.param(
        _SAMPLED["string-typed-numbers"],
        _turn(
            reasoning="Both look like numbers.",
            calls=[("search_edicts", {"query": "2026"}), ("fetch_holder", {"holder_id": "1068"})],
        ),
        True,
        id="string-typed-numbers",
    ),
    pytest.param(
        _SAMPLED["final-answer"],
        _turn(reasoning="All filed.", content="Filed H-6666 on Hearth."),
        True,
        id="final-answer",
    ),
    pytest.param(
        _channel("self", "a", first=True)
        + _channel("self", "b")
        + _channel("commentary", "c")
        + _channel("user", "d", "<|eot|>"),
        _turn(reasoning="a\nb", content="cd"),
        True,
        id="recipients-route-to-fields",
    ),
    *(
        pytest.param(_channel("user", "done", end, first=True) + "trailing text", _turn(content="done"), True, id=id_)
        for end, id_ in [("<|eot|>", "ends-at-eot"), ("<|end_of_text|>", "ends-at-end-of-text")]
    ),
    pytest.param(
        _REASONED + _call("fetch_holder", "<|end_of_text|>", holder_id="H-6666"),
        _turn(reasoning="r", calls=[_FETCH_6666]),
        True,
        id="call-ended-by-end-of-text",
    ),
    pytest.param(
        _REASONED + "\n" + _CALL, _turn(reasoning="r", calls=[_FETCH_6666]), True, id="newline-before-header"
    ),
    # The template's own examples address `to=namespace.function`; the name is kept as written.
    *(
        pytest.param(
            _channel(name, _atem(name, holder_id="H-1"), "<|eot|>", first=True),
            _turn(calls=[(name, {"holder_id": "H-1"})]),
            True,
            id=f"namespaced-{name}",
        )
        for name in ("ns.fetch_holder", "fetch_holder.fetch_holder")
    ),
    pytest.param(
        _channel("get-weather", _atem("get-weather", city="Paris"), "<|eot|>", first=True),
        _turn(calls=[("get-weather", {"city": "Paris"})]),
        True,
        id="hyphenated-tool-name",
    ),
    # As every serving parser reads it.
    pytest.param(
        _channel("commentary", _FETCH_H1, "<|eot|>", first=True),
        _turn(content=_FETCH_H1),
        True,
        id="commentary-invoke-is-content",
    ),
    pytest.param(_REASONED + "\n", _turn(reasoning="r"), True, id="trailing-whitespace-is-not-content"),
    pytest.param(
        _call("search_edicts", "<|eom|>", query="q") + _call("fetch_holder", holder_id="H-1"),
        _turn(calls=[("search_edicts", {"query": "q"}), ("fetch_holder", {"holder_id": "H-1"})]),
        True,
        id="call-ids-run-across-channels",
    ),
    pytest.param(
        _channel(
            "fetch_holder",
            _FETCH_H1.replace(
                "</atem:invoke>\n</atem:function_calls>",
                "</atem:invoke>\n" + _atem("fetch_holder", holder_id="H-2").removeprefix("<atem:function_calls>\n"),
            ),
            "<|eot|>",
            first=True,
        ),
        _turn(calls=[("fetch_holder", {"holder_id": "H-1"}), ("fetch_holder", {"holder_id": "H-2"})]),
        True,
        id="two-invokes-in-one-block",
    ),
    # Typed by the schema, as the other codecs type them: a string-typed `1068` is not the integer 1068.
    pytest.param(
        _call("search_edicts", "<|eom|>", query="240000")
        + _call("file_allotment", holder_id="1068", weave_id="null", requested_allotment_motes="240000", rush="false"),
        _turn(
            calls=[
                ("search_edicts", {"query": "240000"}),
                (
                    "file_allotment",
                    {"holder_id": "1068", "weave_id": "null", "requested_allotment_motes": 240000, "rush": False},
                ),
            ]
        ),
        True,
        id="declared-strings-stay-verbatim",
    ),
    # `&amp;` goes last, so a double-escaped `&amp;lt;` stays `&lt;`.
    pytest.param(
        _channel(
            "x",
            '<atem:function_calls>\n<atem:invoke name="a&amp;b&lt;c&gt;">\n'
            '<atem:parameter name="k&quot;&#39;&amp;lt;">v</atem:parameter>\n</atem:invoke>\n</atem:function_calls>',
            "<|eot|>",
            first=True,
        ),
        _turn(calls=[("a&b<c>", {"k\"'&lt;": "v"})]),
        True,
        id="attribute-entities-are-unescaped",
    ),
]

# Multi-turn GRPO detects a length-truncated turn only by `ok`: a cut inside reasoning or prose keeps what it
# produced, a cut inside a header or a tool channel fails.
_TRUNCATED = [
    pytest.param(" to=self<|message|>half a tho", _turn(reasoning="half a tho"), True, id="cut-in-reasoning"),
    pytest.param(
        _REASONED + "<|start|>assistant to=commentary<|message|>Fil",
        _turn(reasoning="r", content="Fil"),
        True,
        id="cut-in-commentary",
    ),
    pytest.param(_REASONED, _turn(reasoning="r"), True, id="cut-between-channels"),
    pytest.param(" t", _turn(), False, id="cut-at-t"),
    pytest.param(" to", _turn(), False, id="cut-at-to"),
    pytest.param(" to=fetch_hol", _turn(), False, id="cut-in-first-header"),
    pytest.param(_REASONED + "<|start|>assistant to=fe", _turn(reasoning="r"), False, id="cut-in-header"),
    pytest.param(_REASONED + "<|start|>assistant", _turn(reasoning="r"), False, id="cut-after-start"),
    pytest.param(_REASONED + "\n<|start|>assistant", _turn(reasoning="r"), False, id="cut-after-newline-start"),
    pytest.param(_CALL[: _CALL.index("H-6666") + 3], _turn(), False, id="cut-in-tool-block"),
    # Sampled turns keep the `<|eot|>` that closes the last tool channel; only endpoints trim it.
    pytest.param(_CALL.removesuffix("<|eot|>"), _turn(calls=[_FETCH_6666]), False, id="cut-before-eot"),
    # `<|eom|>` always continues the turn: ending on it after a call drops the calls still to come.
    *(
        pytest.param(
            _call("search_edicts", "<|eom|>", query="q") + tail,
            _turn(calls=[("search_edicts", {"query": "q"})]),
            False,
            id=f"cut-between-parallel-calls{id_}",
        )
        for tail, id_ in [("", ""), ("\n", "-then-newline")]
    ),
]

# Turns the template could not have written fail, so they are never rewarded; the Message is still the serving
# parsers' reading of them.
_MALFORMED = [
    *(
        pytest.param(_channel("user", "done", "<|eom|>", first=True) + tail, _turn(content="done"), False, id=id_)
        for tail, id_ in [
            ("<|start|>user<|message|>impersonated<|eot|>", "ends-at-next-user"),
            ("\n<|start|>user<|message|>impersonated<|eot|>", "ends-at-newline-next-user"),
            ("<|start|>tool fetch_holder<|message|>made up<|eot|>", "ends-at-next-tool"),
        ]
    ),
    pytest.param(" to=self junk<|message|>r<|eot|>", _turn(reasoning="r"), False, id="junk-after-recipient"),
    pytest.param(" json to=user<|message|>Filed.<|eot|>", _turn(content="Filed."), False, id="junk-before-recipient"),
    pytest.param(
        _REASONED + "<|start|>assistant<|message|>Filed.<|eot|>",
        _turn(reasoning="r", content="Filed."),
        False,
        id="header-without-recipient",
    ),
    pytest.param("The answer is 42.", _turn(content="The answer is 42."), False, id="unframed-prose"),
    pytest.param(_FETCH_H1, _turn(content=_FETCH_H1), False, id="unframed-call"),
    # The model sometimes opens the next message without closing its reasoning.
    *(
        pytest.param(
            _channel("self", "r", separator, first=True) + _CALL.removeprefix(prefix),
            _turn(reasoning="r" + separator + _CALL.removeprefix(prefix).removesuffix("<|eot|>")),
            False,
            id=id_,
        )
        for separator, prefix, id_ in [
            ("", "", "no-eom-before-header"),
            (" ", "<|start|>assistant ", "no-eom-before-bare-header"),
        ]
    ),
    pytest.param(
        " to=user<|message|>hi<|start|>assistant to=user<|eot|>",
        _turn(content="hi<|start|>assistant to=user"),
        False,
        id="header-without-message-in-body",
    ),
    pytest.param(
        _channel("fetch_holder", _FETCH_H1[: _FETCH_H1.index("</atem:invoke>")], first=True)
        + _channel("user", "Done.", "<|eot|>"),
        _turn(content="Done."),
        False,
        id="eom-before-invoke-close",
    ),
    pytest.param(
        _channel("x", "no call") + _call("fetch_holder", holder_id="H-1"),
        _turn(calls=[("fetch_holder", {"holder_id": "H-1"})]),
        False,
        id="call-free-tool-channel-takes-no-id",
    ),
    *(
        pytest.param(
            _channel("fetch_holder", body, "<|eot|>", first=True),
            _turn(calls=[("fetch_holder", {"holder_id": holder}) for holder in kept]),
            False,
            id=id_,
        )
        for body, kept, id_ in [
            ("no call here", [], "no-invoke"),
            (_FETCH_H1.replace("</atem:invoke>", ""), [], "unclosed-invoke"),
            (
                _FETCH_H1.replace("</atem:function_calls>", '<atem:invoke name="x">\n</atem:function_calls>'),
                ["H-1"],
                "unclosed-2nd-invoke",
            ),
            (
                _FETCH_H1.replace("</atem:invoke>", '<atem:parameter name="rush">true\n</atem:invoke>'),
                ["H-1"],
                "unclosed-param",
            ),
            (_atem("", holder_id="H-1"), [], "empty-name"),
            (_atem("fetch_holder", **{"": "H-0", "holder_id": "H-1"}), ["H-1"], "empty-key"),
            (_FETCH_H1.removeprefix("<atem:function_calls>\n"), ["H-1"], "no-block-open"),
            (
                _FETCH_H1.removeprefix("<atem:function_calls>\n").removesuffix("</atem:function_calls>"),
                ["H-1"],
                "no-block",
            ),
            ("<atem:function_calls>\n</atem:function_calls>", [], "empty-block"),
            ("Calling. " + _FETCH_H1, ["H-1"], "text-before-block"),
            (_FETCH_H1 + "\n" + _FETCH_H1, ["H-1", "H-1"], "two-blocks"),
        ]
    ),
]


class TestParse:
    """Sampled text → Message, through the public parse entry points."""

    @pytest.mark.parametrize("text, expected, ok", _COMPLETE + _TRUNCATED + _MALFORMED)
    def test_parse_response(self, text: str, expected: Message, ok: bool):
        assert _parse(text) == (expected, ok)

    def test_only_strings_are_typed_by_the_schema(self):
        """A nullable string reads ``null`` as None, a number keeps its literal (``3``, not ``3.0``), and an
        undeclared value is JSON-or-raw."""
        properties = {
            "zip": {"type": ["string", "null"]},
            "alt": {"type": ["string", "null"]},
            "n": {"type": "number"},
        }
        lookup = ToolSpec(name="lookup", description="", parameters={"type": "object", "properties": properties})
        msg = _parse(_call("lookup", zip="1068", alt="null", n="3", tags='["a"]', x="NaN"), [lookup])
        expected = {"zip": "1068", "alt": None, "n": 3, "tags": ["a"], "x": "NaN"}
        assert msg == (_turn(calls=[("lookup", expected)]), True)

    @pytest.mark.parametrize(
        "content, reasoning, expected, ok",
        [
            pytest.param(
                "Filed it.", "thought", _turn(reasoning="thought", content="Filed it."), True, id="plain-content"
            ),
            pytest.param(
                _channel("self", "more", first=True)
                + _call("search_edicts", "<|eom|>", query="EDICT-1")
                + _call("fetch_holder", "", holder_id="1068"),
                "thought",
                _turn(
                    reasoning="thought\nmore",
                    calls=[("search_edicts", {"query": "EDICT-1"}), ("fetch_holder", {"holder_id": "1068"})],
                ),
                True,
                id="channelled-content",
            ),
            # Endpoints trim the stop token `<|eot|>` that ends the last tool channel.
            pytest.param(_CALL.removesuffix("<|eot|>"), None, _turn(calls=[_FETCH_6666]), True, id="stop-trimmed"),
            pytest.param(
                _CALL.removesuffix("<|eot|>") + "\n",
                None,
                _turn(calls=[_FETCH_6666]),
                True,
                id="stop-trimmed-then-newline",
            ),
            pytest.param(
                _CALL[: _CALL.index("</atem:function_calls>")],
                None,
                _turn(calls=[_FETCH_6666]),
                False,
                id="block-still-open",
            ),
        ],
    )
    def test_parse_completion(self, content: str, reasoning: str | None, expected: Message, ok: bool):
        assert MuseSparkCodec.parse_completion(content=content, reasoning=reasoning, tools=MUSE_SPARK_TOOLS) == (
            expected,
            ok,
        )


_TWO_LOOPS: list[Message] = _OPENING + [
    _turn(reasoning="old", calls=[("fetch_holder", {})]),
    Message(role="tool", content="tier gold", tool_call_id="call_1"),
    Message(role="assistant", content="Checking."),
    Message(role="user", content="And H-7777?"),
    _turn(reasoning="new", calls=[("search_edicts", {})]),
    Message(role="tool", content="cap 240000", tool_call_id="call_1"),
]
# `datasets` gives every message every field of the rows, `None` where it is absent.
_UNION_ROW = dict.fromkeys(("content", "reasoning_content", "tool_calls", "tool_call_id", "name"))


class TestRenderInputs:
    """Message → template input: what the codec hands the template, checked without the gated tokenizer."""

    @staticmethod
    def _render(messages: list[Message], **codec_kwargs) -> tuple[list[dict], dict]:
        stub = _StubTokenizer()
        MuseSparkCodec(tokenizer=stub, **codec_kwargs).encode_messages(
            messages, tools=MUSE_SPARK_TOOLS, add_generation_prompt=True
        )
        [rendered] = stub.rendered
        return rendered

    def test_no_thinking_switch(self):
        with pytest.raises(ValueError, match="no thinking switch"):
            MuseSparkCodec(tokenizer=_StubTokenizer(), enable_thinking=False)

    def test_turn_terminator_is_empty(self):
        codec = MuseSparkCodec(tokenizer=_StubTokenizer())
        assert [codec.turn_terminator(role) for role in (None, "user", "tool", "assistant")] == [[]] * 4

    @pytest.mark.parametrize("effort", [None, "high"])
    def test_template_date_and_effort(self, effort: str | None):
        _, kwargs = self._render(_OPENING, thinking_effort=effort, current_date=datetime.date(2026, 8, 1))
        assert (kwargs["current_date"], kwargs.get("reasoning_effort")) == ("2026-08-01", effort)

    def test_date_defaults_to_today(self):
        before = datetime.date.today().isoformat()
        _, kwargs = self._render(_OPENING)
        assert kwargs["current_date"] in {before, datetime.date.today().isoformat()}

    @pytest.mark.parametrize("key", ["reasoning_content", "reasoning"])
    def test_reasoning_only_turn_refused_last(self, key: str):
        codec = MuseSparkCodec(tokenizer=_StubTokenizer())
        turn = {"role": "assistant", key: "r"}
        with pytest.raises(ValueError, match="no stop token"):
            codec.encode_messages(_OPENING + [turn])
        codec.encode_messages(_OPENING + [turn], add_generation_prompt=True)
        codec.encode_messages(_OPENING + [turn, Message(role="user", content="go on")], add_generation_prompt=True)

    @pytest.mark.parametrize(
        "preserve_thinking, kept",
        [(None, ["old", None, "new"]), (False, [None, None, "new"])],
        ids=["default", "strip"],
    )
    def test_history_reasoning(self, preserve_thinking: bool | None, kept: list[str | None]):
        messages, _ = self._render(_TWO_LOOPS, preserve_thinking=preserve_thinking)
        assert [msg.get("reasoning_content") for msg in messages if msg["role"] == "assistant"] == kept

    @pytest.mark.parametrize(
        "messages, names",
        [
            # Unnamed results (tau-bench's shape) reuse call_1 every turn yet take their own call's name.
            pytest.param(_TWO_LOOPS, ["fetch_holder", "search_edicts"], id="unnamed"),
            pytest.param(
                [{**_UNION_ROW, **msg} for msg in _TWO_LOOPS], ["fetch_holder", "search_edicts"], id="datasets-row"
            ),
            pytest.param(
                _OPENING
                + [
                    _turn(calls=[("fetch_holder", {})]),
                    Message(role="tool", content="tier gold", tool_call_id="call_1", name="registry.fetch_holder"),
                ],
                ["registry.fetch_holder"],
                id="named",
            ),
            # The template names a result whose id matches no call after the id itself.
            pytest.param(
                _OPENING + [Message(role="tool", content="tier gold", tool_call_id="call_9")], [None], id="unknown-id"
            ),
        ],
    )
    def test_tool_result_names(self, messages: list[Message], names: list[str | None]):
        rendered, _ = self._render(messages)
        assert [msg.get("name") for msg in rendered if msg["role"] == "tool"] == names

    def test_tool_result_without_name_or_id(self):
        with pytest.raises(ValueError, match="neither a name nor a tool_call_id"):
            self._render(_OPENING + [{"role": "tool", "content": "tier gold", "tool_call_id": None}])


@pytest.fixture(scope="session")
def muse() -> MuseSparkCodec:
    return load_reference_codec(REFERENCE_MODELS["muse-spark"])


class TestTemplate:
    def test_render_matches_template(self, muse: MuseSparkCodec):
        tokens = muse.encode_messages(_OPENING, tools=MUSE_SPARK_TOOLS, add_generation_prompt=True)
        tools = [{"type": "function", "function": t} for t in MUSE_SPARK_TOOLS]
        template = muse._tokenizer.apply_chat_template(
            messages_for_template(_OPENING),
            tools=tools,
            tokenize=True,
            add_generation_prompt=True,
            current_date=muse._current_date,
        )
        assert tokens == template["input_ids"]
        rendered = muse._tokenizer.decode(tokens)
        assert rendered.count(f"Current date: {muse._current_date}.") == 1
        assert rendered.endswith("<|eot|><|start|>assistant")

    def test_reasoning_only_turn_has_no_stop(self, muse: MuseSparkCodec):
        """Why the codec refuses to render one last: the template ends it on <|eom|>, which does not stop sampling."""
        turn = Message(role="assistant", reasoning_content="r")
        rendered = muse._tokenizer.apply_chat_template(messages_for_template(_OPENING + [turn]), tokenize=False)
        assert rendered.endswith("to=self<|message|>r<|eom|>")


class TestRoundTrip:
    """Tokens → Message → tokens; the shared round trip in test_codec.py re-renders ``MUSE_SPARK_SAMPLED_TURNS``."""

    @pytest.mark.parametrize("shape", list(_SAMPLED))
    def test_sampled_turn_parses_as_its_table_row(self, muse: MuseSparkCodec, shape: str):
        """The hub tokenizer decodes a sampled turn verbatim, as the stub the parse tables run on does."""
        assert muse.parse_response(muse.encode(_SAMPLED[shape]), tools=MUSE_SPARK_TOOLS) == _parse(_SAMPLED[shape])

    def test_rendered_turn_parses_back_and_reencodes(self, muse: MuseSparkCodec):
        args = {"holder_id": "H-6666", "weave_id": "loom_hearth", "requested_allotment_motes": 240000}
        turn = _turn(reasoning="  keep my\nspacing ", content="Filing.", calls=[("file_allotment", args)])
        prompt = muse.encode_messages(_OPENING, tools=MUSE_SPARK_TOOLS, add_generation_prompt=True)
        full = muse.encode_messages(_OPENING + [turn], tools=MUSE_SPARK_TOOLS)
        assert muse.parse_response(full[len(prompt) :], tools=MUSE_SPARK_TOOLS) == (turn, True)
