"""Unit tests for schema coercion: which declared type an argument gets, and how its wire text reads."""

from __future__ import annotations

import json

import pytest

from post_training_cookbook.codecs.schema_coercion import coerce_by_schema, param_types_by_tool, parse_json
from post_training_cookbook.types import ToolSpec
from tests.codecs import ARG_SCHEMA

_LOOKUP = ARG_SCHEMA["lookup"]


@pytest.mark.parametrize(
    "spec, declared",
    [
        ({"type": "string"}, "string"),
        ({"type": ["string", "null"]}, "string|null"),
        ({"type": ["integer", "null"]}, "integer"),
        ({"anyOf": [{"type": "integer"}, {"type": "null"}]}, "integer"),
        ({"oneOf": [{"type": "null"}, {"type": "array"}]}, "array"),
        ({"type": ["string", "integer"]}, None),
        ({"anyOf": [{"type": "string"}, {"type": "integer"}]}, None),
        ({"type": ["null"]}, None),
        ({"enum": ["a", "b"]}, None),
    ],
    ids=[
        "scalar",
        "nullable-string",
        "nullable-list",
        "nullable-anyof",
        "nullable-oneof",
        "union",
        "anyof-union",
        "only-null",
        "enum",
    ],
)
def test_declared_type(spec: dict, declared: str | None):
    """A nullable union types as its one non-null type; a real union or no type falls back to best effort."""
    tool = ToolSpec(name="t", description="", parameters={"type": "object", "properties": {"p": spec}})
    assert param_types_by_tool([tool])["t"].get("p") == declared


@pytest.mark.parametrize(
    "value, declared, coerced",
    [
        ("null", "integer", None),
        ("None", "boolean", None),
        ("null", "object", None),
        ("null", "string", "null"),
        ("NULL", "string|null", None),
        ("None", "string|null", "None"),
    ],
)
def test_null_literal(value: str, declared: str, coerced: object):
    """An explicit null on a nullable param stays null; a plain string keeps the text, and a nullable
    string reads only `null` as null, as vLLM and sglang do."""
    assert coerce_by_schema(value, declared) == coerced


@pytest.mark.parametrize(
    "value, declared, coerced",
    [
        ("32286", _LOOKUP["zip"], "32286"),
        (" x ", _LOOKUP["note"], " x "),
        ("two\nlines", _LOOKUP["note"], "two\nlines"),
        ("3", _LOOKUP["count"], 3),
        ("3", "number", 3),
        ("2.5", "number", 2.5),
        ("true", _LOOKUP["flag"], True),
        ('["a", "b"]', _LOOKUP["tags"], ["a", "b"]),
        ('{"k": 1}', _LOOKUP["meta"], {"k": 1}),
        ("H-1068", None, "H-1068"),
        ("240000", None, 240000),
        ("2.5", None, 2.5),
        ("true", None, True),
        ("null", None, None),
        ('["a", "b"]', None, ["a", "b"]),
        (" x ", None, " x "),
        ("two\nlines", None, "two\nlines"),
    ],
)
def test_coerce(value: str, declared: str | None, coerced: object):
    """A declared type types the value; an undeclared one is JSON-or-raw. Compared as JSON text, which tells
    ``3``, ``3.0`` and ``True`` apart."""
    assert json.dumps(coerce_by_schema(value, declared)) == json.dumps(coerced)


@pytest.mark.parametrize(
    "value, declared",
    [("NaN", "number"), ("1e999", "number"), ("Infinity", None), ("[1, NaN]", "array"), ('{"k": 1e999}', "object")],
)
def test_non_finite_numbers_stay_raw(value: str, declared: str | None):
    """``float`` and ``json.loads`` accept them, but arguments holding them would not be JSON."""
    assert coerce_by_schema(value, declared) == value
    with pytest.raises(ValueError):
        parse_json(value)
