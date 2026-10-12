"""Unit tests for schema coercion: which declared type an argument gets, and how a null literal reads."""

from __future__ import annotations

import pytest

from post_training_cookbook.codecs.schema_coercion import coerce_by_schema, param_types_by_tool
from post_training_cookbook.types import ToolSpec


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
