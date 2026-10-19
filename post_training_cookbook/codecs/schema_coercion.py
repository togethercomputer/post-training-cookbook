"""Recover argument types for tool-call formats that serialize values untyped.

Some chat templates (notably Qwen3.5's nested-XML form) render every scalar
argument **unquoted**, so the wire form carries no type information — "32286"
could be a zip-code string or the integer 32286. We recover the intended type by
coercing each value against the param's declared JSON-schema type.

This module is pure: no tokenizer, no IO, no model knowledge. It is reusable by
any wire format whose values arrive as strings alongside a declared schema.
"""

from __future__ import annotations

import json
import math

from post_training_cookbook.types import ToolSpec

_BOOL_LITERALS = {"true": True, "false": False, "1": True, "0": False}


def parse_json(text: str) -> object:
    """``json.loads`` held to JSON: ``NaN``, ``Infinity`` and an overflowing ``1e999`` raise ``ValueError``.

    ``json.loads`` accepts them, even nested, but arguments built from them would not be JSON.
    """
    value = json.loads(text)
    json.dumps(value, allow_nan=False)
    return value


def _best_effort_json(value: str) -> object:
    try:
        return parse_json(value)
    except ValueError:
        return value


def coerce_by_schema(value: str, declared_type: str | None) -> object:
    """Coerce an unquoted wire value to its declared JSON-schema type.

    - ``string`` → verbatim (interior and edge whitespace preserved)
    - ``string|null`` → ``None`` for ``null`` (case-insensitive), else verbatim, as vLLM
      and sglang read a nullable string
    - any other type: ``null``/``None`` (case-insensitive) → ``None``, which keeps an
      explicit null on a nullable param (``param_types_by_tool`` types it by its non-null type)
    - ``boolean`` → ``true``/``false``/``True``/``False``/``1``/``0`` (case-insensitive)
    - ``integer`` → ``int``; ``number`` → ``int`` when whole, else ``float``; a non-finite number is not JSON,
      so it stays raw
    - ``array``/``object`` → ``parse_json``
    - declared type absent or unknown → best-effort ``parse_json``

    Never raises: a value that does not fit its declared type falls back to the raw string.
    The caller is responsible for stripping any wire-format framing before passing ``value`` in.
    """
    if declared_type == "string":
        return value
    if declared_type == "string|null":
        return None if value.strip().lower() == "null" else value
    if declared_type is not None and value.strip().lower() in ("null", "none"):
        return None
    if declared_type == "boolean":
        return _BOOL_LITERALS.get(value.strip().lower(), value)
    if declared_type == "integer":
        try:
            return int(value.strip())
        except ValueError:
            return value
    if declared_type == "number":
        try:  # a whole number stays an int, as serving reads it, so `3` re-renders as `3`, not `3.0`
            return int(value.strip())
        except ValueError:
            pass
        try:
            number = float(value.strip())
        except ValueError:
            return value
        return number if math.isfinite(number) else value
    # array, object, or an absent/unknown type → best-effort JSON
    return _best_effort_json(value)


def param_schemas_by_tool(tools: list[ToolSpec] | None) -> dict[str, dict]:
    """Map ``{tool_name: {param_name: json_schema}}`` from tool specs."""
    return {t["name"]: t.get("parameters", {}).get("properties") or {} for t in tools or []}


def param_types_by_tool(tools: list[ToolSpec] | None) -> dict[str, dict[str, str]]:
    """Map ``{tool_name: {param_name: json_schema_type}}`` from tool specs.

    Only top-level property types are captured; that's all an arg parser needs to
    decide how to coerce a value. A nullable union (``["integer", "null"]``, or an
    ``anyOf``/``oneOf`` of one type and ``null``) counts as its non-null type, except
    that a nullable string is ``"string|null"``: only a string has to tell the text
    ``null`` from null. So a nullable string-typed ``"123"`` stays a string. Params with
    no single type are omitted and fall through to best-effort coercion.
    """
    return {
        tool: {
            name: declared
            for name, spec in properties.items()
            if isinstance(spec, dict) and (declared := declared_type(spec)) is not None
        }
        for tool, properties in param_schemas_by_tool(tools).items()
    }


def declared_type(spec: dict) -> str | None:
    """The single JSON-schema type ``spec`` declares, as ``param_types_by_tool`` resolves it."""
    declared = spec.get("type")
    if isinstance(declared, str):
        return declared
    options = declared if isinstance(declared, list) else [branch.get("type") for branch in _union(spec)]
    non_null = {option for option in options if isinstance(option, str) and option != "null"}
    if len(non_null) != 1:
        return None
    declared = non_null.pop()
    return "string|null" if declared == "string" and "null" in options else declared


def non_null_branch(spec: dict) -> dict:
    """The lone non-null branch of an ``anyOf``/``oneOf``, which holds its ``properties``/``items``; else ``spec``.

    A union of several same-typed branches still has a ``declared_type`` but no single
    branch, so its children go untyped.
    """
    branches = [branch for branch in _union(spec) if branch.get("type") != "null"]
    return branches[0] if len(branches) == 1 else spec


def _union(spec: dict) -> list[dict]:
    return [branch for branch in spec.get("anyOf") or spec.get("oneOf") or [] if isinstance(branch, dict)]
