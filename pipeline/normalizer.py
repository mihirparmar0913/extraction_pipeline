"""Step 5 - deterministic clean-up in plain code (no LLM = no randomness).

Cleans whitespace, converts dates to ISO, turns '£1,200.50' into 1200.5 where the
schema wants a number, and orders object keys exactly like the schema.
"""
from __future__ import annotations

import re

from dateutil import parser as dateparser

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _resolve(schema, root):
    """Follow a local $ref like '#/$defs/Address'."""
    seen = 0
    while isinstance(schema, dict) and "$ref" in schema and seen < 20:
        node = root
        for part in schema["$ref"].lstrip("#/").split("/"):
            node = node.get(part, {}) if isinstance(node, dict) else {}
        schema, seen = node, seen + 1
    return schema if isinstance(schema, dict) else {}


def _types(schema) -> list[str]:
    t = schema.get("type")
    return t if isinstance(t, list) else ([t] if t else [])


def _to_iso(text: str, dayfirst: bool) -> str | None:
    if _ISO_DATE.match(text):
        return text
    try:
        parsed = dateparser.parse(text, dayfirst=dayfirst)
    except (ValueError, OverflowError):
        return None
    return parsed.date().isoformat() if parsed else None


def _to_number(text: str, want_int: bool):
    cleaned = re.sub(r"[^\d.\-]", "", text)
    try:
        number = float(cleaned)
    except ValueError:
        return None
    return int(number) if (want_int or number.is_integer()) else number


def _walk(value, schema, root, dayfirst):
    schema = _resolve(schema, root)
    if isinstance(value, dict):
        props = schema.get("properties", {})
        ordered = [k for k in props if k in value] + [k for k in value if k not in props]
        return {k: _walk(value[k], props.get(k, {}), root, dayfirst) for k in ordered}
    if isinstance(value, list):
        items = schema.get("items", {})
        return [_walk(v, items if isinstance(items, dict) else {}, root, dayfirst) for v in value]
    if isinstance(value, str):
        value = re.sub(r"\s+", " ", value).strip()
        types = _types(schema)
        if schema.get("format") == "date":
            return _to_iso(value, dayfirst) or value
        if "string" not in types and ("number" in types or "integer" in types):
            converted = _to_number(value, want_int="integer" in types and "number" not in types)
            return value if converted is None else converted
    return value


def normalize(data: dict, schema: dict, *, dayfirst: bool = True) -> dict:
    return _walk(data, schema, schema, dayfirst)
