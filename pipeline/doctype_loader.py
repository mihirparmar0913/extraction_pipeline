"""Load real document_types.json and convert to the pipeline's DocType format.

The real data has a different shape than the simple keys.json/schema.json the
pipeline was prototyped with:

Real format (document_types.json):
  [{ "name": "Lease: Commercial",
     "fields": [
       { "name": "effective_date", "description": "...", "item_type": "string", "schema": {} },
       { "name": "consideration",  "description": "<huge extraction prompt>",
         "item_type": "string",
         "schema": {
           "type": "object",
           "required": ["consideration", "rent_model_terms", "contingent_consideration"],
           "properties": { ... }
         }
       },
       ...
     ]
  }]

This module provides:

1. load_doctype_from_document_types() — full multi-field DocType for a single-call run.

2. split_into_field_groups() — splits into:
   - simple_doctype  : all simple fields (strings) in ONE call
   - complex_dts     : one DocType per complex field, each using that field's schema

3. For 'consideration'-style complex fields whose schema has multiple sub-array
   properties (e.g., consideration, rent_model_terms, contingent_consideration),
   split_complex_field_into_sub_calls() returns one DocType per sub-property,
   so each LLM call only has to produce output for one array. This keeps
   output tokens well within model limits.
"""
from __future__ import annotations

import json
from pathlib import Path

from .doctype import DocType, DocTypeError, KeySpec


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_document_types(path: str | Path) -> list[dict]:
    """Load the full document_types.json array."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DocTypeError(f"File not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DocTypeError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(raw, list):
        raise DocTypeError("document_types.json must be a JSON array")
    return raw


def list_document_type_names(path: str | Path) -> list[str]:
    """Return the names of all document types in the file."""
    return [dt["name"] for dt in load_document_types(path)]


def _get_document_type(path: str | Path, name: str) -> dict:
    doc_types = load_document_types(path)
    for dt in doc_types:
        if dt.get("name") == name:
            return dt
    available = [dt.get("name", "?") for dt in doc_types]
    raise DocTypeError(
        f"Document type '{name}' not found. Available: {available}"
    )


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def _is_complex_field(field: dict) -> bool:
    """A field is 'complex' if it has a non-empty per-field JSON schema."""
    return bool(field.get("schema"))


def _simple_field_schema(field: dict) -> dict:
    item_type = field.get("item_type", "string")
    type_map = {
        "string": ["string", "null"],
        "number": ["number", "null"],
        "boolean": ["boolean", "null"],
    }
    return {"type": type_map.get(item_type, ["string", "null"])}


def _field_to_keyspec(field: dict) -> KeySpec:
    name = field.get("name", "")
    description = field.get("description", "")
    group = field.get("field_type", "default")
    if not description:
        description = f"Extract the {name.replace('_', ' ')} from the document."
    return KeySpec(group=group, key=name, description=description)


def _synthesise_full_schema(fields: list[dict]) -> dict:
    """Combined JSON Schema for ALL fields together."""
    properties: dict = {}
    required: list[str] = []
    for field in fields:
        name = field["name"]
        if _is_complex_field(field):
            properties[name] = field["schema"]
        else:
            properties[name] = _simple_field_schema(field)
        required.append(name)
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_doctype_from_document_types(
    path: str | Path, document_type_name: str
) -> DocType:
    """Single DocType covering ALL fields. For single-call or dry-run use."""
    selected = _get_document_type(path, document_type_name)
    fields = selected.get("fields", [])
    if not fields:
        raise DocTypeError(f"Document type '{document_type_name}' has no fields")
    specs = [_field_to_keyspec(f) for f in fields if f.get("name")]
    if not specs:
        raise DocTypeError(f"No valid fields in document type '{document_type_name}'")
    return DocType(keys=tuple(specs), schema=_synthesise_full_schema(fields))


def split_into_field_groups(
    path: str | Path, document_type_name: str
) -> tuple[DocType | None, list[dict]]:
    """Split a document type into:
      - simple_doctype: all simple (no per-field schema) fields combined
      - complex_field_specs: list of raw field dicts for complex fields

    Returns (simple_doctype_or_None, list_of_complex_field_dicts).
    Callers can then call split_complex_field_into_sub_calls() per complex field.
    """
    selected = _get_document_type(path, document_type_name)
    fields = selected.get("fields", [])

    simple_fields = [f for f in fields if f.get("name") and not _is_complex_field(f)]
    complex_fields = [f for f in fields if f.get("name") and _is_complex_field(f)]

    simple_dt: DocType | None = None
    if simple_fields:
        simple_schema: dict = {
            "type": "object",
            "properties": {f["name"]: _simple_field_schema(f) for f in simple_fields},
            "required": [f["name"] for f in simple_fields],
            "additionalProperties": False,
        }
        simple_dt = DocType(
            keys=tuple(_field_to_keyspec(f) for f in simple_fields),
            schema=simple_schema,
        )

    return simple_dt, complex_fields


def split_complex_field_into_sub_calls(field: dict) -> list[tuple[str, DocType]]:
    """For a complex field, produce one DocType per top-level sub-property.

    For example, the 'consideration' field has schema:
      { required: ["consideration", "rent_model_terms", "contingent_consideration"],
        properties: { consideration: {...}, rent_model_terms: {...}, ... } }

    This returns:
      [ ("consideration.consideration",   DocType for just the consideration array),
        ("consideration.rent_model_terms", DocType for just rent_model_terms),
        ("consideration.contingent_consideration", DocType for just contingent_consideration) ]

    Each DocType has a schema that wraps ONLY that sub-property, so the model
    produces a small, focused output — staying within token limits.

    If the field schema is NOT a multi-property object, returns a single entry
    with the field's schema as-is.
    """
    field_name = field["name"]
    field_schema = field["schema"]
    field_description = field.get("description", "")
    field_group = field.get("field_type", "complex")

    top_props = field_schema.get("properties", {})
    top_required = field_schema.get("required", [])

    # Only split if schema has multiple top-level properties
    if len(top_props) <= 1:
        return [(
            field_name,
            DocType(
                keys=(KeySpec(group=field_group, key=field_name, description=field_description),),
                schema=field_schema,
            )
        )]

    results: list[tuple[str, DocType]] = []
    for prop_name, prop_schema in top_props.items():
        # Build a minimal schema for just this sub-property
        sub_schema: dict = {
            "type": "object",
            "properties": {prop_name: prop_schema},
            "required": [prop_name] if prop_name in top_required else [],
            "additionalProperties": False,
        }
        # Carry over any $defs from the parent schema
        if "$defs" in field_schema:
            sub_schema["$defs"] = field_schema["$defs"]
        if "definitions" in field_schema:
            sub_schema["definitions"] = field_schema["definitions"]

        # Description: tell the model exactly what sub-section to extract
        sub_description = (
            f"{field_description}\n\n"
            f"--- FOCUS ONLY ON: {prop_name} ---\n"
            f"Extract ONLY the '{prop_name}' section. "
            f"Return a JSON object with a single key '{prop_name}'."
        )

        call_key = f"{field_name}.{prop_name}"
        results.append((
            call_key,
            DocType(
                keys=(KeySpec(group=field_group, key=call_key, description=sub_description),),
                schema=sub_schema,
            )
        ))

    return results
