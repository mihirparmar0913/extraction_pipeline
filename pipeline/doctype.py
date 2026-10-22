"""Step 2 - load the 'document type': keys + descriptions and the output JSON schema."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from jsonschema.validators import validator_for


class DocTypeError(Exception):
    """The keys file or schema is malformed."""


@dataclass(frozen=True)
class KeySpec:
    group: str
    key: str
    description: str


@dataclass(frozen=True)
class DocType:
    keys: tuple[KeySpec, ...]
    schema: dict

    @property
    def fingerprint(self) -> str:
        """Stable hash of everything that defines what we extract (used in cache keys)."""
        payload = {
            "keys": [(k.group, k.key, k.description) for k in self.keys],
            "schema": self.schema,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _read_json(path: str | Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DocTypeError(f"File not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DocTypeError(f"Invalid JSON in {path}: {exc}") from exc


def parse_keys(raw) -> tuple[KeySpec, ...]:
    """Accepts {"group": [{"Key":..,"Description":..}]} or a flat list of such objects."""
    grouped = raw if isinstance(raw, dict) else {"default": raw}
    specs: list[KeySpec] = []
    for group, items in grouped.items():
        if not isinstance(items, list):
            raise DocTypeError(f"Group '{group}' must be a list of key objects")
        for item in items:
            key, desc = item.get("Key"), item.get("Description")
            if not key or not desc:
                raise DocTypeError(f"Each key needs 'Key' and 'Description': {item}")
            specs.append(KeySpec(group, key, desc))
    if not specs:
        raise DocTypeError("The keys file contains no keys")
    return tuple(specs)


def load_doctype(keys_path: str | Path, schema_path: str | Path) -> DocType:
    keys = parse_keys(_read_json(keys_path))
    schema = _read_json(schema_path)
    try:
        validator_for(schema).check_schema(schema)
    except Exception as exc:
        raise DocTypeError(f"Output schema is not a valid JSON Schema: {exc}") from exc
    return DocType(keys, schema)
