"""Step 6 - check the result: (a) matches the JSON schema, (b) quotes are really in the PDF."""
from __future__ import annotations

import re

from jsonschema.validators import validator_for

from .pdf_loader import LoadedPDF


def validate_against_schema(data, schema: dict) -> list[str]:
    """Return human-readable schema violations (empty list = valid)."""
    validator = validator_for(schema)(schema)
    errors = sorted(validator.iter_errors(data), key=lambda e: list(e.absolute_path))
    return [
        f"{'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}" for e in errors
    ]


def _norm(text: str) -> str:
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", " ", text).strip().lower()


def verify_evidence(evidence: list[dict], pdf: LoadedPDF) -> list[str]:
    """Anti-hallucination check. Every quote must literally appear in the document.

    Scanned pages have no text layer, so quotes on them cannot be checked.
    """
    page_text = {p.number: _norm(p.text) for p in pdf.pages}
    all_text = " ".join(page_text.values())
    scanned = set(pdf.scanned_pages)
    warnings: list[str] = []
    for item in evidence:
        key, page, quote = item.get("key"), item.get("page"), _norm(item.get("quote", ""))
        if not quote or page in scanned:
            continue
        if quote in page_text.get(page, ""):
            continue
        if quote in all_text:
            warnings.append(f"'{key}': quote found, but not on cited page {page}")
        else:
            warnings.append(f"'{key}': quote not found in document -> possible hallucination")
    return warnings
