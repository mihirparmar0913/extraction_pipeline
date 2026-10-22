"""Step 3 - build ONE prompt that covers every key.

Layout matters for cost: the (large) document comes first and is marked
cacheable; the key instructions come after it. If we ever need a retry, the
document prefix is read from the provider's prompt cache at a fraction of the price.

Blocks are provider-neutral dicts; llm_client.py converts them to the API format.

For real document types, complex fields (consideration, key_dates, etc.) have
very long descriptions (tens of thousands of chars) that are full extraction
prompts with detailed rules. These are included as separate instruction blocks
rather than inlined on a single line.
"""
from __future__ import annotations

import copy

from .doctype import DocType
from .pdf_loader import LoadedPDF

PROMPT_VERSION = "v2"  # bumped: changed prompt layout for real document types

SYSTEM_PROMPT = (
    "You are a precise data-extraction engine for property lease documents. "
    "You only report what the document explicitly states. You never guess, infer "
    "from outside knowledge, or invent values."
)

EVIDENCE_SCHEMA = {
    "type": "array",
    "description": "Proof for each extracted value. Fill this BEFORE 'extraction'.",
    "items": {
        "type": "object",
        "properties": {
            "key": {"type": "string"},
            "page": {"type": "integer", "description": "1-based page number"},
            "quote": {
                "type": "string",
                "description": "Short verbatim quote (max ~15 words) copied exactly from that page",
            },
        },
        "required": ["key", "page", "quote"],
    },
}

# Threshold: descriptions longer than this get their own instruction block
_LONG_DESCRIPTION_THRESHOLD = 500


def build_response_schema(output_schema: dict) -> dict:
    """Wrap the customer's schema so the model returns evidence + extraction together.

    The customer's schema is embedded untouched (apart from hoisting $defs so that
    '#/$defs/..' references still resolve from the new root).
    """
    inner = copy.deepcopy(output_schema)
    inner.pop("$schema", None)
    inner.pop("$id", None)
    wrapper: dict = {
        "type": "object",
        "properties": {"evidence": EVIDENCE_SCHEMA, "extraction": inner},
        "required": ["evidence", "extraction"],
    }
    for defs_key in ("$defs", "definitions"):
        if defs_key in inner:
            wrapper[defs_key] = inner.pop(defs_key)
    return wrapper


def _instructions(doctype: DocType) -> list[str]:
    """Build instruction text blocks.

    Returns a list of text blocks. Simple fields are grouped together.
    Complex fields with long descriptions get their own separate block each.
    """
    simple_lines = ["Extract the fields listed below from the lease document above.", "", "FIELDS"]
    long_blocks: list[str] = []

    current_group = None
    for spec in doctype.keys:
        if spec.group != current_group:
            current_group = spec.group
            simple_lines.append(f"[{current_group}]")

        if len(spec.description) > _LONG_DESCRIPTION_THRESHOLD:
            # Complex field: provide its extraction prompt as a separate block
            long_blocks.append(
                f"=== EXTRACTION INSTRUCTIONS FOR FIELD: {spec.key} ===\n\n"
                f"{spec.description}\n\n"
                f"=== END OF INSTRUCTIONS FOR: {spec.key} ==="
            )
            simple_lines.append(f"- {spec.key}: (see detailed instructions below)")
        else:
            simple_lines.append(f"- {spec.key}: {spec.description}")

    simple_lines += [
        "",
        "RULES",
        "1. Use ONLY information stated in the document. Do not guess.",
        "2. If a value is not stated, use null where the schema allows it; otherwise use the "
        "schema's empty value (empty list / empty string).",
        "3. Dates: YYYY-MM-DD. Numbers: plain digits without currency symbols or thousands separators.",
        "4. Copy names and addresses exactly as written in the document.",
        "5. Fill 'evidence' first: for every field you extract give the key, the page number "
        "and a short verbatim quote. Then fill 'extraction' so it matches the schema.",
        "6. Fields may appear several times; prefer the most specific / latest operative clause.",
    ]

    # Return: simple fields overview first, then each complex field's instructions
    blocks = ["\n".join(simple_lines)]
    blocks.extend(long_blocks)
    return blocks


def build_blocks(pdf: LoadedPDF, doctype: DocType, correction: str | None = None) -> list[dict]:
    blocks: list[dict] = [{"type": "text", "text": "LEASE DOCUMENT (page by page)"}]
    for page in pdf.pages:
        if page.is_scanned:
            header = f"--- PAGE {page.number} (scanned image; text below may be empty) ---"
            blocks.append({"type": "text", "text": f"{header}\n{page.text}".strip()})
            blocks.append({"type": "image", "media_type": "image/png", "data": page.image_png})
        else:
            blocks.append({"type": "text", "text": f"--- PAGE {page.number} ---\n{page.text}"})
    blocks.append({"type": "text", "text": "--- END OF DOCUMENT ---", "cache": True})

    # Add instruction blocks
    instruction_blocks = _instructions(doctype)
    for text in instruction_blocks:
        blocks.append({"type": "text", "text": text})

    if correction:
        blocks.append({"type": "text", "text": correction})
    return blocks
