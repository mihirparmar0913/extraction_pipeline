"""Command line interface.

  # New format (document_types.json):
  python3 -m pipeline extract --pdf "rent increase 2025-2026 NT2.pdf" \
      --doc-types document_types.json --doc-type "Notice: Rent Change"

  python3 -m pipeline extract --pdf "AST nt 2.pdf" \
      --doc-types document_types.json --doc-type "Lease: Residential (AST)"

  python3 -m pipeline extract \
      --pdf "2025.08.22 - 4th Fl, RSH - Lease - Signed by Landlord.pdf" \
      --doc-types document_types.json --doc-type "Lease: Commercial"

  # Old format (separate keys + schema files):
  python3 -m pipeline extract --pdf lease.pdf --keys keys.json --schema schema.json

  # Other commands:
  python3 -m pipeline list-types  --doc-types document_types.json
  python3 -m pipeline consistency --pdf lease.pdf \
      --doc-types document_types.json --doc-type "Lease: Commercial" --runs 3

Add --dry-run to preview the prompt without calling the API.
Add --no-split to force everything into a single LLM call.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from . import cost
from .cache import DiskCache
from .consistency import run_consistency
from .doctype import DocType, DocTypeError, load_doctype
from .doctype_loader import (
    list_document_type_names,
    load_doctype_from_document_types,
    split_complex_field_into_sub_calls,
    split_into_field_groups,
)
from .extractor import ExtractionError, ExtractionResult, Extractor
from .llm_client import LLMError, LLMResult, OpenAICompatClient, Usage
from .pdf_loader import PDFLoadError, load_pdf
from .prompt_builder import SYSTEM_PROMPT, build_blocks, build_response_schema

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------

def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--pdf", required=True)
    common.add_argument("--keys",       default=None, help="keys JSON (old format)")
    common.add_argument("--schema",     default=None, help="output schema JSON (old format)")
    common.add_argument("--doc-types",  default=None, help="path to document_types.json")
    common.add_argument("--doc-type",   default=None, help="name of the document type")
    common.add_argument("--model",      default=None, help="override model (default: glm-5.2)")
    common.add_argument("--cache-dir",  default=".cache/extractions")
    common.add_argument("--no-split",   action="store_true",
                        help="Force all fields into ONE call (not recommended for complex types)")

    parser = argparse.ArgumentParser(prog="pipeline", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    ex = sub.add_parser("extract", parents=[common])
    ex.add_argument("--out",      help="write JSON here instead of stdout")
    ex.add_argument("--no-cache", action="store_true")
    ex.add_argument("--dry-run",  action="store_true",
                    help="print prompt + token estimate, no API call")

    co = sub.add_parser("consistency", parents=[common])
    co.add_argument("--runs", type=int, default=3)

    lt = sub.add_parser("list-types")
    lt.add_argument("--doc-types", required=True)

    return parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client(args) -> OpenAICompatClient:
    return OpenAICompatClient(model=args.model)


def _resolve_doctype(args) -> DocType:
    if args.doc_types and args.doc_type:
        return load_doctype_from_document_types(args.doc_types, args.doc_type)
    elif args.keys and args.schema:
        return load_doctype(args.keys, args.schema)
    raise DocTypeError(
        "Provide --doc-types + --doc-type (new format) or --keys + --schema (old format)"
    )


def _dry_run(args, doctype: DocType) -> None:
    pdf = load_pdf(args.pdf)
    blocks = build_blocks(pdf, doctype)
    for b in blocks:
        if b["type"] == "text":
            text = b["text"]
            if len(text) > 2000:
                print(text[:1000])
                print(f"\n... [{len(text) - 2000} chars truncated] ...\n")
                print(text[-1000:])
            else:
                print(text)
        else:
            print(f"<image: {len(b['data'])} bytes PNG>")
        print()
    instruction_chars = sum(len(b["text"]) for b in blocks if b["type"] == "text")
    est  = cost.estimate_input_tokens(pdf, prompt_chars=instruction_chars)
    base = cost.estimate_baseline_input_tokens(len(doctype.keys), len(pdf.pages))
    print(f"[dry-run] pages={len(pdf.pages)} scanned={pdf.scanned_pages} keys={len(doctype.keys)}")
    print(f"[dry-run] ~{est} input tokens (1 call) vs ~{base} estimated baseline")


# ---------------------------------------------------------------------------
# Direct LLM call (no evidence wrapper) — used for complex sub-field calls
# ---------------------------------------------------------------------------

def _direct_llm_call(
    llm: OpenAICompatClient,
    pdf_path: str,
    sub_doctype: DocType,
    field_label: str,
) -> tuple[dict, Usage]:
    """Call the LLM directly with a sub-field schema, NO evidence wrapper.

    Returns (data_dict, usage).  Does NOT go through Extractor so there
    is no evidence-wrapping — this keeps output tokens small.
    """
    pdf = load_pdf(pdf_path)
    blocks = build_blocks(pdf, sub_doctype)
    # Use the schema directly (no evidence wrapper) to minimise output tokens
    schema = sub_doctype.schema
    try:
        result: LLMResult = llm.extract(
            system=SYSTEM_PROMPT, blocks=blocks, tool_schema=schema
        )
    except LLMError:
        raise
    return result.data, result.usage


# ---------------------------------------------------------------------------
# Split-mode extraction
# ---------------------------------------------------------------------------

def _extract_split(args, pdf_path: str, cache: DiskCache | None) -> ExtractionResult:
    """Split extraction:
      1. All simple fields in ONE call (uses full Extractor with evidence + retry).
      2. Each complex field split further into per-sub-property calls
         (direct LLM calls, no evidence wrapper → small output per call).
    Results are merged into a single dict.
    """
    pdf = load_pdf(pdf_path)
    num_pages = len(pdf.pages)
    llm = _make_client(args)

    combined_data:     dict       = {}
    combined_warnings: list[str]  = []
    combined_usage:    Usage      = Usage()
    total_attempts:    int        = 0

    # ---- 1. Simple fields -----------------------------------------------
    simple_dt, complex_field_specs = split_into_field_groups(
        args.doc_types, args.doc_type
    )

    if simple_dt:
        log.info("Extracting %d simple fields in one call...", len(simple_dt.keys))
        extractor = Extractor(llm, cache)
        result = extractor.extract(pdf_path, simple_dt)
        combined_data.update(result.data)
        combined_warnings.extend(result.warnings)
        combined_usage  = combined_usage + result.usage
        total_attempts += result.attempts

    # ---- 2. Complex fields (sub-call per top-level property) ------------
    for field_spec in complex_field_specs:
        field_name = field_spec["name"]
        sub_calls = split_complex_field_into_sub_calls(field_spec)

        if len(sub_calls) == 1:
            # Single sub-call — field schema has one top-level property
            call_key, sub_dt = sub_calls[0]
            log.info("Extracting complex field '%s'...", field_name)
            data, usage = _direct_llm_call(llm, pdf_path, sub_dt, field_name)
            combined_data[field_name] = data
            combined_usage = combined_usage + usage
            total_attempts += 1
        else:
            # Multiple sub-calls — merge sub-results under field_name
            log.info(
                "Extracting complex field '%s' in %d sub-calls: %s",
                field_name, len(sub_calls),
                [k.split(".")[-1] for k, _ in sub_calls]
            )
            field_data: dict = {}
            for call_key, sub_dt in sub_calls:
                prop_name = call_key.split(".")[-1]
                log.info("  Sub-call: %s.%s", field_name, prop_name)
                data, usage = _direct_llm_call(llm, pdf_path, sub_dt, call_key)
                # data is {"prop_name": <value>} — extract the value
                if prop_name in data:
                    field_data[prop_name] = data[prop_name]
                else:
                    # fallback: merge whatever was returned
                    field_data.update(data)
                combined_usage = combined_usage + usage
                total_attempts += 1
            combined_data[field_name] = field_data

    return ExtractionResult(
        data=combined_data,
        warnings=combined_warnings,
        usage=combined_usage,
        attempts=total_attempts,
        from_cache=False,
        num_pages=num_pages,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr
    )
    try:
        if args.command == "list-types":
            for name in list_document_type_names(args.doc_types):
                print(f"  - {name}")
            return 0

        doctype = _resolve_doctype(args)

        if getattr(args, "dry_run", False):
            _dry_run(args, doctype)
            return 0

        use_cache  = args.command == "extract" and not getattr(args, "no_cache", False)
        cache      = DiskCache(args.cache_dir) if use_cache else None
        use_split  = (
            args.doc_types and args.doc_type
            and not getattr(args, "no_split", False)
            and args.command == "extract"
        )

        if use_split:
            result = _extract_split(args, args.pdf, cache)
        else:
            llm       = _make_client(args)
            extractor = Extractor(llm, cache)
            if args.command == "consistency":
                report = run_consistency(extractor, args.pdf, doctype, args.runs)
                print(json.dumps(report, indent=2))
                return 0
            result = extractor.extract(args.pdf, doctype)

        text = json.dumps(result.data, indent=2, ensure_ascii=False)
        if getattr(args, "out", None):
            with open(args.out, "w", encoding="utf-8") as fh:
                fh.write(text + "\n")
            print(f"Output written to {args.out}", file=sys.stderr)
        else:
            print(text)

        for w in result.warnings:
            print(f"WARNING: {w}", file=sys.stderr)

        print(
            cost.format_report(
                result.usage, len(doctype.keys),
                result.num_pages, result.attempts, result.from_cache,
            ),
            file=sys.stderr,
        )
        return 0

    except (DocTypeError, PDFLoadError, LLMError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except ExtractionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        for e in exc.errors:
            print(f"  - {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
