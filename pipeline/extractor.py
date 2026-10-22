"""Step 7 - orchestrate: load -> (cache?) -> ONE LLM call -> normalize -> validate -> retry."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from . import cost
from .cache import DiskCache, make_cache_key
from .doctype import DocType
from .llm_client import LLMClient, LLMError, Usage
from .normalizer import normalize
from .pdf_loader import load_pdf
from .prompt_builder import PROMPT_VERSION, SYSTEM_PROMPT, build_blocks, build_response_schema
from .validator import validate_against_schema, verify_evidence

log = logging.getLogger(__name__)


class ExtractionError(Exception):
    """Extraction could not produce schema-valid output."""

    def __init__(self, message: str, errors: list[str] | None = None, last_output: dict | None = None):
        super().__init__(message)
        self.errors = errors or []
        self.last_output = last_output


@dataclass
class ExtractionResult:
    data: dict
    warnings: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    attempts: int = 0
    from_cache: bool = False
    num_pages: int = 0


class Extractor:
    def __init__(
        self,
        llm: LLMClient,
        cache: DiskCache | None = None,
        *,
        max_attempts: int = 2,
        max_input_tokens: int = 150_000,
    ):
        self._llm = llm
        self._cache = cache
        self._max_attempts = max_attempts
        self._max_input_tokens = max_input_tokens

    def extract(self, pdf_path: str, doctype: DocType) -> ExtractionResult:
        pdf = load_pdf(pdf_path)
        num_pages = len(pdf.pages)

        cache_key = make_cache_key(pdf.sha256, doctype.fingerprint, PROMPT_VERSION, self._llm.model)
        if self._cache and (hit := self._cache.get(cache_key)):
            log.info("Cache hit - no LLM call")
            return ExtractionResult(hit["data"], hit["warnings"], from_cache=True, num_pages=num_pages)

        estimate = cost.estimate_input_tokens(pdf, prompt_chars=3000)
        if estimate > self._max_input_tokens:
            raise ExtractionError(
                f"Document is too large for a single call (~{estimate} tokens > "
                f"{self._max_input_tokens}). Page-selection / chunking is not implemented yet."
            )

        tool_schema = build_response_schema(doctype.schema)
        usage, correction, last_output, errors = Usage(), None, None, []

        for attempt in range(1, self._max_attempts + 1):
            blocks = build_blocks(pdf, doctype, correction)
            try:
                result = self._llm.extract(system=SYSTEM_PROMPT, blocks=blocks, tool_schema=tool_schema)
            except LLMError as exc:
                raise ExtractionError(f"LLM call failed: {exc}") from exc
            usage = usage + result.usage

            extraction = result.data.get("extraction")
            last_output = result.data
            if not isinstance(extraction, dict):
                errors = ["response has no 'extraction' object"]
            else:
                extraction = normalize(extraction, doctype.schema)
                errors = validate_against_schema(extraction, doctype.schema)
            if not errors:
                warnings = verify_evidence(result.data.get("evidence", []), pdf)
                for w in warnings:
                    log.warning("evidence check: %s", w)
                if self._cache:
                    self._cache.put(cache_key, {"data": extraction, "warnings": warnings})
                return ExtractionResult(extraction, warnings, usage, attempt, False, num_pages)

            log.warning("Attempt %d failed schema validation: %s", attempt, errors)
            correction = (
                "Your previous answer did not match the schema. Problems:\n- "
                + "\n- ".join(errors[:15])
                + "\nReturn a corrected answer."
            )

        raise ExtractionError(
            f"Output still invalid after {self._max_attempts} attempts", errors, last_output
        )
