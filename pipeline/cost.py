"""Token accounting + a rough estimate of what the old per-key baseline would cost."""
from __future__ import annotations

from .llm_client import Usage
from .pdf_loader import LoadedPDF

TOKENS_PER_IMAGE_PAGE = 1500  # rough average for a full page image
CHARS_PER_TOKEN = 4  # rough average for English text
PER_KEY_PROMPT_TOKENS = 400  # key prompt + key schema + instructions in the baseline


def estimate_input_tokens(pdf: LoadedPDF, prompt_chars: int = 0) -> int:
    text_chars = sum(len(p.text) for p in pdf.pages)
    return (text_chars + prompt_chars) // CHARS_PER_TOKEN + TOKENS_PER_IMAGE_PAGE * len(pdf.scanned_pages)


def estimate_baseline_input_tokens(num_keys: int, num_pages: int) -> int:
    """Baseline = every key re-sends every page as an image."""
    return num_keys * (num_pages * TOKENS_PER_IMAGE_PAGE + PER_KEY_PROMPT_TOKENS)


def format_report(usage: Usage, num_keys: int, num_pages: int, attempts: int, from_cache: bool) -> str:
    baseline = estimate_baseline_input_tokens(num_keys, num_pages)
    lines = [f"Pages: {num_pages} | Keys: {num_keys}"]
    if from_cache:
        lines.append("Result served from cache: 0 LLM calls, 0 tokens.")
    else:
        saving = 100 * (1 - usage.total_input / baseline) if baseline else 0
        lines += [
            f"LLM calls: {attempts}",
            f"Input tokens: {usage.total_input} (fresh {usage.input_tokens}, "
            f"cache-write {usage.cache_write_tokens}, cache-read {usage.cache_read_tokens}) | "
            f"output tokens: {usage.output_tokens}",
        ]
        lines.append(
            f"Estimated baseline input (per-key, images): ~{baseline} tokens "
            f"-> this run is ~{saving:.0f}% smaller (estimate, not measured)"
        )
    return "\n".join(lines)
