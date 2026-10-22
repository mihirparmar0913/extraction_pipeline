# Lease Extraction Pipeline

Turns a lease PDF + a `document_types.json` definition into structured JSON —
accurately, cost-efficiently, and consistently.

## Quick start

```bash
pip install -r requirements.txt

# ── NeuralWatt / any OpenAI-compatible provider ──────────────────────────────
export OPENAI_COMPAT_API_KEY="sk-your-key"
export OPENAI_COMPAT_BASE_URL="https://api.neuralwatt.com/v1"   # default

# ── or Anthropic ──────────────────────────────────────────────────────────────
export ANTHROPIC_API_KEY="sk-ant-..."
```

> **Note:** `document_types.json` and real lease PDFs are not bundled in this
> repository (they contain proprietary data). Provide your own
> `document_types.json` and PDF files.

```bash
# List available document types in your file
python3 -m pipeline list-types --doc-types document_types.json

# Extract from a PDF using the new format
python3 -m pipeline extract \
    --pdf your-lease.pdf \
    --doc-types document_types.json \
    --doc-type "Notice: Rent Change" \
    --out results.json

# Preview prompt + token estimate — no API call needed
python3 -m pipeline extract \
    --pdf your-lease.pdf \
    --doc-types document_types.json \
    --doc-type "Lease: Residential (AST)" \
    --dry-run

# Generate a small synthetic lease PDF for a quick demo (no real data needed)
python3 examples/make_sample_pdf.py sample_lease.pdf

# Old format (keys.json + schema.json) still works
python3 -m pipeline extract \
    --pdf sample_lease.pdf \
    --keys examples/keys.json --schema examples/schema.json

# Run all 23 unit tests — no network needed (LLM is mocked)
python3 -m pytest
```

Output JSON → stdout (or `--out file`).  Warnings and token report → stderr.  
Exit codes: `0` ok · `1` bad input / API error · `2` schema-invalid after retries.

## Provider support

| Provider | Client class | How to select |
|---|---|---|
| NeuralWatt / any OpenAI-compatible | `OpenAICompatClient` | Default. Set `OPENAI_COMPAT_API_KEY` + `OPENAI_COMPAT_BASE_URL` |
| Groq, Together, local Ollama | `OpenAICompatClient` | Set `OPENAI_COMPAT_BASE_URL` to provider URL |
| Anthropic Claude | `AnthropicClient` | Set `ANTHROPIC_API_KEY`; pass `--no-split` |

## Design

```
PDF → pdf_loader → prompt_builder → llm_client → normalizer → validator → JSON
        (text)      (per-group)     (JSON mode)   (code only)  (schema+quotes)
                        ↑
                   doctype_loader  ← document_types.json
                   (splits fields into simple + complex sub-calls)
```

| Module | Responsibility |
|---|---|
| `pdf_loader` | Extract position-sorted text per page; render image only for scanned pages |
| `doctype_loader` | Load `document_types.json`; split fields into simple/complex groups |
| `doctype` | KeySpec + DocType dataclasses; fingerprint for cache keys |
| `prompt_builder` | Build message blocks; long field descriptions get their own block |
| `llm_client` | `OpenAICompatClient` (JSON-mode) + `AnthropicClient` (tool-use) |
| `normalizer` | Deterministic: ISO dates, plain numbers, whitespace, key order |
| `validator` | JSON-schema check + evidence-quote anti-hallucination check |
| `cache` | Disk cache keyed by PDF hash + doctype fingerprint + prompt version |
| `extractor` | Orchestration, retry-with-error-feedback, loud failure on max retries |
| `consistency` | N-run agreement measurement (field-level stability report) |

## How cost is reduced

Baseline cost ≈ `keys × pages × tokens/page`. Here:

1. **Simple fields in one call** — `effective_date`, `property_address`, `landlord_name`, etc. are extracted together, not one call per key.
2. **Complex fields split by sub-property** — `consideration` has 3 sub-arrays (consideration, rent_model_terms, contingent_consideration); each gets one focused call so output stays small and accurate.
3. **Text instead of images** — page images only used for scanned/text-poor pages.
4. **Prompt caching** (Anthropic) — document placed first and marked cacheable; retries read it at discount.
5. **Result cache** — same PDF + same doctype = 0 API tokens on re-run.

**Example measurement (1-page rent-increase notice, 7 fields):**
```
LLM calls: 6  |  Input: 8,882 tokens  |  Output: 18,490 tokens
Baseline estimate (per-key): ~13,300 tokens → this run is ~33% smaller
```

## How consistency is achieved

- **Temperature 0** — deterministic sampling.
- **Structured JSON output** — model instructed to return only a JSON object matching the exact schema; no free-text parsing.
- **Deterministic inputs** — same PDF → same extracted text (position-sorted), fixed key order, versioned prompt (`PROMPT_VERSION`).
- **Focused sub-calls** — complex field sub-arrays extracted independently; can't contaminate each other.
- **Normalizer in code** — removes formatting differences (date formats, `"GBP 48,000"` vs `48000`, whitespace, key order).
- **Result cache** — identical input → byte-identical output.
- **Consistency command** — `python3 -m pipeline consistency --runs N` measures per-field agreement across N fresh runs.

## Accuracy safeguards

- Output validated against the per-field JSON schema; on failure the model gets the error list and one retry.
- Evidence: model must give a `page` + verbatim `quote` per field; code checks the quote appears in the PDF (hallucination detection → warning).
- Scanned pages sent as images so they aren't silently lost.
- Explicit rule for missing values (`null`) so the model doesn't improvise.

## Sample extraction output

Example output produced by running the pipeline against a 1-page rent-increase
notice with the `Notice: Rent Change` document type:

```json
{
  "effective_date": "2025-10-01",
  "property_address": "Flat 2, 5 Nugent Terrace, NE8 9QB",
  "landlord_name": "Azil Capital Limited",
  "tenant_name": "Sasha and Talia",
  "Consideration": {
    "consideration": [{
      "consideration_type": "rent",
      "payment_amount": 2415,
      "currency": "GBP",
      "amount_basis": "monthly",
      "consideration_change_type": "variation",
      "payment_start_date": "2025-10-01"
    }],
    "rent_model_terms": [],
    "contingent_consideration": []
  },
  "key_dates": {
    "key_dates": [
      {"date_value": "2025-10-09", "date_type": "document_date"},
      {"date_value": "2025-10-01", "date_type": "effective_date"},
      {"date_value": "2025-10-01", "date_type": "other"}
    ],
    "term": {"term_period": "", "is_fixed_term": false}
  }
}
```

## Token estimates (dry-run, no API needed)

| PDF | Pages | Keys | Est. input tokens | Baseline |
|---|---|---|---|---|
| Rent Increase Notice | 1 | 7 | ~8,882 (measured) | ~13,300 |
| AST Residential | 9 | 8 | ~30,759 | ~111,200 |
| Commercial Lease | 70 | 8 | ~179,385 | ~843,200 |

## Tradeoffs & limitations

- Complex field extraction prompts are very large (20–85K chars each); included verbatim so all rules are preserved.
- Whole-document context per call; very long leases (>~150K tokens total) hit a clear "too large" guard. Chunking / page-selection not implemented.
- Tables extracted as plain text; complex tables may lose structure.
- Evidence checks cover text pages only (not scanned images).
- Cost numbers above are estimates; real numbers from `cost.py` are printed after each run.

## Next steps

1. Hand-label gold set from the sample leases; report field-level precision/recall.
2. Page-selection (BM25 / cheap first-pass) for very long commercial leases.
3. Majority-vote over N runs for low-confidence fields.
4. Table-aware extraction (`PyMuPDF find_tables`) + OCR fallback for scans.
5. Streaming output for large documents to reduce time-to-first-token.
