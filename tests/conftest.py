import json
from pathlib import Path

import pytest

from examples.make_sample_pdf import make_sample_pdf
from pipeline.doctype import load_doctype
from pipeline.llm_client import LLMResult, Usage

EXAMPLES = Path(__file__).parent.parent / "examples"

GOOD_RESPONSE = {
    "evidence": [
        {"key": "landlord_name", "page": 1, "quote": "GREENFIELD PROPERTIES LIMITED"},
        {"key": "unit_postcode", "page": 1, "quote": "Leeds, LS1 4AB"},
        {"key": "annual_rent", "page": 2, "quote": "The annual rent is GBP 48,000 per annum"},
    ],
    "extraction": {
        "parties": {"landlord_name": "Greenfield Properties Limited", "tenant_name": "Brightway Retail Ltd"},
        "unit": {"unit_address": "Unit 4, Riverside Business Park, Leeds", "unit_postcode": "LS1 4AB"},
        "lease": {
            "lease_start_date": "2024-04-01",
            "lease_end_date": "2029-03-31",
            "annual_rent": 48000,
            "has_break_clause": True,
        },
    },
}


class FakeLLM:
    """Returns scripted responses and records every call (no network needed)."""

    model = "fake-model"

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def extract(self, *, system, blocks, tool_schema):
        self.calls.append({"system": system, "blocks": blocks, "tool_schema": tool_schema})
        return LLMResult(json.loads(json.dumps(self.responses.pop(0))), Usage(input_tokens=1000, output_tokens=200))


@pytest.fixture
def pdf_path(tmp_path):
    path = tmp_path / "lease.pdf"
    make_sample_pdf(str(path))
    return str(path)


@pytest.fixture
def doctype():
    return load_doctype(EXAMPLES / "keys.json", EXAMPLES / "schema.json")


@pytest.fixture
def good_response():
    return json.loads(json.dumps(GOOD_RESPONSE))
