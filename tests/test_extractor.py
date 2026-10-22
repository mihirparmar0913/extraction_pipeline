import pytest

from pipeline.cache import DiskCache
from pipeline.extractor import ExtractionError, Extractor
from pipeline.prompt_builder import build_response_schema
from tests.conftest import FakeLLM


def test_happy_path_uses_exactly_one_call_for_all_keys(pdf_path, doctype, good_response):
    llm = FakeLLM(good_response)
    result = Extractor(llm).extract(pdf_path, doctype)
    assert len(llm.calls) == 1  # NOT one call per key
    assert result.data["lease"]["annual_rent"] == 48000
    assert result.warnings == []
    prompt = "\n".join(b["text"] for b in llm.calls[0]["blocks"] if b["type"] == "text")
    assert all(k.key in prompt for k in doctype.keys)  # every key is in that single prompt


def test_document_block_is_marked_cacheable(pdf_path, doctype, good_response):
    llm = FakeLLM(good_response)
    Extractor(llm).extract(pdf_path, doctype)
    blocks = llm.calls[0]["blocks"]
    cached = [i for i, b in enumerate(blocks) if b.get("cache")]
    assert cached and cached[0] < len(blocks) - 1  # cache marker sits before the key instructions


def test_invalid_output_is_retried_with_the_error_message(pdf_path, doctype, good_response):
    bad = {"evidence": [], "extraction": {"parties": {}, "unit": {}, "lease": {}}}
    llm = FakeLLM(bad, good_response)
    result = Extractor(llm).extract(pdf_path, doctype)
    assert result.attempts == 2 and len(llm.calls) == 2
    assert "did not match the schema" in llm.calls[1]["blocks"][-1]["text"]
    assert result.usage.input_tokens == 2000  # usage is summed across attempts


def test_gives_up_loudly_when_still_invalid(pdf_path, doctype):
    bad = {"evidence": [], "extraction": {"parties": {}}}
    with pytest.raises(ExtractionError) as exc:
        Extractor(FakeLLM(bad, bad)).extract(pdf_path, doctype)
    assert exc.value.errors


def test_second_run_is_served_from_cache_and_identical(pdf_path, doctype, good_response, tmp_path):
    cache = DiskCache(tmp_path / "cache")
    llm = FakeLLM(good_response)
    first = Extractor(llm, cache).extract(pdf_path, doctype)
    second = Extractor(llm, cache).extract(pdf_path, doctype)
    assert len(llm.calls) == 1 and second.from_cache
    assert first.data == second.data


def test_hallucinated_quote_produces_warning(pdf_path, doctype, good_response):
    good_response["evidence"].append({"key": "tenant_name", "page": 1, "quote": "made up sentence"})
    result = Extractor(FakeLLM(good_response)).extract(pdf_path, doctype)
    assert any("hallucination" in w for w in result.warnings)


def test_oversized_document_fails_clearly(pdf_path, doctype, good_response):
    with pytest.raises(ExtractionError, match="too large"):
        Extractor(FakeLLM(good_response), max_input_tokens=10).extract(pdf_path, doctype)


def test_response_schema_hoists_defs():
    schema = {"type": "object", "properties": {"a": {"$ref": "#/$defs/X"}}, "$defs": {"X": {"type": "string"}}}
    wrapped = build_response_schema(schema)
    assert "$defs" in wrapped and "$defs" not in wrapped["properties"]["extraction"]
