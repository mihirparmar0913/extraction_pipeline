from pipeline.pdf_loader import load_pdf
from pipeline.validator import validate_against_schema, verify_evidence


def test_schema_errors_are_reported(doctype, good_response):
    bad = good_response["extraction"]
    bad["lease"]["annual_rent"] = "lots"
    errors = validate_against_schema(bad, doctype.schema)
    assert any("lease/annual_rent" in e for e in errors)
    assert validate_against_schema(good_response["extraction"] | {"lease": {**bad["lease"], "annual_rent": 1}},
                                   doctype.schema) == []


def test_evidence_verified(pdf_path):
    pdf = load_pdf(pdf_path)
    ok = [{"key": "k", "page": 1, "quote": "greenfield   properties limited"}]  # case/space insensitive
    assert verify_evidence(ok, pdf) == []


def test_evidence_hallucination_and_wrong_page_flagged(pdf_path):
    pdf = load_pdf(pdf_path)
    warnings = verify_evidence(
        [
            {"key": "a", "page": 1, "quote": "rent is one million pounds"},
            {"key": "b", "page": 1, "quote": "The annual rent is GBP 48,000"},  # really on page 2
        ],
        pdf,
    )
    assert "hallucination" in warnings[0]
    assert "not on cited page" in warnings[1]
