from pipeline.consistency import agreement, flatten, run_consistency
from pipeline.extractor import Extractor
from tests.conftest import FakeLLM


def test_flatten():
    assert flatten({"a": {"b": 1}, "c": [2, 3]}) == {"a.b": "1", "c[0]": "2", "c[1]": "3"}


def test_agreement_detects_unstable_field():
    runs = [{"x": 1, "y": 1}, {"x": 1, "y": 2}, {"x": 1, "y": 1}, {"x": 1, "y": 1}]
    scores = agreement(runs)
    assert scores["x"] == 1.0 and scores["y"] == 0.75


def test_run_consistency_report(pdf_path, doctype, good_response):
    changed = {**good_response, "extraction": {**good_response["extraction"],
               "lease": {**good_response["extraction"]["lease"], "annual_rent": 50000}}}
    llm = FakeLLM(good_response, good_response, changed)
    report = run_consistency(Extractor(llm), pdf_path, doctype, runs=3)
    assert report["runs"] == 3 and "lease.annual_rent" in report["unstable"]
