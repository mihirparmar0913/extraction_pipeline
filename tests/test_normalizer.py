from pipeline.normalizer import normalize

SCHEMA = {
    "type": "object",
    "properties": {
        "start": {"type": ["string", "null"], "format": "date"},
        "rent": {"type": ["number", "null"]},
        "name": {"type": "string"},
    },
}


def test_dates_become_iso_uk_style():
    assert normalize({"start": "1 April 2024"}, SCHEMA)["start"] == "2024-04-01"
    assert normalize({"start": "01/04/2024"}, SCHEMA)["start"] == "2024-04-01"  # day first


def test_currency_string_becomes_number():
    assert normalize({"rent": "£48,000.50"}, SCHEMA)["rent"] == 48000.5
    assert normalize({"rent": "£48,000"}, SCHEMA)["rent"] == 48000


def test_whitespace_cleaned_and_keys_follow_schema_order():
    out = normalize({"name": "  Acme \n  Ltd ", "rent": 1, "start": None}, SCHEMA)
    assert out["name"] == "Acme Ltd"
    assert list(out) == ["start", "rent", "name"]


def test_unparseable_values_are_left_alone():
    assert normalize({"start": "sometime soon"}, SCHEMA)["start"] == "sometime soon"


def test_refs_are_followed():
    schema = {"type": "object", "properties": {"a": {"$ref": "#/$defs/D"}},
              "$defs": {"D": {"type": "string", "format": "date"}}}
    assert normalize({"a": "2 May 2023"}, schema)["a"] == "2023-05-02"
