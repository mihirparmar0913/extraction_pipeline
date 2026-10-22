"""Measure consistency: run the same extraction N times (cache OFF) and compare field by field."""
from __future__ import annotations

import json
from collections import Counter

from .doctype import DocType
from .extractor import Extractor


def flatten(obj, prefix: str = "") -> dict[str, str]:
    """{'a': {'b': 1}, 'c': [2]} -> {'a.b': '1', 'c[0]': '2'}"""
    out: dict[str, str] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(flatten(v, f"{prefix}[{i}]"))
        if not obj:
            out[prefix] = "[]"
    else:
        out[prefix] = json.dumps(obj, sort_keys=True)
    return out


def agreement(runs: list[dict]) -> dict[str, float]:
    """Per-field share of runs that agree with the most common value (1.0 = perfectly stable)."""
    flat = [flatten(r) for r in runs]
    paths = sorted({p for f in flat for p in f})
    return {
        p: Counter(f.get(p, "<missing>") for f in flat).most_common(1)[0][1] / len(runs) for p in paths
    }


def run_consistency(extractor: Extractor, pdf_path: str, doctype: DocType, runs: int) -> dict:
    outputs = [extractor.extract(pdf_path, doctype).data for _ in range(runs)]
    scores = agreement(outputs)
    unstable = {p: s for p, s in scores.items() if s < 1.0}
    return {
        "runs": runs,
        "fields": len(scores),
        "fully_stable_fields": len(scores) - len(unstable),
        "overall_agreement": sum(scores.values()) / len(scores) if scores else 1.0,
        "unstable": unstable,
    }
