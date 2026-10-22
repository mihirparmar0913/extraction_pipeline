"""Result cache: same PDF + same document type + same prompt + same model -> same answer.

This is the strongest consistency guarantee: a repeat run never calls the LLM at all.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


def make_cache_key(pdf_sha256: str, doctype_fingerprint: str, prompt_version: str, model: str) -> str:
    raw = "|".join([pdf_sha256, doctype_fingerprint, prompt_version, model])
    return hashlib.sha256(raw.encode()).hexdigest()


class DiskCache:
    def __init__(self, directory: str | Path = ".cache/extractions"):
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> dict | None:
        path = self._dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None  # corrupt entry -> treat as a miss

    def put(self, key: str, value: dict) -> None:
        # write to a temp file then rename, so a crash never leaves half a file
        fd, tmp = tempfile.mkstemp(dir=self._dir, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(value, fh, ensure_ascii=False)
        os.replace(tmp, self._dir / f"{key}.json")
