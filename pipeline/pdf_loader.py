"""Step 1 - turn a PDF into per-page text.

Text is far cheaper (and more stable) for an LLM than page images, so we only
render a page to an image when it has (almost) no extractable text, e.g. scans.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pymupdf  # PyMuPDF


class PDFLoadError(Exception):
    """The PDF is missing, corrupt, encrypted or empty."""


@dataclass(frozen=True)
class Page:
    number: int  # 1-based
    text: str
    image_png: bytes | None = None  # only set for scanned / text-poor pages

    @property
    def is_scanned(self) -> bool:
        return self.image_png is not None


@dataclass(frozen=True)
class LoadedPDF:
    path: str
    sha256: str
    pages: tuple[Page, ...]

    @property
    def scanned_pages(self) -> list[int]:
        return [p.number for p in self.pages if p.is_scanned]


def load_pdf(path: str | Path, *, min_chars: int = 50, image_dpi: int = 110) -> LoadedPDF:
    p = Path(path)
    if not p.is_file():
        raise PDFLoadError(f"PDF not found: {p}")
    data = p.read_bytes()
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises several exception types
        raise PDFLoadError(f"Could not open {p} as a PDF: {exc}") from exc

    with doc:
        if doc.needs_pass:
            raise PDFLoadError(f"{p} is password protected")
        if doc.page_count == 0:
            raise PDFLoadError(f"{p} has no pages")
        pages: list[Page] = []
        for number, page in enumerate(doc, start=1):
            # sort=True orders text blocks by position -> same text on every run
            text = page.get_text("text", sort=True).strip()
            if len(text) >= min_chars:
                pages.append(Page(number, text))
            else:
                png = page.get_pixmap(dpi=image_dpi).tobytes("png")
                pages.append(Page(number, text, png))

    return LoadedPDF(str(p), hashlib.sha256(data).hexdigest(), tuple(pages))
