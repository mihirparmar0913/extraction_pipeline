import pytest
import pymupdf

from pipeline.pdf_loader import PDFLoadError, load_pdf


def test_text_pages_have_no_image(pdf_path):
    pdf = load_pdf(pdf_path)
    assert len(pdf.pages) == 2 and pdf.scanned_pages == []
    assert "Riverside Business Park" in pdf.pages[0].text


def test_blank_page_is_treated_as_scanned(tmp_path):
    doc = pymupdf.open()
    doc.new_page()
    path = tmp_path / "blank.pdf"
    doc.save(path)
    pdf = load_pdf(path)
    assert pdf.scanned_pages == [1] and pdf.pages[0].image_png.startswith(b"\x89PNG")


def test_missing_and_corrupt_files(tmp_path):
    with pytest.raises(PDFLoadError):
        load_pdf(tmp_path / "nope.pdf")
    bad = tmp_path / "bad.pdf"
    bad.write_text("not a pdf")
    with pytest.raises(PDFLoadError):
        load_pdf(bad)


def test_same_file_same_hash_and_text(pdf_path):
    a, b = load_pdf(pdf_path), load_pdf(pdf_path)
    assert a.sha256 == b.sha256 and [p.text for p in a.pages] == [p.text for p in b.pages]
