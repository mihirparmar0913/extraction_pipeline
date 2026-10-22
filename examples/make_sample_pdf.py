"""Create a small fake lease PDF for trying the pipeline:  python examples/make_sample_pdf.py"""
import sys

import pymupdf

PAGE_1 = """COMMERCIAL LEASE AGREEMENT

This lease is made on 1 March 2024 between GREENFIELD PROPERTIES LIMITED
(the Landlord) and BRIGHTWAY RETAIL LTD (the Tenant).

1. THE UNIT
The premises let are Unit 4, Riverside Business Park, Leeds, LS1 4AB.

2. THE TERM
The term is 5 years commencing on 1 April 2024 and expiring on 31 March 2029.
"""

PAGE_2 = """3. RENT
The annual rent is GBP 48,000 per annum, payable quarterly in advance.

4. BREAK CLAUSE
The Tenant may terminate this lease on the third anniversary of the term
commencement date by giving not less than six months written notice.
"""


def make_sample_pdf(path: str, pages: tuple[str, ...] = (PAGE_1, PAGE_2)) -> None:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 90), text, fontsize=11)
    doc.save(path)
    doc.close()


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "examples/sample_lease.pdf"
    make_sample_pdf(out)
    print(f"wrote {out}")
