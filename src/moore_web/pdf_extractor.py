"""Extract raw text from PDF files using PyMuPDF."""

from typing import Optional

import pymupdf

# Typographic ligatures that PyMuPDF may emit verbatim when the PDF font's
# encoding table lacks proper Unicode mappings.  Expand them to plain ASCII so
# downstream parsing is not confused.
#
# This dictionary's font maps its fi-ligature glyph to U+03E7 (ϧ, COPTIC
# SMALL LETTER HAI) rather than U+FB01 (ﬁ) or plain "fi", so we handle both.
_LIGATURE_MAP = str.maketrans(
    {
        "\u03e7": "fi",  # ϧ  — font-specific mis-mapping for fi ligature
        "\ufb00": "ff",  # ﬀ
        "\ufb01": "fi",  # ﬁ
        "\ufb02": "fl",  # ﬂ
        "\ufb03": "ffi",  # ﬃ
        "\ufb04": "ffl",  # ﬄ
        "\ufb05": "st",  # ﬅ
        "\ufb06": "st",  # ﬆ
    }
)


def _expand_ligatures(text: str) -> str:
    return text.translate(_LIGATURE_MAP)


def extract_pdf_blocks(
    pdf_path: str,
    page_range: Optional[tuple] = None,
    page_separator: Optional[str] = None,
) -> str:
    """Extract text blocks from a PDF file.

    Args:
        pdf_path: Path to the PDF file.
        page_range: Optional (start_page, end_page) tuple as 1-based indices.
                    If None, all pages are extracted.
        page_separator: String inserted between pages. Defaults to a blank line.
                        Pass ``"---PAGE BREAK---"`` to get explicit markers.

    Returns:
        Extracted text with pages joined by *page_separator*.
    """
    extracted_pages = []

    with pymupdf.open(pdf_path) as doc:
        total_pages = len(doc)

        if page_range:
            start = max(0, page_range[0] - 1)
            end = min(total_pages, page_range[1])
        else:
            start = 0
            end = total_pages

        for page_num in range(start, end):
            page = doc[page_num]
            blocks = page.get_text("blocks", sort=True)
            page_text = [block[4].strip() for block in blocks if block[4].strip()]
            if page_text:
                extracted_pages.append("\n\n".join(page_text))

    sep = f"\n\n{page_separator}\n\n" if page_separator else "\n\n"
    return _expand_ligatures(sep.join(extracted_pages))


def _extract_multicolumn_page(page: pymupdf.Page, num_columns: int) -> str:
    """Return text from a single page laid out in *num_columns* columns."""
    blocks = page.get_text("blocks", sort=True)
    column_width = page.rect.width / num_columns
    columns: list[list[str]] = [[] for _ in range(num_columns)]

    for x0, _y0, _x1, _y1, text, *_ in blocks:
        text = text.strip()
        if not text or text.isdigit():
            continue
        col_idx = min(int(x0 / column_width), num_columns - 1)
        columns[col_idx].append(text)

    return "\n".join("\n".join(col) for col in columns)


def extract_multicolumn_blocks(
    pdf_path: str,
    num_columns: int = 2,
    page_range: Optional[tuple] = None,
    page_separator: Optional[str] = None,
) -> str:
    """Extract text from a multi-column PDF.

    Args:
        pdf_path: Path to the PDF file.
        num_columns: Number of columns per page (default: 2).
        page_range: Optional (start_page, end_page) tuple as 1-based indices.
                    If None, all pages are extracted.
        page_separator: String inserted between pages. Defaults to a blank line.

    Returns:
        Extracted text with columns read left-to-right and pages joined by
        *page_separator*.
    """
    extracted_pages = []

    with pymupdf.open(pdf_path) as doc:
        total_pages = len(doc)

        if page_range:
            start = max(0, page_range[0] - 1)
            end = min(total_pages, page_range[1])
        else:
            start = 0
            end = total_pages

        for page_num in range(start, end):
            text = _extract_multicolumn_page(doc[page_num], num_columns)
            if text.strip():
                extracted_pages.append(text)

    sep = f"\n\n{page_separator}\n\n" if page_separator else "\n\n"
    return _expand_ligatures(sep.join(extracted_pages))
