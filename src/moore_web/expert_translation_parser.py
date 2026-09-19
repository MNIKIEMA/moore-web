"""Extract aligned French–Mooré rows from an expert translation batch PDF.

The source is a Google Sheets PDF export with eight ruled columns: ID, DOMAIN,
CONTEXT, SOURCE, TRANSLATION, COMMENTS, qa.check, and qa.comments. Each table
row is already a translation pair; sentence alignment would lose the supplied
row boundary and is deliberately not performed here.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import msgspec
import pdfplumber


HEADERS = (
    "ID",
    "DOMAIN",
    "CONTEXT",
    "SOURCE",
    "TRANSLATION",
    "COMMENTS",
    "qa.check",
    "qa.comments",
)


class ExpertTranslation(msgspec.Struct):
    """A source row, including its context and reviewer annotations."""

    id: str
    src_lang: str
    tgt_lang: str
    source_text: str
    target_text: str
    source: str
    doc_id: str
    page: int
    domain: str | None = None
    context: str | None = None
    comments: str | None = None
    qa_check: str | None = None
    qa_comments: str | None = None


def _clean_cell(value: str | None) -> str:
    """Remove PDF line wraps without changing punctuation or word spelling."""
    if not value:
        return ""
    value = unicodedata.normalize("NFC", value)
    value = re.sub(r"(?<=\w)-[ \t]*\n[ \t]*(?=\w)", "-", value)
    return " ".join(value.split())


def _optional(value: str | None) -> str | None:
    return _clean_cell(value) or None


def parse_table_rows(
    rows: list[list[str | None]],
    *,
    page: int,
    doc_id: str,
    seen_ids: set[str],
) -> list[ExpertTranslation]:
    """Convert one page's extracted table, rejecting malformed data rows."""
    parsed: list[ExpertTranslation] = []
    for row_number, row in enumerate(rows, start=1):
        if len(row) != len(HEADERS):
            raise ValueError(f"Page {page}, row {row_number}: expected eight columns, got {len(row)}")

        cells = [_clean_cell(value) for value in row]
        if tuple(cells) == HEADERS:
            continue

        row_id = "".join((row[0] or "").split())
        if not row_id or not cells[3] or not cells[4]:
            raise ValueError(f"Page {page}, row {row_number}: missing ID, source, or translation")
        if row_id in seen_ids:
            raise ValueError(f"Page {page}, row {row_number}: duplicate ID {row_id!r}")
        seen_ids.add(row_id)

        parsed.append(
            ExpertTranslation(
                id=row_id,
                src_lang="fra",
                tgt_lang="mos",
                source_text=cells[3],
                target_text=cells[4],
                source="expert-translations",
                doc_id=doc_id,
                page=page,
                domain=_optional(row[1]),
                context=_optional(row[2]),
                comments=_optional(row[5]),
                qa_check=_optional(row[6]),
                qa_comments=_optional(row[7]),
            )
        )
    return parsed


def parse_expert_translations(input_pdf: str | Path, *, doc_id: str | None = None) -> list[ExpertTranslation]:
    """Read every table row in a translation batch PDF, in document order."""
    pdf_path = Path(input_pdf)
    document_id = doc_id or pdf_path.stem
    if not document_id:
        raise ValueError("document_id must not be empty")

    records: list[ExpertTranslation] = []
    seen_ids: set[str] = set()
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            tables = [table for table in page.extract_tables() if table and len(table[0]) == len(HEADERS)]
            if len(tables) != 1:
                raise ValueError(f"Page {page_number}: expected one eight-column table, found {len(tables)}")
            records.extend(
                parse_table_rows(tables[0], page=page_number, doc_id=document_id, seen_ids=seen_ids)
            )

    if not records:
        raise ValueError(f"No translation rows found in {pdf_path}")
    return records


def write_jsonl(records: list[ExpertTranslation], output_path: str | Path) -> None:
    """Write one self-contained aligned record per line."""
    with Path(output_path).open("wb") as output:
        output.writelines(msgspec.json.encode(record) + b"\n" for record in records)
