"""Parse glossary PDFs and produce aligned French/Mooré pairs.

Sources
-------
- Mooré:  Glossaire_des_termes_usuels_du_numerique_et_de_la_poste_en_Moore__valide.pdf
- French: Lexique_de_l_economie_numerique_et_des_postes.pdf

Mooré PDF layout (47 pages)
-----------------------------
  Pages 1-3  : text (Mooré preface/introduction) → sentence segments
  Page  4    : section header "Pipi palle Nimeriki" → skip
  Pages 5-47 : glossary tables (4 cols: N°, TERMES, GOM-BI-TIGSI, B VÔOR WILGRI)
               page 45 is a postal-section header → automatically skipped (no valid table)

French PDF layout (94 pages)
------------------------------
  Pages 1-4  : text (French preface/introduction) → sentence segments
  Page  5    : section header "Partie 1 Économie Numérique" → skip
  Pages 6-86 : glossary tables (3 cols: N°, Mots clés, Définitions)
  Pages 87-94: appendices / end matter → skip

Alignment heuristic
--------------------
  Match TERMES (Mooré PDF, French term column) to Mots clés (French PDF) using
  case-insensitive, accent-stripped key comparison.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import msgspec
import pdfplumber

from moore_web.pdf_extractor import extract_pdf_blocks
from moore_web.segment import split_sentences


# ---------------------------------------------------------------------------
# pdfplumber table detection settings
# ---------------------------------------------------------------------------

_SETTINGS_STRICT = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "snap_tolerance": 3,
    "join_tolerance": 3,
    "edge_min_length": 50,  # filters out decorative borders
    "min_words_vertical": 1,
    "min_words_horizontal": 1,
}

# Fallback for pages where the section-letter box confuses strict detection
_SETTINGS_LOOSE = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "snap_tolerance": 5,
    "join_tolerance": 5,
    "edge_min_length": 30,
    "min_words_vertical": 1,
    "min_words_horizontal": 1,
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class MooreEntry(msgspec.Struct):
    num: str
    fr_term: str
    mos_term: str
    mos_definition: str
    source: str = "moore_glossary"


class FrenchEntry(msgspec.Struct):
    num: str
    fr_term: str
    fr_definition: str
    source: str = "french_lexique"


class AlignedEntry(msgspec.Struct):
    fr_term: str
    mos_term: str
    fr_definition: str
    mos_definition: str
    source: str = "glossary_aligned"


class TextSegment(msgspec.Struct):
    text: str
    lang: str
    source: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HEADER_KEYWORDS_MOORE = {"N°", "TERMES", "GOM", "TIGSI", "VÕOR", "WILGRI"}
_HEADER_KEYWORDS_FRENCH = {"N°", "Mots", "clés", "Définitions"}

# Typographic characters → plain ASCII equivalents
_UNICODE_MAP = str.maketrans(
    {
        "\u2018": "'",   # '  LEFT SINGLE QUOTATION MARK
        "\u2019": "'",   # '  RIGHT SINGLE QUOTATION MARK  ← reported
        "\u201a": "'",   # ‚  SINGLE LOW-9 QUOTATION MARK
        "\u201b": "'",   # ‛  SINGLE HIGH-REVERSED-9 QUOTATION MARK
        "\u201c": '"',   # "  LEFT DOUBLE QUOTATION MARK
        "\u201d": '"',   # "  RIGHT DOUBLE QUOTATION MARK
        "\u201e": '"',   # „  DOUBLE LOW-9 QUOTATION MARK
        "\u2013": "-",   # –  EN DASH
        "\u2014": "-",   # —  EM DASH
        "\u2015": "-",   # ―  HORIZONTAL BAR
        "\u2026": "...", # …  HORIZONTAL ELLIPSIS
        "\u00a0": " ",   # non-breaking space
        "\u202f": " ",   # narrow no-break space
        "\u2009": " ",   # thin space
        "\u00ab": '"',   # «  LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
        "\u00bb": '"',   # »  RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
    }
)


def normalize_unicode(text: str) -> str:
    """Replace typographic punctuation with plain ASCII equivalents."""
    return text.translate(_UNICODE_MAP)


def _clean(value: str | None) -> str:
    """Collapse whitespace, normalize typographic chars, and strip a cell value."""
    if value is None:
        return ""
    return " ".join(normalize_unicode(value).split())


def _normalize_key(text: str) -> str:
    """Lowercase + strip diacritics for fuzzy term matching."""
    nfkd = unicodedata.normalize("NFKD", text.lower().strip())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _fix_moore_hyphens(text: str) -> str:
    """Remove spurious spaces around hyphens in Mooré compound words.

    PDF extraction inserts whitespace at cell line-breaks, splitting
    compound words like ``tʋʋm-teed`` into ``tʋʋm- teed``.
    Fix: collapse any whitespace immediately after (or before) a hyphen
    that sits between two word characters.
    """
    # space(s) after hyphen:  "tʋʋm- teed"  → "tʋʋm-teed"
    text = re.sub(r"(?<=\w)-\s+(?=\w)", "-", text)
    # space(s) before hyphen: "tʋʋm -teed"  → "tʋʋm-teed"
    text = re.sub(r"(?<=\w)\s+-(?=\w)", "-", text)
    return text


def _is_header_row(row: list[str], keywords: set[str]) -> bool:
    joined = " ".join(row)
    return any(kw in joined for kw in keywords)


def _best_table(page, min_cols: int) -> list[list[str]] | None:
    """Return the best valid table from a page, trying strict then loose settings."""
    for settings in (_SETTINGS_STRICT, _SETTINGS_LOOSE):
        tables = page.extract_tables(settings)
        candidates = [t for t in tables if t and len(t[0]) >= min_cols]
        if candidates:
            # prefer the table with the most data rows (not the decorative border)
            best = max(candidates, key=lambda t: sum(1 for r in t if any(_clean(c) for c in r)))
            cleaned = [[_clean(c) for c in row] for row in best]
            return cleaned
    return None


# ---------------------------------------------------------------------------
# Mooré glossary extractor
# ---------------------------------------------------------------------------

MOORE_PDF = "Glossaire_des_termes_usuels_du_numerique_et_de_la_poste_en_Moore__valide.pdf"

_MOORE_TEXT_PAGE_RANGE = (1, 3)   # 1-based inclusive
_MOORE_SKIP_PAGES = {4}           # section header pages (1-based)


def _parse_moore_row(row: list[str], ncols: int) -> MooreEntry | None:
    """Map a row to MooreEntry regardless of whether TERMES column is present."""
    if ncols == 4:
        num, fr_term, mos_term, mos_def = row[0], row[1], row[2], row[3]
    elif ncols == 3:
        # TERMES column missing (e.g. page 24 K-section)
        num, mos_term, mos_def = row[0], row[1], row[2]
        fr_term = ""
    else:
        return None

    if not mos_term and not fr_term:
        return None

    return MooreEntry(
        num=num,
        fr_term=fr_term,
        mos_term=_fix_moore_hyphens(mos_term),
        mos_definition=_fix_moore_hyphens(mos_def),
    )


def extract_moore_tables(pdf_path: str = MOORE_PDF) -> list[MooreEntry]:
    entries: list[MooreEntry] = []

    with pdfplumber.open(pdf_path) as pdf:
        for idx, page in enumerate(pdf.pages):
            page_num = idx + 1
            if page_num <= _MOORE_TEXT_PAGE_RANGE[1] or page_num in _MOORE_SKIP_PAGES:
                continue

            rows = _best_table(page, min_cols=3)
            if not rows:
                continue

            # skip header row(s)
            start = 1 if _is_header_row(rows[0], _HEADER_KEYWORDS_MOORE) else 0
            ncols = len(rows[0])

            for row in rows[start:]:
                entry = _parse_moore_row(row, ncols)
                if entry:
                    entries.append(entry)

    return entries


def extract_moore_text_segments(pdf_path: str = MOORE_PDF) -> list[TextSegment]:
    text = normalize_unicode(extract_pdf_blocks(pdf_path, page_range=_MOORE_TEXT_PAGE_RANGE))
    return [
        TextSegment(text=sent, lang="mos", source="moore_glossary_preface")
        for sent in split_sentences(text)
        if sent.strip()
    ]


# ---------------------------------------------------------------------------
# French lexique extractor
# ---------------------------------------------------------------------------

FRENCH_PDF = "Lexique_de_l_economie_numerique_et_des_postes.pdf"

_FRENCH_TEXT_PAGE_RANGE = (1, 4)   # 1-based inclusive
_FRENCH_SKIP_PAGES = {5}           # section header pages (1-based)
_FRENCH_TABLE_LAST_PAGE = 86       # 1-based; pages 87-94 are appendices


def extract_french_tables(pdf_path: str = FRENCH_PDF) -> list[FrenchEntry]:
    entries: list[FrenchEntry] = []

    with pdfplumber.open(pdf_path) as pdf:
        for idx, page in enumerate(pdf.pages):
            page_num = idx + 1
            if page_num <= _FRENCH_TEXT_PAGE_RANGE[1] or page_num in _FRENCH_SKIP_PAGES:
                continue
            if page_num > _FRENCH_TABLE_LAST_PAGE:
                break

            rows = _best_table(page, min_cols=2)
            if not rows:
                continue

            start = 1 if _is_header_row(rows[0], _HEADER_KEYWORDS_FRENCH) else 0

            for row in rows[start:]:
                if len(row) < 3:
                    continue
                num, fr_term, fr_def = row[0], row[1], row[2]
                if not fr_term:
                    continue
                entries.append(FrenchEntry(num=num, fr_term=fr_term, fr_definition=fr_def))

    return entries


def extract_french_text_segments(pdf_path: str = FRENCH_PDF) -> list[TextSegment]:
    text = normalize_unicode(extract_pdf_blocks(pdf_path, page_range=_FRENCH_TEXT_PAGE_RANGE))
    return [
        TextSegment(text=sent, lang="fr", source="french_lexique_preface")
        for sent in split_sentences(text)
        if sent.strip()
    ]


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


def align_glossaries(
    moore_entries: list[MooreEntry],
    french_entries: list[FrenchEntry],
) -> list[AlignedEntry]:
    """Match Mooré entries to French entries via a three-pass lookup.

    Pass 1 — exact normalized match
        Lowercase + strip diacritics on both sides.

    Pass 2 — acronym-in-parentheses
        Index French terms by their parenthesised substring.
        e.g. "Asymetric Digital Subscriber Line (ADSL)" is also reachable by "adsl",
        so a Mooré entry with fr_term="ADSL" gets matched.

    Pass 3 — space-collapse (PDF line-break artefacts)
        Some terms in the Mooré PDF have mid-word spaces introduced by the PDF
        renderer at line boundaries (e.g. "Communicatio ns électroniques").
        Collapsing all whitespace before comparing catches these.
    """
    # Pass 1 index: normalised term → entry
    fr_exact: dict[str, FrenchEntry] = {
        _normalize_key(fe.fr_term): fe for fe in french_entries
    }

    # Pass 2 index: normalised text-in-parens → entry
    fr_paren: dict[str, FrenchEntry] = {}
    for fe in french_entries:
        m = re.search(r"\(([^)]+)\)", fe.fr_term)
        if m:
            fr_paren[_normalize_key(m.group(1))] = fe

    # Pass 3 index: space-collapsed normalised term → entry
    fr_nospace: dict[str, FrenchEntry] = {
        re.sub(r"\s+", "", _normalize_key(fe.fr_term)): fe for fe in french_entries
    }

    aligned: list[AlignedEntry] = []
    for me in moore_entries:
        if not me.fr_term:
            continue

        key = _normalize_key(me.fr_term)
        fe = (
            fr_exact.get(key)
            or fr_paren.get(key)
            or fr_nospace.get(re.sub(r"\s+", "", key))
        )
        if fe:
            aligned.append(
                AlignedEntry(
                    fr_term=fe.fr_term,
                    mos_term=me.mos_term,
                    fr_definition=fe.fr_definition,
                    mos_definition=me.mos_definition,
                )
            )

    return aligned


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def _write_jsonl(items: list, path: Path) -> None:
    encoder = msgspec.json.Encoder()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for item in items:
            f.write(encoder.encode(item))
            f.write(b"\n")
    print(f"  wrote {len(items):>5} rows  →  {path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def parse_glossaries(
    moore_pdf: str = MOORE_PDF,
    french_pdf: str = FRENCH_PDF,
    output_dir: str = "data/glossary",
    skip_preface: bool = True,
) -> None:
    """Extract, align, and save glossary data.

    Args:
        moore_pdf:     Path to the Mooré glossary PDF.
        french_pdf:    Path to the French lexique PDF.
        output_dir:    Directory where JSONL files are written.
        skip_preface:  When True (default), omit preface/introduction text
                       segments — they are thematic prose unrelated to the
                       glossary entries and not useful for alignment.
    """
    out = Path(output_dir)

    print("Extracting Mooré glossary tables…")
    moore_entries = extract_moore_tables(moore_pdf)
    _write_jsonl(moore_entries, out / "moore_entries.jsonl")

    print("Extracting French lexique tables…")
    french_entries = extract_french_tables(french_pdf)
    _write_jsonl(french_entries, out / "french_entries.jsonl")

    print("Aligning glossaries…")
    aligned = align_glossaries(moore_entries, french_entries)
    _write_jsonl(aligned, out / "aligned.jsonl")
    if moore_entries:
        print(f"  matched {len(aligned)} / {len(moore_entries)} Mooré entries "
              f"({len(aligned)/len(moore_entries)*100:.1f}%)")

    if not skip_preface:
        print("Extracting Mooré text segments (pages 1-3)…")
        moore_text = extract_moore_text_segments(moore_pdf)
        _write_jsonl(moore_text, out / "moore_text_segments.jsonl")

        print("Extracting French text segments (pages 1-4)…")
        french_text = extract_french_text_segments(french_pdf)
        _write_jsonl(french_text, out / "french_text_segments.jsonl")
    else:
        print("  skipping preface/introduction segments (skip_preface=True)")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Parse glossary PDFs and produce aligned pairs.")
    ap.add_argument("--moore-pdf", default=MOORE_PDF)
    ap.add_argument("--french-pdf", default=FRENCH_PDF)
    ap.add_argument("--output-dir", default="data/glossary")
    ap.add_argument(
        "--include-preface",
        action="store_true",
        help="Also extract preface/introduction text segments (skipped by default).",
    )
    args = ap.parse_args()

    parse_glossaries(
        moore_pdf=args.moore_pdf,
        french_pdf=args.french_pdf,
        output_dir=args.output_dir,
        skip_preface=not args.include_preface,
    )
