#!/usr/bin/env python3
"""Parse Kadé facilitator manual texts into chapter-structured JSON.

Accepts both PDF and plain-text (.txt) input directly — no intermediate step needed.

Usage:
    # Parse from PDF
    uv run parse_kade_texts.py parse -i kadé_fr.pdf -o kade_fr_parsed.json -l french

    # Parse from pre-extracted text
    uv run parse_kade_texts.py parse -i kadé_fr.txt -o kade_fr_parsed.json -l french
    uv run parse_kade_texts.py parse -i kadé_mos.txt -o kade_mos_parsed.json -l moore

    # Flatten a parsed JSON to a plain text list
    uv run parse_kade_texts.py flatten -i kade_fr_parsed.json -o kade_fr_flat.txt

    # Run the default batch (both Kadé files)
    uv run parse_kade_texts.py
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from moore_web.pdf_extractor import extract_pdf_blocks
from moore_web.book_parser_facilitateur import (
    parse_with_chapters,
    parse_book_from_json,
    flatten_book_to_list,
    SECTION_TITLES,
    MOORE_SECTION_TITLES,
    MOORE_INTRO_SECTION_TITLES,
    MOORE_INTRO_SUBSECTION_TITLES,
    FRENCH_INTRO_SECTION_TITLES,
    FRENCH_INTRO_SUBSECTION_TITLES,
    Chapter,
    Section,
    Subsection,
)


# Subsection headings appear as "N. Title" or standalone
_MOORE_INTRO_SUB_PATTERNS = [
    re.compile(r"(?:\d+\.\s+)?" + re.escape(t), re.IGNORECASE) for t in MOORE_INTRO_SUBSECTION_TITLES
]

MOORE_INTRO_SUBSECTION_MAP = {
    "Sẽn n kẽed ne seb kãngã": (
        _MOORE_INTRO_SUB_PATTERNS,
        MOORE_INTRO_SUBSECTION_TITLES,
    )
}

# Anchor with $ to avoid matching "L'histoire de Kadé" as "L'histoire"
_FRENCH_INTRO_SUB_PATTERNS = [
    re.compile(r"(?:\d+\.\s+)?" + re.escape(t) + r"\s*$", re.IGNORECASE)
    for t in FRENCH_INTRO_SUBSECTION_TITLES
]

FRENCH_INTRO_SUBSECTION_MAP = {
    "Comment utiliser ce manuel": (
        _FRENCH_INTRO_SUB_PATTERNS,
        FRENCH_INTRO_SUBSECTION_TITLES,
    )
}


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def subsection_to_dict(sub: Subsection) -> dict[str, Any]:
    return {
        "title": sub.title,
        "body": sub.body or None,
        "items": [{"number": i.number, "text": i.text} for i in sub.items],
        "bullet_items": [{"text": b.text} for b in sub.bullet_items],
    }


def section_to_dict(section: Section) -> dict[str, Any]:
    d: dict[str, Any] = {
        "title": section.title,
        "body": section.body or None,
        "items": [{"number": i.number, "text": i.text} for i in section.items],
        "bullet_items": [{"text": b.text} for b in section.bullet_items],
    }
    if section.subsections:
        d["subsections"] = [subsection_to_dict(s) for s in section.subsections]
    return d


def chapter_to_dict(chapter: Chapter) -> dict[str, Any]:
    return {
        "number": chapter.number,
        "title": chapter.title,
        "sections": [section_to_dict(s) for s in chapter.sections],
    }


# ---------------------------------------------------------------------------
# Core parse helpers
# ---------------------------------------------------------------------------


def _build_section_config(language: str) -> tuple:
    if language == "moore":
        titles = MOORE_SECTION_TITLES
    else:
        titles = SECTION_TITLES
    patterns = [re.compile(re.escape(t), re.IGNORECASE) for t in titles]
    return patterns, titles


def _build_intro_config(language: str) -> tuple:
    if language == "moore":
        patterns = [re.compile(re.escape(t), re.IGNORECASE) for t in MOORE_INTRO_SECTION_TITLES]
        return patterns, MOORE_INTRO_SECTION_TITLES, MOORE_INTRO_SUBSECTION_MAP
    else:
        patterns = [re.compile(re.escape(t), re.IGNORECASE) for t in FRENCH_INTRO_SECTION_TITLES]
        return patterns, FRENCH_INTRO_SECTION_TITLES, FRENCH_INTRO_SUBSECTION_MAP


def _detect_language(path: Path, language: str) -> str:
    if language:
        return language.lower()
    return "moore" if "mos" in path.name.lower() else "french"


def parse_and_save(
    txt_path: str,
    output_path: str,
    language: str = "",
    verbose: bool = False,
    page_range: tuple | None = None,
) -> bool:
    txt = Path(txt_path)
    if not txt.exists():
        print(f"File not found: {txt_path}", file=sys.stderr)
        return False

    lang = _detect_language(txt, language)
    sec_patterns, sec_titles = _build_section_config(lang)
    intro_patterns, intro_titles, intro_sub_map = _build_intro_config(lang)

    if txt.suffix.lower() == ".pdf":
        text = extract_pdf_blocks(str(txt), page_range=page_range)
        fmt = "pdf"
    else:
        text = txt.read_text(encoding="utf-8")
        fmt = "txt"

    print(f"Processing: {txt.name}  [{lang}, {fmt}, {len(sec_patterns)} section types]")

    book = parse_with_chapters(
        text,
        sec_patterns,
        sec_titles,
        intro_section_regexes=intro_patterns,
        intro_section_titles=intro_titles,
        intro_subsection_map=intro_sub_map,
    )

    data = {
        "filename": txt.name,
        "chapters": [chapter_to_dict(ch) for ch in book.chapters],
    }
    Path(output_path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    total_sections = sum(len(ch.sections) for ch in book.chapters)
    print(f"  {len(book.chapters)} chapters, {total_sections} sections → {output_path}")

    if verbose:
        for ch in book.chapters:
            print(f"\n  Chapter {ch.number}: {ch.title}")
            for sec in ch.sections:
                sub_info = f"  subsections={len(sec.subsections)}" if sec.subsections else ""
                print(f"    [{sec.title}]{sub_info}  items={len(sec.items)}  body={len(sec.body)}c")

    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cmd_parse(args: argparse.Namespace) -> None:
    page_range = None
    if args.start_page or args.end_page:
        page_range = (args.start_page or 1, args.end_page or 999999)
    ok = parse_and_save(
        args.input,
        args.output,
        language=args.language,
        verbose=args.verbose,
        page_range=page_range,
    )
    sys.exit(0 if ok else 1)


def cmd_flatten(args: argparse.Namespace) -> None:
    book = parse_book_from_json(args.input)
    flattened = flatten_book_to_list(book)
    Path(args.output).write_text("\n".join(flattened), encoding="utf-8")
    print(f"Flattened {len(flattened)} items → {args.output}")


def cmd_batch(base_dir: Path) -> None:
    parse_and_save(str(base_dir / "kadé_fr.txt"), str(base_dir / "kade_fr_parsed.json"), "french")
    parse_and_save(str(base_dir / "kadé_mos.txt"), str(base_dir / "kade_mos_parsed.json"), "moore")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Parse Kadé facilitator manual texts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="command")

    parse_p = sub.add_parser("parse", help="Parse a text file into structured JSON.")
    parse_p.add_argument("--input", "-i", required=True, help="Path to the input text file.")
    parse_p.add_argument(
        "--output", "-o", default="output.json", help="Output JSON file (default: %(default)s)."
    )
    parse_p.add_argument(
        "--language",
        "-l",
        default="",
        choices=["french", "moore", ""],
        help="Language override (auto-detected from filename if omitted).",
    )
    parse_p.add_argument(
        "--start-page", "-s", type=int, default=None, help="First page to extract from PDF (1-based)."
    )
    parse_p.add_argument(
        "--end-page", "-e", type=int, default=None, help="Last page to extract from PDF (1-based, inclusive)."
    )
    parse_p.add_argument("--verbose", "-v", action="store_true", help="Print chapter/section summary.")

    flatten_p = sub.add_parser("flatten", help="Flatten a parsed JSON book to a text list.")
    flatten_p.add_argument("--input", "-i", required=True, help="Path to the parsed JSON file.")
    flatten_p.add_argument("--output", "-o", required=True, help="Output text file.")

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "parse":
        cmd_parse(args)
    elif args.command == "flatten":
        cmd_flatten(args)
    else:
        cmd_batch(Path(__file__).parent)


if __name__ == "__main__":
    main()
