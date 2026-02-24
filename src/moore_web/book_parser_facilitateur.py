"""
Parser for the French AIDS church manual (pymupdf txt output).

Pipeline:
    raw text
      → split_into_chapters()           # by "Chapitre N" headings
          → split_into_sections()       # by known section titles
              → parse_section()         # numbered items, subsections, body

Alternative flexible pipeline:
    raw text
      → split_and_parse_by_sections()   # by custom section regex patterns
                                        # Returns all text + enums

Structure per chapter:
  - L'histoire de Kadé
  - Questions à discuter
  - Choses à apprendre
  - Sketch
  - Sketch et chant
  - Ce que dit la Bible
  - Prier et agir

Usage:
    # Using the new flexible function
    import re
    section_regexes = [
        re.compile(r"^Chapitre \\d+", re.IGNORECASE),
        re.compile(r"^Questions à discuter", re.IGNORECASE),
    ]
    sections = split_and_parse_by_sections(text, section_regexes)
    # Each section now contains both body text and enumerated items
"""

import unicodedata
import re
import json
import sys
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class NumberedItem:
    number: int
    text: str


@dataclass
class BulletItem:
    text: str


@dataclass
class Subsection:
    title: str
    items: list[NumberedItem] = field(default_factory=list)
    bullet_items: list[BulletItem] = field(default_factory=list)
    body: str = ""


@dataclass
class Section:
    title: str
    subsections: list[Subsection] = field(default_factory=list)
    items: list[NumberedItem] = field(default_factory=list)
    bullet_items: list[BulletItem] = field(default_factory=list)
    body: str = ""
    raw_text: str = ""


@dataclass
class Chapter:
    number: int
    title: str
    sections: list[Section] = field(default_factory=list)
    raw_text: str = ""


@dataclass
class Book:
    title: str = ""
    chapters: list[Chapter] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Constants & regexes
# ---------------------------------------------------------------------------

SECTION_TITLES = [
    "L'histoire de Kadé",
    "Questions à discuter",
    "Choses à apprendre",
    "Sketch et chant",
    "Sketch",
    "Ce que dit la Bible",
    "Prier et agir",
]

MOORE_SECTIION_TITLES = [
    "Karem-y kibarã",
    "Kibarã",
    "Sõaseg sokdse",
    "D sẽn segd n zãms bũmb niisi",
    "Reem",
    "Reem la yɩɩlle",
    "Reem la yɩɩla",
    "Wẽnnaam sebra sẽn yet bũmb ningã",
    "Pʋʋsg la tʋʋmde",
    "Pʋʋsog la tʋʋma",
]

MOORE_SUBSECTION_TITLES = []
CHAPTER_RE = re.compile(r"^(?:chapitre|sak\s+a)\s+(\d+)(?:\s+soaba)?\s*[:\-–]?\s*(.*)$", re.IGNORECASE)

NUMBERED_ITEM_RE = re.compile(r"^\s*(\d+)\.\s+(.+)")
BULLET_ITEM_RE = re.compile(r"^\s*•\s+(.+)")

LISEZ_RE = re.compile(r"^Lisez\s+.+", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u2019", "'")  # Right single quotation mark (')
    text = text.replace("\u2018", "'")  # Left single quotation mark (')
    text = text.replace("\u201c", '"')  # Left double quotation mark (")
    text = text.replace("\u201d", '"')  # Right double quotation mark (")
    text = text.replace("\u00ab", '"')  # Left-pointing double angle quotation mark («)
    text = text.replace("\u00bb", '"')
    return re.sub(r"\s+", " ", text).strip()


def is_chapter_heading(line: str) -> Optional[tuple[int, str]]:
    m = CHAPTER_RE.match(normalize(line))
    if m:
        return int(m.group(1)), m.group(2).strip()
    return None


def match_section_title(line: str) -> Optional[str]:
    """Return canonical section title if line matches, else None."""
    norm = normalize(line)
    match_norm = re.sub(r"^\d+\.?\s+", "", norm)
    for title in SECTION_TITLES + MOORE_SECTIION_TITLES:
        if match_norm.lower().startswith(title.lower()):
            return title
    return None


def looks_like_subsection_heading(line: str) -> bool:
    """Heuristic: subsection titles are questions or Lisez references."""
    if LISEZ_RE.match(line):
        return True
    if line.endswith("?"):
        return True
    return False


# ---------------------------------------------------------------------------
# Parser functions
# ---------------------------------------------------------------------------


def split_text_by_regex(text: str, regex: re.Pattern) -> list[tuple[Optional[str], str]]:
    """
    Splits text by a regex that identifies headings.
    Returns a list of (heading_line, content_text) pairs.
    Text before the first heading is returned with heading=None.
    """
    parts = []
    lines = text.splitlines()

    current_heading = None
    current_lines = []

    for line in lines:
        norm_line = normalize(line)
        if regex.match(norm_line):
            if current_heading is not None or current_lines:
                parts.append((current_heading, "\n".join(current_lines)))
            current_heading = line
            current_lines = []
        else:
            current_lines.append(line)

    if current_heading is not None or current_lines:
        parts.append((current_heading, "\n".join(current_lines)))

    return parts


def split_into_chapters(text: str) -> list[Chapter]:
    """
    Returns list of Chapter objects.
    """
    chapters: list[Chapter] = []
    blocks = split_text_by_regex(text, CHAPTER_RE)

    for heading, content in blocks:
        if heading is None:
            continue

        res = is_chapter_heading(heading)
        if res:
            num, title = res
            chapters.append(
                Chapter(number=num, title=title, raw_text=heading + ("\n" + content if content else ""))
            )

    return chapters


def split_into_sections(text: str) -> list[Section]:
    """
    Returns list of Section objects for single chapter's raw text.
    """
    sections: list[Section] = []

    all_titles = SECTION_TITLES + MOORE_SECTIION_TITLES
    escaped_titles = [re.escape(t) for t in all_titles]

    section_regex = re.compile(r"^(?:\d+\.?\s+)?(" + "|".join(escaped_titles) + r").*", re.IGNORECASE)

    blocks = split_text_by_regex(text, section_regex)

    for heading, content in blocks:
        if heading is None:
            if content.strip():
                sections.append(Section(title="Intro", raw_text=content))
            continue

        title = match_section_title(heading) or heading

        sections.append(Section(title=title, raw_text=heading + ("\n" + content if content else "")))

    return sections


def collect_numbered_items(lines: list[str]) -> list[NumberedItem]:
    """
    Collect numbered items (possibly multi-line) from a flat list of lines.
    Lines that don't belong to any item are ignored.
    """
    items: list[NumberedItem] = []
    current_num: Optional[int] = None
    current_parts: list[str] = []

    def flush():
        if current_num is not None and current_parts:
            items.append(NumberedItem(current_num, normalize(" ".join(current_parts))))

    for raw in lines:
        line = normalize(raw)
        m = NUMBERED_ITEM_RE.match(raw)
        if m:
            flush()
            current_num = int(m.group(1))
            current_parts = [m.group(2).strip()]
        elif current_num is not None and line:
            current_parts.append(line)

    flush()
    return items


def collect_bullet_items(lines: list[str]) -> list[BulletItem]:
    """
    Collect bullet items (possibly multi-line) from a flat list of lines.
    Handles bullets that may be on their own line followed by text on next line(s).
    Lines that don't belong to any item are ignored.
    """
    items: list[BulletItem] = []
    current_parts: list[str] = []
    in_bullet_item = False

    def flush():
        if in_bullet_item and current_parts:
            items.append(BulletItem(normalize(" ".join(current_parts))))

    for raw in lines:
        line = normalize(raw)
        if re.match(r"^\s*•\s*$", raw):
            flush()
            in_bullet_item = True
            current_parts = []
        elif re.match(r"^\s*•\s+(.+)", raw):
            flush()
            in_bullet_item = True
            m = re.match(r"^\s*•\s+(.+)", raw)
            current_parts = [m.group(1).strip()]
        elif in_bullet_item and line:
            current_parts.append(line)

    flush()
    return items


def collect_items(lines: list[str]) -> tuple[list[NumberedItem], list[BulletItem]]:
    """
    Collect both numbered and bullet items from lines.
    Returns (numbered_items, bullet_items).
    """
    return collect_numbered_items(lines), collect_bullet_items(lines)


def parse_questions_section(lines: list[str]) -> Section:
    sec = Section(title="Questions à discuter")
    numbered_items, bullet_items = collect_items(lines)
    sec.items = numbered_items
    sec.bullet_items = bullet_items

    body_text = normalize(
        " ".join(
            normalize(line)
            for line in lines
            if normalize(line) and not NUMBERED_ITEM_RE.match(line) and not BULLET_ITEM_RE.match(line)
        )
    )
    if body_text:
        sec.body = body_text
    return sec


def parse_choses_section(lines: list[str]) -> Section:
    """
    Choses à apprendre contains subsection headings (questions ending in '?'
    or descriptive headings) each followed by body text, numbered items, or bullet items.
    """
    sec = Section(title="Choses à apprendre")

    blocks: list[tuple[str, list[str]]] = []
    current_title: Optional[str] = None
    current_block: list[str] = []

    for raw in lines:
        line = normalize(raw)
        if not line:
            if current_title is not None:
                current_block.append(raw)
            continue

        if looks_like_subsection_heading(line):
            if current_title is not None:
                blocks.append((current_title, current_block))
            current_title = line
            current_block = []
        else:
            if current_title is not None:
                current_block.append(raw)
            else:
                sec.body += (" " + line) if sec.body else line

    if current_title is not None:
        blocks.append((current_title, current_block))

    for title, block_lines in blocks:
        sub = Subsection(title=title)
        numbered_items, bullet_items = collect_items(block_lines)
        if numbered_items:
            sub.items = numbered_items
        if bullet_items:
            sub.bullet_items = bullet_items
        if not numbered_items and not bullet_items:
            sub.body = normalize(" ".join(normalize(line) for line in block_lines if normalize(line)))
        sec.subsections.append(sub)

    return sec


def parse_bible_section(lines: list[str]) -> Section:
    """
    Ce que dit la Bible contains subsections headed by 'Lisez X.Y-Z'
    each followed by numbered or bullet discussion items.
    """
    sec = Section(title="Ce que dit la Bible")

    blocks: list[tuple[str, list[str]]] = []
    current_ref: Optional[str] = None
    current_block: list[str] = []

    for raw in lines:
        line = normalize(raw)
        if LISEZ_RE.match(line):
            if current_ref is not None:
                blocks.append((current_ref, current_block))
            current_ref = line
            current_block = []
        elif current_ref is not None:
            current_block.append(raw)

    if current_ref is not None:
        blocks.append((current_ref, current_block))

    for ref, block_lines in blocks:
        sub = Subsection(title=ref)
        numbered_items, bullet_items = collect_items(block_lines)
        if numbered_items:
            sub.items = numbered_items
        if bullet_items:
            sub.bullet_items = bullet_items
        if not numbered_items and not bullet_items:
            sub.body = normalize(" ".join(normalize(line) for line in block_lines if normalize(line)))
        sec.subsections.append(sub)

    return sec


def parse_generic_section(title: str, lines: list[str]) -> Section:
    """Fallback: collect body text, numbered items, and bullet items."""
    sec = Section(title=title)
    numbered_items, bullet_items = collect_items(lines)
    if numbered_items:
        sec.items = numbered_items
    if bullet_items:
        sec.bullet_items = bullet_items
    
    body = normalize(" ".join(
        normalize(line) for line in lines 
        if normalize(line) and not NUMBERED_ITEM_RE.match(line) and not BULLET_ITEM_RE.match(line)
    ))
    if body:
        sec.body = body
    return sec


SECTION_PARSERS = {
    "Questions à discuter": parse_questions_section,
    "Choses à apprendre": parse_choses_section,
    "Ce que dit la Bible": parse_bible_section,
}


def parse_section(title: str, lines: list[str]) -> Section:
    parser = SECTION_PARSERS.get(title)
    if parser:
        return parser(lines)
    return parse_generic_section(title, lines)


def split_and_parse_by_sections(
    text: str, section_regexes: list[re.Pattern], section_titles: list[str] = None
) -> list[Section]:
    """
    Splits text by provided section regex patterns and returns all text content + any enums.

    Args:
        text: Raw text to parse
        section_regexes: List of compiled regex patterns to identify section headers
        section_titles: Optional list of canonical section titles (must match length of regexes)

    Returns:
        List of Section objects with both body text and enumerated items preserved
    """
    if section_titles is None:
        section_titles = [None] * len(section_regexes)

    if len(section_regexes) != len(section_titles):
        raise ValueError("section_regexes and section_titles must have the same length")

    sections: list[Section] = []
    lines = text.splitlines()

    current_title: Optional[str] = None
    current_heading: Optional[str] = None
    current_lines: list[str] = []

    for line in lines:
        norm_line = normalize(line)

        matched = False
        for regex, canonical_title in zip(section_regexes, section_titles):
            if regex.match(norm_line):
                if current_heading is not None or current_lines:
                    title = current_title or "Intro"
                    sec = Section(title=title, raw_text="\n".join(current_lines))

                    numbered_items, bullet_items = collect_items(current_lines)
                    sec.items = numbered_items
                    sec.bullet_items = bullet_items

                    body_text = normalize(
                        " ".join(
                            normalize(ln)
                            for ln in current_lines
                            if normalize(ln) and not NUMBERED_ITEM_RE.match(ln) and not BULLET_ITEM_RE.match(ln)
                        )
                    )
                    if body_text:
                        sec.body = body_text

                    sections.append(sec)

                current_heading = line
                current_title = canonical_title or match_section_title(line) or line
                current_lines = []
                matched = True
                break

        if not matched:
            current_lines.append(line)

    if current_heading is not None or current_lines:
        title = current_title or "Intro"
        sec = Section(title=title, raw_text="\n".join(current_lines))
        numbered_items, bullet_items = collect_items(current_lines)
        sec.items = numbered_items
        sec.bullet_items = bullet_items
        
        body_text = normalize(
            " ".join(
                normalize(ln) for ln in current_lines 
                if normalize(ln) and not NUMBERED_ITEM_RE.match(ln) and not BULLET_ITEM_RE.match(ln)
            )
        )
        if body_text:
            sec.body = body_text
        sections.append(sec)

    return sections


def parse(text: str) -> Book:
    book = Book()

    for chapter in split_into_chapters(text):
        for section in split_into_sections(chapter.raw_text):
            parsed_section = parse_section(section.title, section.raw_text.splitlines())
            parsed_section.raw_text = section.raw_text
            chapter.sections.append(parsed_section)

        book.chapters.append(chapter)

    return book


def parse_file(txt_path: str) -> Book:
    text = Path(txt_path).read_text(encoding="utf-8")
    return parse(text)


def book_to_dict(book: Book) -> dict:
    def item_d(i: NumberedItem):
        return {"number": i.number, "text": i.text}

    def bullet_item_d(b: BulletItem):
        return {"text": b.text}

    def sub_d(s: Subsection):
        d: dict = {"title": s.title}
        if s.items:
            d["items"] = [item_d(i) for i in s.items]
        if s.bullet_items:
            d["bullet_items"] = [bullet_item_d(b) for b in s.bullet_items]
        if s.body:
            d["body"] = s.body
        return d

    def sec_d(s: Section):
        d: dict = {"title": s.title}
        if s.subsections:
            d["subsections"] = [sub_d(sub) for sub in s.subsections]
        if s.items:
            d["items"] = [item_d(i) for i in s.items]
        if s.bullet_items:
            d["bullet_items"] = [bullet_item_d(b) for b in s.bullet_items]
        if s.body:
            d["body"] = s.body
        if s.raw_text:
            d["raw_text"] = s.raw_text
        return d

    def chap_d(c: Chapter):
        return {
            "number": c.number,
            "title": c.title,
            "sections": [sec_d(s) for s in c.sections],
            "raw_text": c.raw_text,
        }

    return {"title": book.title, "chapters": [chap_d(c) for c in book.chapters]}


def example_split_and_parse():
    """
    Example: using split_and_parse_by_sections with custom section regex patterns.
    This function demonstrates how to split text and preserve all text + enums.
    """
    french_section_patterns = [
        re.compile(r"^L'histoire de Kadé", re.IGNORECASE),
        re.compile(r"^Questions à discuter", re.IGNORECASE),
        re.compile(r"^Choses à apprendre", re.IGNORECASE),
    ]

    section_titles = [
        "L'histoire de Kadé",
        "Questions à discuter",
        "Choses à apprendre",
    ]
    text = "Your parsed text here..."

    sections = split_and_parse_by_sections(text, french_section_patterns, section_titles)

    for section in sections:
        print(f"Section: {section.title}")
        print(f"  Items: {len(section.items)}")
        print(f"  Body text: {section.body[:100]}..." if section.body else "  Body text: (empty)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parser.py <extracted.txt> [output.json]")
        sys.exit(1)

    txt_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "output.json"

    book = parse_file(txt_path)
    result = book_to_dict(book)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Parsed {len(book.chapters)} chapters → {out_path}")
    for ch in book.chapters:
        print(f"\n  Chapter {ch.number}: {ch.title}")
        for sec in ch.sections:
            print(f"    [{sec.title}]  subsections={len(sec.subsections)}  items={len(sec.items)}")
