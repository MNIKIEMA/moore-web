"""
Parser for the French AIDS church manual (pymupdf txt output).

Pipeline:
    raw text
      → split_into_chapters()     # by "Chapitre N" headings
          → split_into_sections() # by known section titles
              → parse_section()   # numbered items, subsections, body

Structure per chapter:
  - L'histoire de Kadé
  - Questions à discuter
  - Choses à apprendre
  - Sketch
  - Sketch et chant
  - Ce que dit la Bible
  - Prier et agir
"""

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
class Subsection:
    title: str
    items: list[NumberedItem] = field(default_factory=list)
    body: str = ""


@dataclass
class Section:
    title: str
    subsections: list[Subsection] = field(default_factory=list)
    items: list[NumberedItem] = field(default_factory=list)
    body: str = ""


@dataclass
class Chapter:
    number: int
    title: str
    sections: list[Section] = field(default_factory=list)


@dataclass
class Book:
    title: str = ""
    chapters: list[Chapter] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Constants & regexes
# ---------------------------------------------------------------------------

# Order matters: "Sketch et chant" must come before "Sketch"
SECTION_TITLES = [
    "L'histoire de Kad",
    "Questions à discuter",
    "Choses à apprendre",
    "Sketch et chant",
    "Sketch",
    "Ce que dit la Bible",
    "Prier et agir",
]

CHAPTER_RE = re.compile(r"^chapitre\s+(\d+)\s*[:\-–]?\s*(.+)$", re.IGNORECASE)

NUMBERED_ITEM_RE = re.compile(r"^\s*(\d+)\.\s+(.+)")

LISEZ_RE = re.compile(r"^Lisez\s+.+", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def is_chapter_heading(line: str) -> Optional[tuple[int, str]]:
    m = CHAPTER_RE.match(normalize(line))
    if m:
        return int(m.group(1)), m.group(2).strip()
    return None


def match_section_title(line: str) -> Optional[str]:
    """Return canonical section title if line matches, else None."""
    norm = normalize(line)
    for title in SECTION_TITLES:
        if norm.lower().startswith(title.lower()):
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
# Stage 1 — split raw text into chapters
# ---------------------------------------------------------------------------


def split_into_chapters(text: str) -> list[tuple[int, str, list[str]]]:
    """
    Returns list of (chapter_number, chapter_title, lines).
    Lines before the first chapter heading are discarded (front matter).
    """
    chapters: list[tuple[int, str, list[str]]] = []
    current: Optional[tuple[int, str, list[str]]] = None

    for raw_line in text.splitlines():
        result = is_chapter_heading(raw_line)
        if result:
            if current:
                chapters.append(current)
            num, title = result
            current = (num, title, [])
        elif current is not None:
            current[2].append(raw_line)

    if current:
        chapters.append(current)

    return chapters


# ---------------------------------------------------------------------------
# Stage 2 — split chapter lines into sections
# ---------------------------------------------------------------------------


def split_into_sections(lines: list[str]) -> list[tuple[str, list[str]]]:
    """
    Returns list of (section_title, lines) for a single chapter's lines.
    Lines before the first known section heading are discarded.
    """
    sections: list[tuple[str, list[str]]] = []
    current: Optional[tuple[str, list[str]]] = None

    for line in lines:
        title = match_section_title(line)
        if title:
            if current:
                sections.append(current)
            current = (title, [])
        elif current is not None:
            current[1].append(line)

    if current:
        sections.append(current)

    return sections


# ---------------------------------------------------------------------------
# Stage 3 — parse individual sections
# ---------------------------------------------------------------------------


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
            # continuation line of the current item
            current_parts.append(line)
        # blank lines between items — just skip

    flush()
    return items


def parse_questions_section(lines: list[str]) -> Section:
    sec = Section(title="Questions à discuter")
    sec.items = collect_numbered_items(lines)
    return sec


def parse_choses_section(lines: list[str]) -> Section:
    """
    Choses à apprendre contains subsection headings (questions ending in '?'
    or descriptive headings) each followed by body text or numbered items.
    """
    sec = Section(title="Choses à apprendre")

    # Split lines into subsection blocks by heading detection
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
                # text before first heading — attach to section body
                sec.body += (" " + line) if sec.body else line

    if current_title is not None:
        blocks.append((current_title, current_block))

    for title, block_lines in blocks:
        sub = Subsection(title=title)
        items = collect_numbered_items(block_lines)
        if items:
            sub.items = items
        else:
            sub.body = normalize(" ".join(normalize(line) for line in block_lines if normalize(line)))
        sec.subsections.append(sub)

    return sec


def parse_bible_section(lines: list[str]) -> Section:
    """
    Ce que dit la Bible contains subsections headed by 'Lisez X.Y-Z'
    each followed by numbered discussion questions.
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
        # lines before first Lisez are discarded

    if current_ref is not None:
        blocks.append((current_ref, current_block))

    for ref, block_lines in blocks:
        sub = Subsection(title=ref)
        items = collect_numbered_items(block_lines)
        if items:
            sub.items = items
        else:
            sub.body = normalize(" ".join(normalize(line) for line in block_lines if normalize(line)))
        sec.subsections.append(sub)

    return sec


def parse_generic_section(title: str, lines: list[str]) -> Section:
    """Fallback: collect body text and any numbered items."""
    sec = Section(title=title)
    items = collect_numbered_items(lines)
    if items:
        sec.items = items
    else:
        sec.body = normalize(" ".join(normalize(line) for line in lines if normalize(line)))
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


# ---------------------------------------------------------------------------
# Top-level parse
# ---------------------------------------------------------------------------


def parse(text: str) -> Book:
    book = Book()

    for num, title, chapter_lines in split_into_chapters(text):
        chapter = Chapter(number=num, title=title)

        for sec_title, sec_lines in split_into_sections(chapter_lines):
            section = parse_section(sec_title, sec_lines)
            chapter.sections.append(section)

        book.chapters.append(chapter)

    return book


def parse_file(txt_path: str) -> Book:
    text = Path(txt_path).read_text(encoding="utf-8")
    return parse(text)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def book_to_dict(book: Book) -> dict:
    def item_d(i: NumberedItem):
        return {"number": i.number, "text": i.text}

    def sub_d(s: Subsection):
        d: dict = {"title": s.title}
        if s.items:
            d["items"] = [item_d(i) for i in s.items]
        if s.body:
            d["body"] = s.body
        return d

    def sec_d(s: Section):
        d: dict = {"title": s.title}
        if s.subsections:
            d["subsections"] = [sub_d(sub) for sub in s.subsections]
        if s.items:
            d["items"] = [item_d(i) for i in s.items]
        if s.body:
            d["body"] = s.body
        return d

    def chap_d(c: Chapter):
        return {"number": c.number, "title": c.title, "sections": [sec_d(s) for s in c.sections]}

    return {"title": book.title, "chapters": [chap_d(c) for c in book.chapters]}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
