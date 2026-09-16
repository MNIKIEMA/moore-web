"""
Parser library for bilingual (French/Moore) church manual text.

Core pipeline (parse_with_chapters):
    raw text
      → split_text_by_regex()          # soft chapter split by CHAPTER_RE
          → split_and_parse_by_sections()  # section split per chapter
              → _split_into_subsections()  # optional subsection split

Structure per chapter:
  - L'histoire de Kadé  /  Karem-y kibarã
  - Questions à discuter  /  Sõaseg sokdse
  - Choses à apprendre  /  D sẽn segd n zãms bũmb niisi
  - Sketch et chant  /  Reem la yɩɩla
  - Ce que dit la Bible  /  Wẽnnaam sebra sẽn yet bũmb ningã
  - Prier et agir  /  Pʋʋsg la tʋʋmde
"""

import msgspec

import unicodedata
import re
from msgspec import field, Struct
from typing import Optional
from pathlib import Path


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class NumberedItem(Struct):
    number: int
    text: str


class BulletItem(Struct):
    text: str


class Subsection(Struct):
    title: str
    items: list[NumberedItem] = field(default_factory=list)
    bullet_items: list[BulletItem] = field(default_factory=list)
    body: str = ""


class Section(Struct):
    title: str
    subsections: list[Subsection] = field(default_factory=list)
    items: list[NumberedItem] = field(default_factory=list)
    bullet_items: list[BulletItem] = field(default_factory=list)
    body: str = ""
    raw_text: str = ""


class Chapter(Struct):
    number: int
    title: str
    sections: list[Section] = field(default_factory=list)
    raw_text: str = ""


class Book(Struct):
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

MOORE_SECTION_TITLES = [
    "Karem-y kibarã",
    "Kibarã",
    "Sõaseg sokdse",
    "D sẽn segd n zãms bũmb niisi",
    "Bũmb d sẽn tõe n zãmse",
    "Reem",
    "Reem la yɩɩlle",
    "Reem la yɩɩla",
    "Wẽnnaam sebra sẽn yet bũmb ningã",
    "Wẽnnaam sebra sẽn yet bũmb ninga",
    "Wẽnnaam sebra sẽn yet bûmb ninga",
    "Pʋʋsg la tʋʋmde",
    "Pʋʋsog la tʋʋma",
    "Pʋʋsgo la tʋʋma",
]

MOORE_INTRO_SECTION_TITLES = [
    "Yellã yaa bʋgo ?",
    "Sẽn n kẽed ne seb kãngã",
]

# Subsections of "Sẽn n kẽed ne seb kãngã" (appear as "N. Title" or standalone)
MOORE_INTRO_SUBSECTION_TITLES = [
    "Karem-y kibarã",
    "Sõaseg sokdse",
    "D sẽn segd n zãms bũmb niisi",
    "Reem la yɩɩla",
    "Wẽnnaam sebra sẽn yet bũmb ningã",
    "Pʋʋsg la tʋʋmde",
]

FRENCH_INTRO_SECTION_TITLES = [
    "Quel est le problème ?",
    "Comment l'église pourrait-elle répondre à ce problème ?",
    "À propos de ce manuel",
    "Comment utiliser ce manuel",
]

# Subsections of "Comment utiliser ce manuel" (appear as "N. Title" or standalone)
FRENCH_INTRO_SUBSECTION_TITLES = [
    "L'histoire",
    "Questions à discuter",
    "Choses à apprendre",
    "Sketch et chant",
    "Ce que dit la Bible",
    "Prier et agir",
]

CHAPTER_RE = re.compile(r"^(?:chapitre|sak\s+a)\s+(\d+)(?:\s+soaba)?\s*[:\-–]?\s*(.*)$", re.IGNORECASE)

# Name mapping: facilitateur names → Mooré SIDA book standard names
FACILITATEUR_NAME_MAP: dict[str, str] = {
    "Kadé": "Poko",
    "Kaluu": "Mariam",
    "Katiu": "Yembi",
    "Kayaga": "Aminata",
    "Kande": "Poko",
    "Atiana": "Séni",
    "Betaro": "Yõdi",
    "Apiu": "Adama",
    "Atega": "Abdou",
}

_FR_NAME_RE = re.compile(r"\b(" + "|".join(re.escape(k) for k in FACILITATEUR_NAME_MAP) + r")\b")


def replace_facilitateur_names_fr(text: str) -> str:
    """Replace French facilitateur character names with their SIDA equivalents."""
    return _FR_NAME_RE.sub(lambda m: FACILITATEUR_NAME_MAP[m.group(0)], text)


NUMBERED_ITEM_RE = re.compile(r"^\s*(?:\([^)]+\)\s+)?(\d+)\.\s+(.+)")
# A numbered marker alone on its own line, with the item's text starting on
# the next line (e.g. Mooré scripture-reference lists: "1. \nWẽnnaam...").
NUMBERED_ITEM_BARE_RE = re.compile(r"^\s*(?:\([^)]+\)\s+)?(\d+)\.\s*$")
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
    for title in SECTION_TITLES + MOORE_SECTION_TITLES:
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


_BARE_BULLET_RE = re.compile(r"^\s*•\s*$")


def _classify_items_lines(
    lines: list[str],
) -> tuple[list[NumberedItem], list[BulletItem], set[int]]:
    """Single pass that splits `lines` into numbered items, bullet items, and
    the set of line indices consumed by either.

    Both kinds of item are often wrapped across many short PDF lines (or
    split across a block boundary as a bare "N."/"•" with the text starting
    on the next line). Tracking both in one pass -- instead of two
    independent walks that don't know about each other's markers -- means a
    numbered item's continuation stops the moment a bullet marker appears
    (and vice versa), so interleaved lists don't get glued into the wrong
    item or, worse, captured by *both* (previously: a numbered item's
    continuation logic didn't recognize a following "•" as a boundary, so it
    absorbed the bullets into its own text while collect_bullet_items
    independently captured those same bullets again -- duplicating them).
    """
    items: list[NumberedItem] = []
    bullets: list[BulletItem] = []
    consumed: set[int] = set()

    mode: Optional[str] = None  # None | "item" | "bullet"
    current_num: Optional[int] = None
    current_parts: list[str] = []

    def flush() -> None:
        if mode == "item" and current_num is not None and current_parts:
            items.append(NumberedItem(current_num, normalize(" ".join(current_parts))))
        elif mode == "bullet" and current_parts:
            bullets.append(BulletItem(normalize(" ".join(current_parts))))

    for idx, raw in enumerate(lines):
        line = normalize(raw)
        num_m = NUMBERED_ITEM_RE.match(raw)
        bare_num_m = NUMBERED_ITEM_BARE_RE.match(raw)
        bullet_m = BULLET_ITEM_RE.match(raw)
        bare_bullet_m = _BARE_BULLET_RE.match(raw)

        if num_m:
            flush()
            mode, current_num, current_parts = "item", int(num_m.group(1)), [num_m.group(2).strip()]
            consumed.add(idx)
        elif bare_num_m:
            flush()
            mode, current_num, current_parts = "item", int(bare_num_m.group(1)), []
            consumed.add(idx)
        elif bullet_m:
            flush()
            mode, current_parts = "bullet", [bullet_m.group(1).strip()]
            consumed.add(idx)
        elif bare_bullet_m:
            flush()
            mode, current_parts = "bullet", []
            consumed.add(idx)
        elif mode is not None and line:
            current_parts.append(line)
            consumed.add(idx)

    flush()
    return items, bullets, consumed


def collect_numbered_items(lines: list[str]) -> list[NumberedItem]:
    """
    Collect numbered items (possibly multi-line) from a flat list of lines.
    Lines that don't belong to any item are ignored.
    """
    items, _, _ = _classify_items_lines(lines)
    return items


def collect_bullet_items(lines: list[str]) -> list[BulletItem]:
    """
    Collect bullet items (possibly multi-line) from a flat list of lines.
    """
    _, bullets, _ = _classify_items_lines(lines)
    return bullets


def collect_items(lines: list[str]) -> tuple[list[NumberedItem], list[BulletItem]]:
    items, bullets, _ = _classify_items_lines(lines)
    return items, bullets


def _lines_consumed_by_items(lines: list[str]) -> set[int]:
    """Indices of lines absorbed into a numbered item or bullet item.

    Used so callers can exclude every such line from `body` instead of only
    excluding each item's first line, which would otherwise duplicate the
    continuation text into both.
    """
    _, _, consumed = _classify_items_lines(lines)
    return consumed


def _body_from_lines(lines: list[str]) -> str:
    """Join the lines that aren't part of any numbered/bullet item into body text."""
    consumed = _lines_consumed_by_items(lines)
    return normalize(
        " ".join(normalize(ln) for idx, ln in enumerate(lines) if idx not in consumed and normalize(ln))
    )


def _extract_intro_title(text: str, section_patterns: list[re.Pattern]) -> str:
    """Return the last non-empty paragraph before the first section heading."""
    lines = text.splitlines()
    first_section_line = len(lines)
    for i, line in enumerate(lines):
        if any(p.match(normalize(line)) for p in section_patterns):
            first_section_line = i
            break

    pre = "\n".join(lines[:first_section_line])
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", pre) if p.strip()]
    if paragraphs:
        return normalize(" ".join(paragraphs[-1].split()))
    return "Intro"


def _split_into_subsections(
    lines: list[str], patterns: list[re.Pattern], titles: list[str]
) -> list[Subsection]:
    """Split lines into Subsection objects using the given heading patterns."""
    subsections: list[Subsection] = []
    current_title: Optional[str] = None
    current_lines: list[str] = []

    def flush() -> None:
        if current_title is None:
            return
        sub = Subsection(title=current_title)
        sub.items = collect_numbered_items(current_lines)
        sub.bullet_items = collect_bullet_items(current_lines)
        body = _body_from_lines(current_lines)
        if body:
            sub.body = body
        subsections.append(sub)

    for line in lines:
        matched = False
        for pattern, canonical in zip(patterns, titles):
            if pattern.match(normalize(line)):
                flush()
                current_title = canonical
                current_lines = []
                matched = True
                break
        if not matched:
            current_lines.append(line)

    flush()
    return subsections


def _match_heading_line(
    norm_line: str, patterns: list[re.Pattern], titles: list[Optional[str]]
) -> Optional[str]:
    """Return the canonical title if `norm_line` starts with a heading, else None.

    Rejects a match immediately followed by a lowercase letter: that means the
    title text was found as a *prefix of a longer word or phrase* (e.g. Mooré
    "Reem" matching inside "reemd", a running-text verb form), not a real
    heading. A heading is followed by whitespace, punctuation, end of line, or
    another heading glued on with no separator (uppercase).
    """
    for regex, canonical_title in zip(patterns, titles):
        m = regex.match(norm_line)
        if m and not norm_line[m.end() : m.end() + 1].islower():
            return canonical_title or match_section_title(norm_line) or norm_line
    return None


def _lookahead_heading(
    lines: list[str],
    start: int,
    patterns: list[re.Pattern],
    titles: list[Optional[str]],
    max_lookahead: int = 6,
) -> tuple[Optional[str], int]:
    """Match a heading whose text was split across consecutive PDF text blocks
    (e.g. "Wẽnnaam Sebra sẽn yet" / "bũmb ninga" on separate lines, with a
    *blank* line in between from `extract_pdf_blocks` joining each block with
    "\\n\\n" — a block boundary landed mid-heading).

    Grows the accumulated text by skipping blank lines and appending the next
    non-blank one, but only while the result stays a strict prefix of some
    canonical title; stops the moment growth is no longer plausible. Returns
    (canonical_title, raw_lines_consumed_including_blanks) or (None, 0).
    """
    acc = normalize(lines[start])
    idx = start + 1
    end = min(len(lines), start + 1 + max_lookahead)
    while idx < end:
        next_line = normalize(lines[idx])
        idx += 1
        if not next_line:
            continue  # blank separator between PDF blocks -- keep looking
        candidate = f"{acc} {next_line}"
        title = _match_heading_line(candidate, patterns, titles)
        if title:
            return title, idx - start
        if not any(
            t and len(candidate) < len(t) and t.lower().startswith(candidate.lower())
            for t in titles
            if t
        ):
            return None, 0
        acc = candidate
    return None, 0


def split_and_parse_by_sections(
    text: str,
    section_regexes: list[re.Pattern],
    section_titles: list[str] | None = None,
    subsection_map: dict[str, tuple[list[re.Pattern], list[str]]] | None = None,
    stop_before: Optional[re.Pattern] = None,
) -> list[Section]:
    """Split text by section patterns and parse each section's content.

    Args:
        text: Raw text to parse.
        section_regexes: Compiled patterns identifying section headings.
        section_titles: Canonical titles matching each pattern (same length).
        subsection_map: Optional map of section title → (patterns, titles)
            for splitting that section further into subsections.
        stop_before: Optional pattern that acts as a hard stop. When a line
            matches, parsing halts immediately and no further sections are
            collected. Used to exclude back-matter (bibliography, credits)
            that follows the last real section of each language variant:
            Mooré stops at "Tʋʋm teedo" (p. 55), French at
            "Matériels de formation" (p. 57).

    Returns:
        List of Section objects with body text, items, and optional subsections.
    """
    _titles = section_titles if section_titles is not None else [None] * len(section_regexes)

    if len(section_regexes) != len(_titles):
        raise ValueError("section_regexes and section_titles must have the same length")

    def _build_section(title: str, sec_lines: list[str]) -> Section:
        sec = Section(title=title, raw_text="\n".join(sec_lines))
        if subsection_map and title in subsection_map:
            sub_patterns, sub_titles = subsection_map[title]
            sec.subsections = _split_into_subsections(sec_lines, sub_patterns, sub_titles)
        else:
            sec.items, sec.bullet_items = collect_items(sec_lines)
            body = _body_from_lines(sec_lines)
            if body:
                sec.body = body
        return sec

    sections: list[Section] = []
    lines = text.splitlines()

    current_title: Optional[str] = None
    current_heading: Optional[str] = None
    current_lines: list[str] = []

    i = 0
    while i < len(lines):
        line = lines[i]
        norm_line = normalize(line)

        if stop_before and stop_before.match(norm_line):
            break

        canonical_title = _match_heading_line(norm_line, section_regexes, _titles)
        consumed = 1
        if canonical_title is None:
            # The heading may have been split across a PDF block boundary
            # (e.g. "Wẽnnaam Sebra sẽn yet" / "bũmb ninga" on two lines).
            canonical_title, consumed = _lookahead_heading(lines, i, section_regexes, _titles)

        if canonical_title is not None:
            if current_heading is not None or current_lines:
                sections.append(_build_section(current_title or "Intro", current_lines))

            current_heading = line
            current_title = canonical_title
            current_lines = []
            i += consumed
            continue

        current_lines.append(line)
        i += 1

    if current_heading is not None or current_lines:
        sections.append(_build_section(current_title or "Intro", current_lines))

    return sections


def parse_with_chapters(
    text: str,
    section_regexes: list[re.Pattern],
    section_titles: list[str],
    intro_section_regexes: list[re.Pattern] | None = None,
    intro_section_titles: list[str] | None = None,
    intro_subsection_map: dict[str, tuple[list[re.Pattern], list[str]]] | None = None,
    stop_before: Optional[re.Pattern] = None,
) -> Book:
    """Two-pass parse: soft chapter split, then reliable section split per chapter.

    Args:
        text: Raw text to parse.
        section_regexes: Compiled patterns identifying section headings.
        section_titles: Canonical titles matching each pattern (same length).
        intro_section_regexes: Optional patterns for splitting the preamble (Chapter 0).
        intro_section_titles: Canonical titles for intro sections.
        intro_subsection_map: Optional map of intro section title → (patterns, titles).
        stop_before: Forwarded to :func:`split_and_parse_by_sections` for each
            chapter. See that function for details.

    Returns:
        A Book whose chapters each contain parsed sections.
    """
    book = Book()
    chapter_number = 0
    seen: dict[int, Chapter] = {}  # deduplicate by chapter number

    for heading, content in split_text_by_regex(text, CHAPTER_RE):
        if heading is None:
            if not content.strip():
                continue
            if intro_section_regexes:
                intro_title = _extract_intro_title(content, intro_section_regexes)
                ch = Chapter(number=0, title=intro_title, raw_text=content)
                ch.sections = split_and_parse_by_sections(
                    content,
                    intro_section_regexes,
                    intro_section_titles,
                    subsection_map=intro_subsection_map,
                )
            else:
                ch = Chapter(number=0, title="Intro", raw_text=content)
        else:
            res = is_chapter_heading(heading)
            num, title = res if res else (chapter_number + 1, heading.strip())
            chapter_number = num

            if not title:
                first_line = next((ln.strip() for ln in content.splitlines() if ln.strip()), "")
                title = first_line

            ch = Chapter(number=num, title=title, raw_text=content)
            ch.sections = split_and_parse_by_sections(
                content, section_regexes, section_titles, stop_before=stop_before
            )

        if ch.number in seen:
            existing = seen[ch.number]
            if len(ch.sections) >= len(existing.sections):
                book.chapters[book.chapters.index(existing)] = ch
                seen[ch.number] = ch
        else:
            seen[ch.number] = ch
            book.chapters.append(ch)

    return book


# ---------------------------------------------------------------------------
# Output utilities
# ---------------------------------------------------------------------------


def clean(s: str) -> str:
    s = re.sub(r"\n+", "\n", s)
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def flatten_content(
    items: list[NumberedItem],
    bullet_items: list[BulletItem],
    body: str,
) -> list[str]:
    """Flatten numbered items, bullet items, and body text into a list of strings."""
    result: list[str] = []
    if body:
        result.append(clean(body))
    for item in items:
        result.append(clean(item.text))
    for bullet in bullet_items:
        result.append(clean(bullet.text))
    return result


def flatten_section_content(sec: Section | Subsection) -> list[str]:
    """Flatten a Section or Subsection's body, items, and bullet_items."""
    return flatten_content(sec.items, sec.bullet_items, sec.body)


def flatten_book_to_list(book: Book) -> list[str]:
    result: list[str] = []
    for chapter in book.chapters:
        for section in chapter.sections:
            result.extend(flatten_section_content(section))
            for sub in section.subsections:
                result.extend(flatten_section_content(sub))
    return result


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
        return d

    def chap_d(c: Chapter):
        return {
            "number": c.number,
            "title": c.title,
            "sections": [sec_d(s) for s in c.sections],
        }

    return {"title": book.title, "chapters": [chap_d(c) for c in book.chapters]}


def parse_book_from_json(json_path: str) -> Book:
    json_data = Path(json_path).read_text(encoding="utf-8")
    book = msgspec.json.decode(json_data, type=Book)
    return book
