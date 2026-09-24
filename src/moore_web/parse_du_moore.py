#!/usr/bin/env python3
"""
Extract parallel French-Mooré sentence pairs from "Du Moore au Français" PDFs.

Each lesson occupies two consecutive pages with the same "Kaoreng … soaba"
header.  The first page is always French, the second always Mooré.

Two lesson layouts exist across the three books:

  sectioned  — lessons with ① (vocab) and ② (sentences) markers
               + a conversation/drill section
  prose      — later lessons (book 3 lessons 40-48) with a plain numbered
               vocab list and a reading passage instead of ①②
"""

import re
import json
import argparse
from dataclasses import dataclass
from pathlib import Path
import pdfplumber

PDFS = [
    ("Du_Moore_au_Francais_1_Noir_et_Blanc_pp_01-30_Lecons_1-16.pdf", 1),
    ("Du_Moore_au_Francais_2_Noir_et_Blanc_pp_31-60_Lecons_17-31.pdf", 2),
    ("Du_Moore_au_Francais_3_Noir_et_Blanc_pp.61-94_Lecons_32-48.pdf", 3),
]

LESSON_RE = re.compile(r"[Kk]aoren[ɡg].*soaba")
SUBTITLE_RE = re.compile(r"\bkaorengo\b|\bkaorenɡo\b|\bkarem", re.IGNORECASE)

_SECTION2_STOP = {"D ɡom fãrende", "Kʋmbɡo", "Expression libre"}

PASSAGE_STOP = {"Questions de compréhension", "Ecriture", "E criture", "Copie", "C opie"}

Y_TOL = 5
BASELINE_TOL = 5  # max bottom-edge step between neighbouring words of one line (line pitch is >= 17 px)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


_GLUED_CAP_RE = re.compile(r"(?<=[^\sA-ZÀ-Ý])[A-ZÀ-Ý]$")
_INNER_CAP_RE = re.compile(r"(?<=[a-zà-ÿ.,])(?=[A-ZÀ-Ý][a-zà-ÿ])")


def _is_dropcap(w: dict, nxt: dict) -> bool:
    """A word ending in a capital noticeably taller than the non-capital word right after it."""
    h, nh = w["bottom"] - w["top"], nxt["bottom"] - nxt["top"]
    return (
        w["text"][-1:].isalpha()
        and w["text"][-1].isupper()
        and not nxt["text"][:1].isupper()
        and h > 1.15 * nh
        and nxt["x0"] - w["x1"] < 8
    )


def _join_words(ws: list[dict]) -> str:
    """
    Join words with spaces, reattaching drop caps to their word.  pdfplumber
    either leaves the drop cap alone ('C', 'écile') or glues it to the
    previous token ('deC', 'éline'); both become '... Cécile'.
    """
    out = ""
    for i, w in enumerate(ws):
        t = w["text"]
        nxt = ws[i + 1] if i + 1 < len(ws) else None
        if nxt and _is_dropcap(w, nxt) and (len(t) == 1 or _GLUED_CAP_RE.search(t)):
            t = (t[:-1] + " " if len(t) > 1 else "") + t[-1] + nxt["text"]
            ws[i + 1] = {**nxt, "text": ""}
        elif nxt and w["bottom"] - w["top"] > 1.15 * (nxt["bottom"] - nxt["top"]):
            # drop cap glued on both sides: 'voisineCaroline' → 'voisine Caroline'
            t = _INNER_CAP_RE.sub(" ", t)
        if t:
            out += (" " if out else "") + t
    return out


def page_lines(page) -> list[tuple[int, str]]:
    """
    Return [(y, line_text)] sorted by y.

    Words are clustered by their bottom edge rather than snapped to a fixed
    grid: drop caps and item numbers sit a few px higher than the text they
    belong to, but share (almost) the same baseline.  A fixed grid split such
    lines in two whenever they straddled a bucket boundary.
    """
    words = sorted(page.extract_words(x_tolerance=5, y_tolerance=Y_TOL), key=lambda w: w["bottom"])
    clusters: list[list[dict]] = []
    for w in words:
        if clusters and w["bottom"] - clusters[-1][-1]["bottom"] <= BASELINE_TOL:
            clusters[-1].append(w)
        else:
            clusters.append([w])
    lines = []
    for ws in clusters:
        ws.sort(key=lambda w: w["x0"])
        lines.append((round(min(w["top"] for w in ws)), _join_words(ws)))
    return sorted(lines)


def get_lesson_header(lines: list[tuple[int, str]]) -> str | None:
    """Return the normalised lesson header line, or None."""
    for _, text in lines:
        if LESSON_RE.search(text) and "soaba" in text:
            return re.sub(r"\s+", " ", text.strip())
    return None


def page_full_text(lines: list[tuple[int, str]]) -> str:
    return " ".join(t for _, t in lines)


def has_section_markers(lines: list[tuple[int, str]]) -> bool:
    full = page_full_text(lines)
    return "①" in full or "②" in full


_LESSON_NO_RE = re.compile(r"\((\d+)\)")


def lesson_number(header: str) -> int | None:
    """Printed lesson number from 'Kaoreng … (30) soaba'."""
    m = _LESSON_NO_RE.search(header)
    return int(m.group(1)) if m else None


def pair_tagged_pages(tagged: list[tuple[str, list]]) -> list[tuple[int | None, list, list]]:
    """
    Pair consecutive (header, lines) pages of the same lesson → (number, fr_lines, mos_lines).

    Pages are matched on the printed lesson number, not the whole header:
    headers carry typos between the two pages ('kaoreng'/'Kaoreng',
    'pisi a la ye'/'pisi la a ye') that used to drop lessons 3 and 21.
    """
    keyed = [(lesson_number(h) or h, lines) for h, lines in tagged]
    pairs = []
    i = 0
    while i < len(keyed):
        if i + 1 < len(keyed) and keyed[i][0] == keyed[i + 1][0]:
            key = keyed[i][0]
            pairs.append((key if isinstance(key, int) else None, keyed[i][1], keyed[i + 1][1]))
            i += 2
        else:
            i += 1  # unpaired lesson page — skip
    return pairs


def pair_lesson_pages(pdf) -> list[tuple[int | None, list, list]]:
    """
    Return (lesson_number, fr_lines, mos_lines) for consecutive pages of the
    same lesson.  First page of each pair is French, second is Mooré —
    consistent across all three books.
    """
    tagged = []
    for page in pdf.pages:
        lines = page_lines(page)
        header = get_lesson_header(lines)
        if header:
            tagged.append((header, lines))
    return pair_tagged_pages(tagged)


# ---------------------------------------------------------------------------
# Key sentence
# ---------------------------------------------------------------------------


def extract_key(lines: list[tuple[int, str]]) -> str | None:
    """
    First sentence-like line after the lesson header and subtitle.

    The subtitle is always the first non-short, non-header line after the
    lesson header.  SUBTITLE_RE catches most subtitles explicitly; for the
    rest we use a positional skip (skip once, then collect).
    """
    past_header = False
    subtitle_seen = False  # True once the subtitle has been consumed

    for _, text in lines:
        t = text.strip()
        if LESSON_RE.search(t):
            past_header = True
            subtitle_seen = False
            continue
        if not past_header:
            continue
        if len(t) <= 4:
            continue
        # Explicit subtitle match — mark seen and keep scanning
        if SUBTITLE_RE.search(t):
            subtitle_seen = True
            continue
        # Positional fallback: if subtitle not yet seen, this line is it —
        # unless it is already a full sentence (lesson 47's MOS page has no subtitle)
        if not subtitle_seen and not _SENT_END_RE.search(t):
            subtitle_seen = True
            continue
        # Stop at section markers or numbered vocab start
        if re.match(r"[①②]", t) or re.match(r"1\s*[-–]", t):
            break
        if len(t) > 8 and " " in t:
            return t
    return None


# ---------------------------------------------------------------------------
# Sectioned layout: ① vocab  ②  sentences  +  conversation
# ---------------------------------------------------------------------------


def _parse_numbered_items(text: str) -> dict[int, str]:
    """
    Split 'N – word  M – word' into {N: word, M: word} using re.split so
    that two-digit item numbers like '17' are never partially consumed.
    """
    parts = re.split(r"(\d+)\s*[–\-]\s*", text)
    items: dict[int, str] = {}
    i = 1  # parts[0] is the prefix before the first number
    while i + 1 < len(parts):
        num = int(parts[i])
        val = parts[i + 1].strip()
        if val and re.search(r"[^\W\d]", val):
            items[num] = val
        i += 2
    return items


def extract_vocab_sectioned(lines: list[tuple[int, str]]) -> dict[int, str]:
    """Section ①: numbered items → {number: text}."""
    items: dict[int, str] = {}
    in_sec = False
    for _, text in lines:
        if "①" in text:
            in_sec = True
        if in_sec and "②" in text:
            break
        if in_sec:
            items.update(_parse_numbered_items(text))
    return items


_SENT_END_RE = re.compile(r"[.!?…][\"»”’)\s]*$")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-ÝƐƆƖƲ])")


def _merge_wrapped(chunks: list[str]) -> list[str]:
    """
    Rebuild sentences from PDF lines: a line without terminal punctuation
    continues on the next, and a line holding several sentences is split.
    """
    sents: list[str] = []
    for t in chunks:
        if sents and not _SENT_END_RE.search(sents[-1]):
            sents[-1] += " " + t
        else:
            sents.append(t)
    return [part for s in sents for part in _SENT_SPLIT_RE.split(s)]


def extract_sentences_sectioned(lines: list[tuple[int, str]]) -> list[str]:
    """
    Section ②: parallel sentences in order.

    French and Mooré pages wrap long sentences at different points, so lines
    are merged into sentences before pairing by index.
    """
    chunks: list[str] = []
    in_sec = False
    for _, text in lines:
        if "②" in text:
            in_sec = True
            text = text.split("②", 1)[1]
        elif not in_sec:
            continue
        elif any(m in text for m in _SECTION2_STOP):
            break
        t = re.sub(r"^\d+\s*[–\-]\s*", "", text.strip())
        if len(t) > 2:
            chunks.append(t)
    return _merge_wrapped(chunks)


# ---------------------------------------------------------------------------
# Prose layout: plain numbered vocab  +  reading passage
# ---------------------------------------------------------------------------


def extract_vocab_prose(lines: list[tuple[int, str]]) -> dict[int, str]:
    """
    Numbered vocab list without a ① marker (prose-layout lessons).
    Items use 'N - word' or 'N – word' format.
    """
    items: dict[int, str] = {}
    in_sec = False
    for _, text in lines:
        if not in_sec and re.match(r"1\s*[-–]", text.strip()):
            in_sec = True
        if not in_sec:
            continue
        if not re.search(r"\d+\s*[-–]", text):
            if len(text.strip()) > 20:
                break  # end of vocab, passage starts
            continue  # short interstitial line (e.g. drop-cap fragment)
        items.update(_parse_numbered_items(text))
    return items


def extract_passage(lines: list[tuple[int, str]]) -> list[str]:
    """
    Reading passage sentences from prose-layout lessons.

    Strategy:
      1. Skip everything until the numbered vocab list starts.
      2. Skip all numbered vocab lines.
      3. Skip short non-sentence lines (titles, drop-cap fragments ≤ 20 chars).
      4. Rebuild sentences: merge wrapped lines, split lines holding several.
      5. Stop at comprehension questions / writing markers.
    """
    chunks: list[str] = []
    past_vocab = False

    for _, text in lines:
        t = text.strip()
        if any(m in t for m in PASSAGE_STOP):
            break
        # Detect start of numbered vocab
        if not past_vocab and re.match(r"1\s*[-–]", t):
            past_vocab = True
        if not past_vocab:
            continue
        # Skip numbered vocab lines
        if re.search(r"\d+\s*[-–]", t):
            continue
        # Before passage: skip short lines (titles, interstitials)
        if not chunks and len(t) <= 20:
            continue
        if len(t) > 1:
            chunks.append(t)
    return _merge_wrapped(chunks)


# ---------------------------------------------------------------------------
# Per-lesson extraction
# ---------------------------------------------------------------------------


@dataclass
class Lesson:
    """Everything extracted from one lesson's FR/MOS page pair, unpaired and unfiltered."""

    book: int
    number: int | None  # printed lesson number, 1–48 across the three books
    key: tuple[str | None, str | None]
    vocab: tuple[dict[int, str], dict[int, str]]
    sentences: tuple[list[str], list[str]]  # section ② (sectioned layout)
    passage: tuple[list[str], list[str]]  # reading text (prose layout)


def extract_lessons(path: Path, book_num: int) -> list[Lesson]:
    with pdfplumber.open(path) as pdf:
        pairs = pair_lesson_pages(pdf)

    lessons = []
    for number, fr_lines, mos_lines in pairs:
        key = (extract_key(fr_lines), extract_key(mos_lines))
        if has_section_markers(fr_lines):
            vocab = (extract_vocab_sectioned(fr_lines), extract_vocab_sectioned(mos_lines))
            sentences = (extract_sentences_sectioned(fr_lines), extract_sentences_sectioned(mos_lines))
            passage: tuple[list[str], list[str]] = ([], [])
        else:
            vocab = (extract_vocab_prose(fr_lines), extract_vocab_prose(mos_lines))
            sentences = ([], [])
            passage = (extract_passage(fr_lines), extract_passage(mos_lines))
        lessons.append(Lesson(book_num, number, key, vocab, sentences, passage))
    return lessons


# ---------------------------------------------------------------------------
# Outputs: sentence pairs (JSONL) and review units
# ---------------------------------------------------------------------------


def lesson_pairs(lesson: Lesson) -> list[dict]:
    """
    Sentence pairs for the JSONL.  Only pairs that are safe without review:
    vocab by item number, and ② sentences only when both sides have the
    same count (otherwise zip would shift every pair).  Passages are left
    out: they are free translations, and even equal counts misalign
    (lesson 48: FR 1 = MOS 1+2, FR 3+4 = MOS 4).  They go through review.
    """
    base = {"source": f"Du_Moore_{lesson.book}", "lesson": lesson.number}
    records = []
    fr_k, mos_k = lesson.key
    if fr_k and mos_k:
        records.append({"fr": fr_k, "mos": mos_k, **base, "section": "key"})

    fr_v, mos_v = lesson.vocab
    for num in sorted(set(fr_v) & set(mos_v)):
        records.append({"fr": fr_v[num], "mos": mos_v[num], **base, "section": "vocab", "item": num})

    fr_s, mos_s = lesson.sentences
    if len(fr_s) != len(mos_s):
        print(f"  [warn] lesson {lesson.number}: {len(fr_s)} fr vs {len(mos_s)} mos sentences, skipped")
        fr_s = mos_s = []
    for j, (f, m) in enumerate(zip(fr_s, mos_s), start=1):
        records.append({"fr": f, "mos": m, **base, "section": "sentences", "item": j})
    return records


def parse_pdf(path: Path, book_num: int) -> list[dict]:
    return [r for lesson in extract_lessons(path, book_num) for r in lesson_pairs(lesson)]


def lesson_review_units(lesson: Lesson) -> list[tuple[str, list[str], list[str]]]:
    """
    Review units for one lesson: (unit_id, fra_lines, mos_lines) per non-empty
    section, nothing dropped.  Uneven sides are left for the reviewer.

    Vocab rows matched by item number come first so they line up; items
    found on one side only are appended at the end of that side.
    """
    prefix = f"du-moore-{lesson.number:02d}" if lesson.number else f"du-moore-b{lesson.book}"
    fr_v, mos_v = lesson.vocab
    both = sorted(set(fr_v) & set(mos_v))
    sections = {
        "key": tuple([t] if t else [] for t in lesson.key),
        "vocab": (
            [fr_v[n] for n in both] + [fr_v[n] for n in sorted(set(fr_v) - set(mos_v))],
            [mos_v[n] for n in both] + [mos_v[n] for n in sorted(set(mos_v) - set(fr_v))],
        ),
        "sentences": lesson.sentences,
        "passage": lesson.passage,
    }
    return [(f"{prefix}-{name}", list(fr), list(mos)) for name, (fr, mos) in sections.items() if fr or mos]


def review_units(pdf_dir: Path) -> list[tuple[str, list[str], list[str]]]:
    """Review units for every lesson of the three books found in pdf_dir, in book order."""
    units = []
    for fname, book_num in PDFS:
        path = pdf_dir / fname
        if not path.exists():
            print(f"  [skip] {fname} not found")
            continue
        for lesson in extract_lessons(path, book_num):
            units.extend(lesson_review_units(lesson))
    return units


def main():
    parser = argparse.ArgumentParser(description="Extract parallel fr-mos pairs from Du Moore PDFs")
    parser.add_argument("--dir", default=".", help="Directory containing the PDFs")
    parser.add_argument("--output", default="du_moore_parallel.jsonl")
    args = parser.parse_args()

    base = Path(args.dir)
    all_records = []

    for fname, book_num in PDFS:
        path = base / fname
        if not path.exists():
            print(f"  [skip] {fname} not found")
            continue
        records = parse_pdf(path, book_num)
        by_section: dict[str, int] = {}
        for r in records:
            by_section[r["section"]] = by_section.get(r["section"], 0) + 1
        print(f"Book {book_num}: {len(records):>4} pairs  {by_section}")
        all_records.extend(records)

    with open(args.output, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nTotal: {len(all_records)} pairs → {args.output}")


if __name__ == "__main__":
    main()
