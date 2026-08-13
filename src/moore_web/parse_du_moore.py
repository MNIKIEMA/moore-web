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
from pathlib import Path
import pdfplumber

PDFS = [
    ("Du_Moore_au_Francais_1_Noir_et_Blanc_pp_01-30_Lecons_1-16.pdf", 1),
    ("Du_Moore_au_Francais_2_Noir_et_Blanc_pp_31-60_Lecons_17-31.pdf", 2),
    ("Du_Moore_au_Francais_3_Noir_et_Blanc_pp.61-94_Lecons_32-48.pdf", 3),
]

LESSON_RE = re.compile(r"[Kk]aoren[ɡg].*soaba")
SUBTITLE_RE = re.compile(r"\bkaorengo\b|\bkaorenɡo\b|\bkarem", re.IGNORECASE)

CONV_HEADERS = {"D ɡom fãrende", "Kʋmbɡo"}
CONV_STOP = {
    "Expression libre",
    "Gʋlsɡo",
    "D bãnɡ n ɡʋls ne fãrende",
    "Bãnɡ n ɡʋls ne fãrende",
    "Bãnɡ n ɡʋls",
}
CONV_SKIP = {"wilɡ", "fãrendẽ wã", "à ne ɡeonf", "sũk bɩ"}

PASSAGE_STOP = {"Questions de compréhension", "Ecriture", "E criture", "Copie", "C opie"}

Y_TOL = 5


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def page_lines(page) -> list[tuple[int, str]]:
    """Return [(y_bucket, line_text)] sorted by y."""
    words = page.extract_words(x_tolerance=5, y_tolerance=Y_TOL)
    buckets: dict[int, list[str]] = {}
    for w in words:
        y = round(w["top"] / Y_TOL) * Y_TOL
        buckets.setdefault(y, []).append(w["text"])
    return [(y, " ".join(ws)) for y, ws in sorted(buckets.items())]


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


def pair_lesson_pages(pdf) -> list[tuple[list, list]]:
    """
    Return (fr_lines, mos_lines) pairs by matching consecutive pages that
    share the same 'Kaoreng … soaba' header.  First page of each pair is
    French, second is Mooré — consistent across all three books.
    """
    tagged = []
    for page in pdf.pages:
        lines = page_lines(page)
        header = get_lesson_header(lines)
        if header:
            tagged.append((header, lines))

    pairs = []
    i = 0
    while i < len(tagged):
        if i + 1 < len(tagged) and tagged[i][0] == tagged[i + 1][0]:
            pairs.append((tagged[i][1], tagged[i + 1][1]))
            i += 2
        else:
            i += 1  # unpaired lesson page — skip
    return pairs


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
        # Positional fallback: if subtitle not yet seen, this line is it
        if not subtitle_seen:
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


def _apply_prefix(prefix: str | None, text: str) -> str:
    if not prefix:
        return text
    sep = "" if len(prefix) == 1 else " "
    return prefix + sep + text


def _fix_dropcap_space(text: str) -> str:
    """'L e bébé' → 'Le bébé' (French drop-cap merged with a spurious space)."""
    return re.sub(r"^([A-Za-zÀ-ÿ]) ([a-zà-ÿ])", lambda m: m.group(1) + m.group(2), text)


def extract_sentences_sectioned(lines: list[tuple[int, str]], fix_dropcap: bool = False) -> list[str]:
    """Section ②: parallel sentences in order."""
    sents = []
    in_sec = False
    prefix: str | None = None
    prev_y: int = 0

    for y, text in lines:
        if not in_sec and re.fullmatch(r"[A-Za-zÀ-ÿ]{1,4}", text.strip()):
            prefix = text.strip()
            prev_y = y
            continue

        if "②" in text:
            in_sec = True
            remainder = re.sub(r"^②\s*", "", text).strip()
            remainder = re.sub(r"^\d+\s*[–\-]\s*", "", remainder).strip()
            if prefix is not None and y - prev_y <= 10:
                remainder = _apply_prefix(prefix, remainder)
            prefix = None
            if fix_dropcap:
                remainder = _fix_dropcap_space(remainder)
            if len(remainder) > 2:
                sents.append(remainder)
            prev_y = y
            continue

        if in_sec:
            if any(m in text for m in CONV_HEADERS | {"Expression libre"}):
                break
            t = text.strip()
            if re.fullmatch(r"[A-Za-zÀ-ÿ]{1,4}", t):
                prefix = t
                prev_y = y
                continue
            t = re.sub(r"^\d+\s*[–\-]\s*", "", t)
            if prefix is not None and y - prev_y <= 10:
                t = _apply_prefix(prefix, t)
            prefix = None
            if fix_dropcap:
                t = _fix_dropcap_space(t)
            if len(t) > 2:
                sents.append(t)

        prev_y = y
    return sents


def extract_conversation(lines: list[tuple[int, str]]) -> list[str]:
    """D gom fãrende / Kʋmbɡo section: drill sentences."""
    sents = []
    in_sec = False
    for _, text in lines:
        t = text.strip()
        if any(m in t for m in CONV_HEADERS):
            in_sec = True
            continue
        if not in_sec:
            continue
        if any(m in t for m in CONV_STOP):
            break
        if any(s in t for s in CONV_SKIP) or len(t) < 5:
            continue
        if t.startswith("Makre"):
            after = t[t.find(":") + 1 :].strip() if ":" in t else ""
            for sent in re.split(r"\.\s+", after):
                sent = sent.strip().rstrip(".")
                if sent:
                    sents.append(sent + ".")
            continue
        if len(t) > 5:
            sents.append(t)
    return sents


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
      4. Collect full sentences; merge wrapped continuation lines.
      5. Stop at comprehension questions / writing markers.
    """
    sents: list[str] = []
    past_vocab = False
    in_passage = False

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
        if not in_passage and len(t) <= 20:
            continue
        # First long non-numbered line marks passage start
        in_passage = True
        if len(t) > 1:
            if sents and not sents[-1].endswith((".", "!", "?")):
                sents[-1] += " " + t  # merge wrapped line
            else:
                sents.append(t)
    return sents


# ---------------------------------------------------------------------------
# Top-level parser
# ---------------------------------------------------------------------------


def parse_pdf(path: Path, book_num: int) -> list[dict]:
    records = []
    with pdfplumber.open(path) as pdf:
        pairs = pair_lesson_pages(pdf)

    for lesson, (fr_lines, mos_lines) in enumerate(pairs, start=1):
        src = f"Du_Moore_{book_num}"

        # Key sentence (same logic for all layouts)
        fr_k = extract_key(fr_lines)
        mos_k = extract_key(mos_lines)
        if fr_k and mos_k:
            records.append({"fr": fr_k, "mos": mos_k, "source": src, "lesson": lesson, "section": "key"})

        if has_section_markers(fr_lines):
            # --- sectioned layout ---
            fr_v = extract_vocab_sectioned(fr_lines)
            mos_v = extract_vocab_sectioned(mos_lines)
            for num in sorted(set(fr_v) & set(mos_v)):
                records.append(
                    {
                        "fr": fr_v[num],
                        "mos": mos_v[num],
                        "source": src,
                        "lesson": lesson,
                        "section": "vocab",
                        "item": num,
                    }
                )

            fr_s = extract_sentences_sectioned(fr_lines, fix_dropcap=True)
            mos_s = extract_sentences_sectioned(mos_lines, fix_dropcap=False)
            for j, (f, m) in enumerate(zip(fr_s, mos_s)):
                records.append(
                    {
                        "fr": f,
                        "mos": m,
                        "source": src,
                        "lesson": lesson,
                        "section": "sentences",
                        "item": j + 1,
                    }
                )

            fr_c = extract_conversation(fr_lines)
            mos_c = extract_conversation(mos_lines)
            for j, (f, m) in enumerate(zip(fr_c, mos_c)):
                records.append(
                    {
                        "fr": f,
                        "mos": m,
                        "source": src,
                        "lesson": lesson,
                        "section": "conversation",
                        "item": j + 1,
                    }
                )

        else:
            # --- prose layout ---
            fr_v = extract_vocab_prose(fr_lines)
            mos_v = extract_vocab_prose(mos_lines)
            for num in sorted(set(fr_v) & set(mos_v)):
                records.append(
                    {
                        "fr": fr_v[num],
                        "mos": mos_v[num],
                        "source": src,
                        "lesson": lesson,
                        "section": "vocab",
                        "item": num,
                    }
                )

            fr_p = extract_passage(fr_lines)
            mos_p = extract_passage(mos_lines)
            for j, (f, m) in enumerate(zip(fr_p, mos_p)):
                records.append(
                    {"fr": f, "mos": m, "source": src, "lesson": lesson, "section": "passage", "item": j + 1}
                )

    return records


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
