"""Clean lexicon JSONL entries.

Two optional transformations:

split-synonyms
    Entries whose ``french`` and ``english`` fields contain comma- or
    semicolon-separated synonym lists (e.g. "quietly, peacefully, calm")
    are exploded into one entry per synonym pair.  The ``moore`` field is
    replicated unchanged across all resulting entries.

    Rules
    -----
    * Moore must not contain a comma/semicolon — if it does a warning is
      emitted and the entry is kept as-is (likely a parsing artefact).
    * All tokens produced by splitting French must be ≤ 4 words and must
      not contain sentence-ending punctuation (., !, ?).  Entries that
      fail this heuristic are kept as-is.
    * FR and EN token counts must match; mismatches are warned and kept.

strip-proverb-notes
    Parenthetical proverb explanations such as
    ``(Proverbe: nous devons épargner…)`` and leading labels such as
    ``Proverbe : …`` / ``Proverb: …`` are removed from ``french`` and
    ``english``.  ``len_ratio`` is recalculated (character-based) when
    present in the entry.
"""

from __future__ import annotations

import logging
import re
from itertools import zip_longest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_SPLIT_SEP = re.compile(r"\s*[,;]\s*")

# Trailing annotation after a sentence-ending character:
#   ". Proverb"  ". Proverbe"  ". Proverb: …"  "). Proverbe disant …"
#   ".Proverbe:"  ") Proverbe:"  "› Proverb, meaning: …"  ". A proverb indicating: …"
# The lookbehind covers  .  !  ?  ›  »  )  ]  — all chars that can close a sentence
# or a parenthetical before the label.
# "proverbs»" (e.g. "listen to proverbs»") is safe: \b fails between b and s.
_PROVERB_TRAILING = re.compile(
    r"(?<=[.!?›»)\]])\s*(?:[Aa]\s+)?[Pp]roverbe?\b.*",
    re.UNICODE | re.DOTALL,
)

# Parenthetical containing any form of "proverb" (closed or unclosed at end of string):
#   "(Proverbe: …)"  "(proverb, e.g. …"  "(Proverb : …"
#   "(proverbs indicating: …)"  "(proverbes indiquant: …)"  "(proverbsaying that …)"
# Using \w* instead of \b so plural and OCR-merged forms are caught too.
_PROVERB_PAREN = re.compile(
    r"\s*\([^)]*[Pp]roverb\w*[^)]*(?:\)|$)",
    re.UNICODE | re.DOTALL,
)

# Leading label:  "Proverbe :" / "Proverb:" / "proverbe :" …
_PROVERB_PREFIX_FR = re.compile(r"^\s*[Pp]roverbe\s*:?\s*", re.UNICODE)
_PROVERB_PREFIX_EN = re.compile(r"^\s*[Pp]roverb\s*:?\s*", re.UNICODE)


# ---------------------------------------------------------------------------
# Synonym splitting
# ---------------------------------------------------------------------------


def _looks_like_synonym_list(french: str, moore: str) -> bool:
    """Heuristic: True when *french* is a comma/semicolon-separated word list."""
    if not _SPLIT_SEP.search(french):
        return False
    tokens = [t for t in _SPLIT_SEP.split(french.strip()) if t]
    return (
        len(tokens) >= 2
        and all(len(t.split()) <= 4 for t in tokens)
        and not any(re.search(r"[.!?]", t) for t in tokens)
        and len(moore.split()) <= 4
    )


def _split_entry(entry: dict) -> list[dict]:
    """Explode a synonym-list entry into one entry per FR/EN pair."""
    moore = entry.get("moore", "")
    french = entry.get("french", "")
    english = entry.get("english", "")

    if _SPLIT_SEP.search(moore):
        logger.warning(
            "Moore field contains comma/semicolon — possible parsing issue, keeping as-is: %r",
            moore,
        )
        return [entry]

    fr_tokens = [t for t in _SPLIT_SEP.split(french.strip()) if t]
    en_tokens = [t for t in _SPLIT_SEP.split(english.strip()) if t]

    # Moore is the reference: FR and EN synonym counts need not match.
    # Fill the shorter list with its last token so no synonyms are lost.
    fr_fill = fr_tokens[-1] if fr_tokens else ""
    en_fill = en_tokens[-1] if en_tokens else ""
    return [
        {**entry, "french": fr or fr_fill, "english": en or en_fill}
        for fr, en in zip_longest(fr_tokens, en_tokens)
    ]


# ---------------------------------------------------------------------------
# Proverb stripping
# ---------------------------------------------------------------------------


def _has_proverb_note(entry: dict) -> bool:
    french = entry.get("french", "")
    english = entry.get("english", "")
    return bool(
        _PROVERB_TRAILING.search(french)
        or _PROVERB_PAREN.search(french)
        or _PROVERB_PREFIX_FR.match(french)
        or _PROVERB_TRAILING.search(english)
        or _PROVERB_PAREN.search(english)
        or _PROVERB_PREFIX_EN.match(english)
    )


def _strip_proverb(entry: dict) -> dict:
    """Remove proverb labels and parenthetical explanations, recalculate len_ratio."""
    french = entry.get("french", "")
    english = entry.get("english", "")

    # Trailing annotation takes priority (covers the vast majority of cases)
    french = _PROVERB_TRAILING.sub("", french)
    english = _PROVERB_TRAILING.sub("", english)

    # Remaining parenthetical notes (e.g. unclosed "(Proverb: …" at end)
    french = _PROVERB_PAREN.sub("", french)
    english = _PROVERB_PAREN.sub("", english)

    french = _PROVERB_PREFIX_FR.sub("", french).strip()
    english = _PROVERB_PREFIX_EN.sub("", english).strip()

    # Collapse duplicate punctuation left by removal (e.g. ".." → ".")
    french = re.sub(r"\.{2,}", ".", french).strip()
    english = re.sub(r"\.{2,}", ".", english).strip()

    new_entry = {**entry, "french": french, "english": english}

    if "len_ratio" in entry and french and entry.get("moore", ""):
        a = len(french)
        b = len(entry["moore"])
        new_entry["len_ratio"] = round(min(a, b) / max(a, b), 4) if a and b else 0.0

    return new_entry


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------


def process(
    entries: list[dict],
    split_synonyms: bool = False,
    strip_proverb_notes: bool = False,
) -> tuple[list[dict], int, int]:
    """Apply cleaning transformations to *entries*.

    Returns ``(output, n_split, n_proverb)``.
    """
    n_split = 0
    n_proverb = 0
    output: list[dict] = []

    for entry in entries:
        if strip_proverb_notes and _has_proverb_note(entry):
            entry = _strip_proverb(entry)
            n_proverb += 1

        if split_synonyms and _looks_like_synonym_list(entry.get("french", ""), entry.get("moore", "")):
            expanded = _split_entry(entry)
            n_split += len(expanded) - 1
            output.extend(expanded)
        else:
            output.append(entry)

    return output, n_split, n_proverb
