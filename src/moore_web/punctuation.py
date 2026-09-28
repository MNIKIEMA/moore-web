"""Normalize the punctuation of reviewed sentence pairs at dataset build time.

The rules and the decisions behind them are in docs/punctuation-logbook.md.
They apply to one side at a time and never compare French with Mooré, so a
pair keeps its translation choices (``!`` vs ``.``); what they fix is
typography and the guillemets left alone on a line by sentence splitting.

Order matters: spacing first, so the later patterns only see one form.
"""

from __future__ import annotations

import re

TERMINAL = "?!.…"

# P1: one space before ? ! : ; and inside guillemets, none before . and ,
_SPACE_BEFORE_HIGH = re.compile(r"(?<=[^\s?!:;.…«])\s*([?!;]|:(?!\d))")
_SPACE_AFTER_OPEN = re.compile(r"«\s*")
_SPACE_BEFORE_CLOSE = re.compile(r"\s*»")
_SPACE_BEFORE_LOW = re.compile(r"\s+([.,])(?!\.)")
_SPACE_AFTER_COMMA = re.compile(r",(?=[^\W\d_])")
_SPACES = re.compile(r"[ \t  ]+")

# P2: sentence punctuation written after the closing guillemet
_PUNCT_AFTER_CLOSE = re.compile(r"(?P<last>\S) » (?P<punct>[?!])|(?P<last2>\S) »(?P<dot>\.)(?!\.)")
# Double terminal punctuation left behind, e.g. "! !" or ". !"
_DOUBLE_TERMINAL = re.compile(r"([?!.…]) ?([?!.])(?![.])")


def fix_spacing(text: str) -> str:
    """P1: ``word ?``, ``word :``, ``« word »``, ``word, word``; single ordinary spaces."""
    text = _SPACES.sub(" ", text)
    text = _SPACE_AFTER_OPEN.sub("« ", text)
    text = _SPACE_BEFORE_CLOSE.sub(" »", text)
    text = _SPACE_BEFORE_HIGH.sub(r" \1", text)
    text = _SPACE_BEFORE_LOW.sub(r"\1", text)
    text = _SPACE_AFTER_COMMA.sub(", ", text)
    return text.strip()


def drop_orphan_guillemets(text: str) -> str:
    """P3: remove a « or » whose partner is not on the same line.

    Sentence splitting leaves the first piece of a quote with only «, the last
    with only ». Straight quotes are not touched.
    """
    open_at: list[int] = []
    orphans: set[int] = set()
    for index, char in enumerate(text):
        if char == "«":
            open_at.append(index)
        elif char == "»":
            if open_at:
                open_at.pop()
            else:
                orphans.add(index)
    orphans.update(open_at)
    if not orphans:
        return text
    kept = "".join(char for index, char in enumerate(text) if index not in orphans)
    kept = fix_spacing(kept)
    # "vient ! »!" loses its », leaving "vient ! !": keep one mark, preferring ? or ! over .
    return _DOUBLE_TERMINAL.sub(lambda m: m.group(1) if m.group(1) != "." else m.group(2), kept)


def _is_embedded(text: str, close_at: int) -> bool:
    """True when the quote closing at ``close_at`` sits inside a running sentence.

    A quote that starts the line, or follows ``:``, ``;`` or a speech-tag comma,
    is a full sentence of dialogue; anything else (``s'appelle « patagsde ».``)
    is embedded and keeps its punctuation outside.
    """
    open_at = text.rfind("«", 0, close_at)
    if open_at < 0:
        return False
    before = text[:open_at].rstrip()
    return bool(before) and before[-1] not in ":;,"


def move_punct_inside(text: str) -> str:
    """P2: ``« … » ?`` → ``« … ? »`` and ``« … ».`` → ``« … . »`` for dialogue quotes."""

    def fix(match: re.Match) -> str:
        last = match.group("last") or match.group("last2")
        punct = match.group("punct") or match.group("dot")
        close_at = match.start() + len(last) + 1  # "x »" / "x »": » follows one space
        if _is_embedded(text, close_at):
            return match.group(0)
        if last in TERMINAL:  # "! »!" → "! »": the quote already ends its sentence
            return f"{last} »"
        return f"{last}{' ' if punct != '.' else ''}{punct} »"

    return _PUNCT_AFTER_CLOSE.sub(fix, text)


def fix_boundaries(text: str, is_title: bool = False) -> str:
    """P4: every link ends like a sentence and starts with a capital; titles end bare."""
    body = text.rstrip(' »"')
    tail = text[len(body) :]
    if is_title:
        body = body.rstrip(".").rstrip()
    elif body and body[-1] in ",;:":
        body = body[:-1].rstrip() + "."
    elif body and body[-1] not in TERMINAL:
        body += "."
    text = body + tail
    start = len(text) - len(text.lstrip('« "'))
    if start < len(text) and text[start].islower():
        text = text[:start] + text[start].upper() + text[start + 1 :]
    return text


def normalize(text: str, is_title: bool = False) -> str:
    """Apply P1–P4 to one side of a pair."""
    text = fix_spacing(text)
    text = drop_orphan_guillemets(text)
    text = move_punct_inside(text)
    text = fix_boundaries(text, is_title)
    return fix_spacing(text)
