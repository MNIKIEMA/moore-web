"""Flatten bilingual structured sources into parallel (French, Mooré) sentence lists.

Each flattener returns a :class:`ParallelText` that can be serialised to JSON
and passed to ``moore_web.align_corpus`` for alignment.

Supported sources
-----------------
- SIDA bilingual book  (``flatten_sida_book``)
- Kadé facilitateur books (``flatten_facilitateur_pair``) — two monolingual parsed books
- Segmented news entries  (``flatten_news_entries``)

Sentence segmentation
---------------------
- French : ``syntok`` segmenter (rule-based, good for European languages)
- Mooré  : punctuation-boundary split — syntok has no Mooré support
- Moses tokeniser (``sacremoses``) normalises French spacing; falls back to regex

Lengths of ``french`` and ``moore`` lists will typically differ — the aligner
handles many-to-many alignment.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import msgspec

if TYPE_CHECKING:
    from moore_web.book_parser import Chapter as SidaChapter
    from moore_web.book_parser_facilitateur import Book

_NEWLINE_RE = re.compile(r"\n+")
_MULTI_SPACE_RE = re.compile(r" {2,}")
_SENT_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


# ---------------------------------------------------------------------------
# ParallelText
# ---------------------------------------------------------------------------


class ParallelText(msgspec.Struct):
    """Parallel sentence lists ready for alignment."""

    french: list[str] = msgspec.field(default_factory=list)
    moore: list[str] = msgspec.field(default_factory=list)
    source: str = ""

    def to_json(self) -> str:
        return msgspec.json.encode(self).decode()

    @classmethod
    def from_json(cls, data: bytes | str) -> ParallelText:
        return msgspec.json.decode(data, type=cls)


class AlignedCorpus(ParallelText):
    """Aligned parallel corpus where every list has the same length.

    Inherits ``french``, ``moore``, ``source`` from :class:`ParallelText`
    and adds a ``scores`` list (LASER cosine similarity per pair).
    ``__post_init__`` enforces the length invariant.
    """

    scores: list[float] = msgspec.field(default_factory=list)

    def __post_init__(self) -> None:
        n_fr, n_mo, n_sc = len(self.french), len(self.moore), len(self.scores)
        if not (n_fr == n_mo == n_sc):
            raise ValueError(
                f"AlignedCorpus requires equal-length lists, got french={n_fr}, moore={n_mo}, scores={n_sc}"
            )

    @classmethod
    def from_pairs(cls, pairs: list[dict], source: str = "") -> AlignedCorpus:
        """Build from a list of ``{"fr", "mo", "score"}`` dicts."""
        return cls(
            french=[p["fr"] for p in pairs],
            moore=[p["mo"] for p in pairs],
            scores=[p["score"] for p in pairs],
            source=source,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _join_lines(text: str) -> str:
    """Collapse all newlines into single spaces."""
    return _MULTI_SPACE_RE.sub(" ", _NEWLINE_RE.sub(" ", text)).strip()


# ---------------------------------------------------------------------------
# Sentence segmentation
# ---------------------------------------------------------------------------


def segment_fr(text: str) -> list[str]:
    """Segment French text into sentences using syntok."""
    import syntok.segmenter as segmenter

    text = _join_lines(text)
    sentences = []
    for paragraph in segmenter.process(text):
        for sentence in paragraph:
            s = "".join(str(t) for t in sentence).strip()
            if s:
                sentences.append(s)
    return sentences or ([text] if text else [])


def segment_mo(text: str) -> list[str]:
    """Segment Mooré text by punctuation boundaries.

    syntok does not support Mooré, so we split on sentence-ending punctuation
    followed by whitespace.  Newlines are collapsed beforehand.
    """
    text = _join_lines(text)
    parts = [s.strip() for s in _SENT_BOUNDARY_RE.split(text) if s.strip()]
    return parts or ([text] if text else [])


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def normalize_fr(sentence: str) -> str:
    """Normalise French spacing.

    Uses ``sacremoses.MosesTokenizer`` when available; falls back to regex
    rules that fix spaces before punctuation and inside guillemets.
    """
    try:
        from sacremoses import MosesTokenizer

        tok = MosesTokenizer(lang="fr")
        return tok.tokenize(sentence, return_str=True, escape=False)
    except ImportError:
        sentence = re.sub(r" +([!?:;»])", r"\1", sentence)
        sentence = re.sub(r"([«]) +", r"\1", sentence)
        return _MULTI_SPACE_RE.sub(" ", sentence).strip()


def normalize_mo(sentence: str) -> str:
    """Basic Mooré normalisation: collapse consecutive spaces."""
    return _MULTI_SPACE_RE.sub(" ", sentence).strip()


# ---------------------------------------------------------------------------
# Source-specific flatteners
# ---------------------------------------------------------------------------


def flatten_sida_book(
    chapters: list[SidaChapter],
    segment: bool = True,
) -> ParallelText:
    """Flatten SIDA bilingual chapter pages and Chapter-5 enum items.

    Each page contributes its ``french_text`` / ``moore_text``.
    Chapter-5 enum items are included as ``title + body`` units (one per question).

    Args:
        chapters: Output of :func:`moore_web.book_parser.parse_pdf_to_json`.
        segment:  If True, run sentence segmentation on each text block.
    """
    result = ParallelText(source="sida")

    for chapter in chapters:
        for page in chapter.pages:
            fr_raw = _join_lines(page.french_text)
            mo_raw = _join_lines(page.moore_text)
            if not fr_raw and not mo_raw:
                continue
            if segment:
                result.french.extend(normalize_fr(s) for s in segment_fr(fr_raw))
                result.moore.extend(normalize_mo(s) for s in segment_mo(mo_raw))
            else:
                if fr_raw:
                    result.french.append(normalize_fr(fr_raw))
                if mo_raw:
                    result.moore.append(normalize_mo(mo_raw))

        # Chapter 5 enum items: title + body as a single text unit
        for enum in chapter.enums:
            fr = _join_lines(f"{enum.french_title} {enum.french_text}")
            mo = _join_lines(f"{enum.moore_title} {enum.moore_text}")
            if fr and mo:
                result.french.append(normalize_fr(fr))
                result.moore.append(normalize_mo(mo))

    return result


def flatten_facilitateur_pair(
    fr_book: Book,
    mo_book: Book,
    segment: bool = True,
) -> ParallelText:
    """Flatten French and Mooré Kadé books into parallel text.

    Each book is parsed independently (monolingual).  Chapter and section
    titles are included as separate units because they have known bilingual
    counterparts and improve alignment anchoring.  Section content is flattened
    using :func:`moore_web.book_parser_facilitateur.flatten_book_to_list`.

    The two lists will rarely be the same length — the aligner handles that.

    Args:
        fr_book: Parsed French Kadé book.
        mo_book: Parsed Mooré Kadé book.
        segment: If True, run sentence segmentation on each item.
    """
    from moore_web.book_parser_facilitateur import flatten_book_to_list

    result = ParallelText(source="kade")

    # Titles as alignment anchors
    for ch in fr_book.chapters:
        if ch.title.strip():
            result.french.append(normalize_fr(ch.title.strip()))
        for sec in ch.sections:
            if sec.title.strip():
                result.french.append(normalize_fr(sec.title.strip()))

    for ch in mo_book.chapters:
        if ch.title.strip():
            result.moore.append(normalize_mo(ch.title.strip()))
        for sec in ch.sections:
            if sec.title.strip():
                result.moore.append(normalize_mo(sec.title.strip()))

    # Section content
    fr_list = flatten_book_to_list(fr_book)
    mo_list = flatten_book_to_list(mo_book)

    if segment:
        for s in fr_list:
            result.french.extend(normalize_fr(sent) for sent in segment_fr(s))
        for s in mo_list:
            result.moore.extend(normalize_mo(sent) for sent in segment_mo(s))
    else:
        result.french.extend(normalize_fr(s) for s in fr_list if s.strip())
        result.moore.extend(normalize_mo(s) for s in mo_list if s.strip())

    return result


def flatten_news_entries(
    entries: list[dict],
    segment: bool = True,
) -> ParallelText:
    """Flatten segmented news entries into parallel text.

    Each entry must have ``entry["segments"]["french"]`` and
    ``entry["segments"]["moore"]`` (lists of text units, as produced by
    :func:`moore_web.segment_news_data.segment_entries`).

    Args:
        entries: Annotated corpus entries.
        segment: If True, run sentence segmentation on each entry's text.
    """
    result = ParallelText(source="news")

    for item in entries:
        segs = item.get("segments", {})
        fr_text = _join_lines(" ".join(segs.get("french") or []))
        mo_text = _join_lines(" ".join(segs.get("moore") or []))

        if not fr_text and not mo_text:
            continue

        if segment:
            if fr_text:
                result.french.extend(normalize_fr(s) for s in segment_fr(fr_text))
            if mo_text:
                result.moore.extend(normalize_mo(s) for s in segment_mo(mo_text))
        else:
            if fr_text:
                result.french.append(normalize_fr(fr_text))
            if mo_text:
                result.moore.append(normalize_mo(mo_text))

    return result
