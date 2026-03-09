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
    english: list[str] = msgspec.field(default_factory=list)
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


def _merge_open_quotes(sentences: list[str]) -> list[str]:
    """Merge syntok fragments produced by splitting inside quoted speech.

    Tracks `"` balance across segments: if a segment leaves an unclosed quote,
    the next segment is merged into it until the quote is closed.
    A lone closing `"` is also merged back unconditionally.
    """
    result: list[str] = []
    in_quote = False

    for s in sentences:
        is_lone_quote = s.strip() == '"'
        if result and (in_quote or is_lone_quote):
            result[-1] += " " + s
        else:
            result.append(s)
        in_quote = result[-1].count('"') % 2 == 1

    return result


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
    return _merge_open_quotes(sentences) or ([text] if text else [])


def segment_mo(text: str) -> list[str]:
    """Segment Mooré text by punctuation boundaries.

    syntok does not support Mooré, so we split on sentence-ending punctuation
    followed by whitespace.  Newlines are collapsed beforehand.
    """
    return segment_fr(text=text)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def normalize_fr(sentence: str) -> str:
    """Normalise French spacing without tokenising.

    Only fixes erroneous spaces before punctuation and inside guillemets,
    then collapses runs of spaces.  Using MosesTokenizer here is wrong because
    it splits contractions (``Qu'`` → ``Qu' ``) and adds spaces before
    sentence-ending punctuation, which corrupts the text for alignment.
    """
    sentence = re.sub(r" +([!?:;».,])", r"\1", sentence)
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
    # FIXME: normalization add extra spaces.@critical

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
        # TODO: maybe segment?
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


def flatten_simple_parser(
    pages: list[list[dict]],
    include_examples: bool = True,
    include_entries: bool = False,
) -> ParallelText:
    """Flatten output of :func:`moore_web.simple_parser.parse_doc` into parallel text.

    Each dictionary entry has:
    - ``entry``      — Mooré headword
    - ``senses``     — list of sub-entry blocks, each a list of sense dicts with
                       ``fr_entry``, ``eng_entry``, ``moore_example``,
                       ``french_example``, ``english_example``

    Args:
        pages:            Output of ``parse_doc`` — list of pages, each a list of
                          entry dicts.
        include_examples: Add pre-aligned example triplets
                          (moore_example, french_example, english_example).
                          All three must be non-null for a triplet to be included.
        include_entries:  Add definition pairs: moore headword (``entry``),
                          French definition (``fr_entry``), English definition
                          (``eng_entry``).
    """
    result = ParallelText(source="simple")

    def _clean(text: str | None) -> str:
        text = text or ""
        text = re.sub(r"\n+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    for page in pages:
        for entry_dict in page:
            moore_headword = _clean(entry_dict.get("entry"))

            for sub_entry in entry_dict.get("senses") or []:
                for sense in sub_entry:
                    if include_entries:
                        fr = _clean(sense.get("fr_entry"))
                        en = _clean(sense.get("eng_entry"))
                        if fr or en or moore_headword:
                            result.french.append(normalize_fr(fr) if fr else "")
                            result.moore.append(normalize_mo(moore_headword) if moore_headword else "")
                            result.english.append(en)

                    if include_examples:
                        mo_ex = _clean(sense.get("moore_example"))
                        fr_ex = _clean(sense.get("french_example"))
                        en_ex = _clean(sense.get("english_example"))
                        if mo_ex and fr_ex and en_ex:
                            result.moore.append(mo_ex if mo_ex else mo_ex)
                            result.french.append(fr_ex)
                            result.english.append(en_ex)

    return result


def flatten_conseils(
    corpus: list[dict],
    segment: bool = True,
) -> list[tuple[str, "ParallelText"]]:
    """Flatten conseil-des-ministres corpus into per-date ParallelText lists.

    Each entry in *corpus* represents one council session.  Only entries that
    have non-empty ``fr`` **and** ``mos`` lists are included.  French and Mooré
    section texts are collected independently; the aligner handles many-to-many
    alignment within each date.

    Args:
        corpus:  List of session dicts with ``date``, ``fr``, ``mos`` keys.
                 Each language value is a list of ``{section, subsection,
                 index, text}`` dicts (output of the conseil-ministres parser).
        segment: If True, run sentence segmentation on each section text.

    Returns:
        List of ``(date, ParallelText)`` pairs, one per bilingual session.
    """
    results: list[tuple[str, ParallelText]] = []

    for entry in corpus:
        date = entry.get("date", "")
        fr_sections = entry.get("fr") or []
        mo_sections = entry.get("mos") or []

        if not fr_sections or not mo_sections:
            continue

        parallel = ParallelText(source=f"conseils/{date}")

        for sec in fr_sections:
            text = _join_lines(sec.get("text", ""))
            if not text:
                continue
            if segment:
                parallel.french.extend(normalize_fr(s) for s in segment_fr(text))
            else:
                parallel.french.append(normalize_fr(text))

        for sec in mo_sections:
            text = _join_lines(sec.get("text", ""))
            if not text:
                continue
            if segment:
                parallel.moore.extend(normalize_mo(s) for s in segment_mo(text))
            else:
                parallel.moore.append(normalize_mo(text))

        if parallel.french and parallel.moore:
            results.append((date, parallel))

    return results


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
