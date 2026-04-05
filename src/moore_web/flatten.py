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
_NUMBER_ONLY_RE = re.compile(r"^\d+\.+$")
_MISSING_SPACE_RE = re.compile(r"(?<=[.!?])(?=[A-ZÀ-Ö][a-zà-öø-ÿ])")


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
        """Build from a list of ``{"fr", "mo", "laser_score"}`` dicts."""
        return cls(
            french=[p["fr"] for p in pairs],
            moore=[p["mo"] for p in pairs],
            scores=[p["laser_score"] for p in pairs],
            source=source,
        )

    def to_jsonl_rows(self) -> list[dict]:
        """Return one dict per aligned pair, ready to write as JSONL."""
        has_english = bool(self.english)
        if has_english:
            return [
                {
                    "french": french,
                    "moore": moore,
                    "english": english,
                    "laser_score": round(score, 4),
                    "source": self.source,
                }
                for i, (french, moore, english, score) in enumerate(
                    zip(self.french, self.moore, self.english, self.scores)
                )
            ]
        return [
            {"french": french, "moore": moore, "laser_score": round(score, 4), "source": self.source}
            for french, moore, score in zip(self.french, self.moore, self.scores)
        ]

    def write_jsonl(self, path: str) -> None:
        """Write aligned pairs to a JSONL file."""
        import json

        with open(path, "w", encoding="utf-8") as f:
            for row in self.to_jsonl_rows():
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _join_lines(text: str) -> str:
    """Collapse all newlines into single spaces and fix missing spaces after sentence-ending punctuation."""
    text = _MULTI_SPACE_RE.sub(" ", _NEWLINE_RE.sub(" ", text)).strip()
    return _MISSING_SPACE_RE.sub(" ", text)


# ---------------------------------------------------------------------------
# Sentence segmentation
# ---------------------------------------------------------------------------


def _merge_open_quotes(sentences: list[str]) -> list[str]:
    """Merge syntok fragments produced by splitting inside quoted speech.

    Tracks both typographic double-quote (`"`) balance and guillemet (`«»`)
    balance across segments.  If a segment leaves an unclosed quote open, the
    next segment is merged into it until the quote is closed.  A lone closing
    `"` or `»` is also merged back unconditionally.

    Texts in this corpus mix both quote styles for the *same* quoted speech
    (e.g. ``«…sentence one.`` / ``sentence two…»``), so both markers must be
    tracked together.
    """
    result: list[str] = []
    in_quote = False

    def _is_open(s: str) -> bool:
        if s.count("«") > s.count("»"):
            return True
        if s.count('"') % 2 == 1:
            return True
        if s.count("\u201c") > s.count("\u201d"):
            return True
        return False

    for s in sentences:
        is_lone_closing = s.strip() in {'"', "»", "\u201d"}
        if result and (in_quote or is_lone_closing):
            result[-1] += " " + s
        else:
            result.append(s)
        in_quote = _is_open(result[-1])

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
    """Normalise French spacing and quotes without tokenising.

    - Converts curly double-quotes (U+201C/U+201D) to straight ASCII `"`.
    - Fixes erroneous spaces before punctuation and inside guillemets.
    - Collapses runs of spaces.

    Using MosesTokenizer here is wrong because it splits contractions
    (``Qu'`` → ``Qu' ``) and adds spaces before sentence-ending punctuation,
    which corrupts the text for alignment.
    """
    sentence = sentence.replace("\u201c", '"').replace("\u201d", '"')
    sentence = re.sub(r" +([!?:;».,])", r"\1", sentence)
    sentence = re.sub(r"([«]) +", r"\1", sentence)
    return _MULTI_SPACE_RE.sub(" ", sentence).strip()


def normalize_mo(sentence: str) -> str:
    """Basic Mooré normalisation: collapse spaces and curly quotes."""
    sentence = sentence.replace("\u201c", '"').replace("\u201d", '"')
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
    from moore_web.book_parser_facilitateur import flatten_book_to_list, replace_facilitateur_names_fr

    result = ParallelText(source="kade")

    # Titles as alignment anchors
    for ch in fr_book.chapters:
        if ch.title.strip():
            result.french.append(normalize_fr(replace_facilitateur_names_fr(ch.title.strip())))
        for sec in ch.sections:
            if sec.title.strip():
                result.french.append(normalize_fr(replace_facilitateur_names_fr(sec.title.strip())))
            for sub in sec.subsections:
                if sub.title.strip():
                    result.french.append(normalize_fr(replace_facilitateur_names_fr(sub.title.strip())))

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
            result.french.extend(normalize_fr(sent) for sent in segment_fr(replace_facilitateur_names_fr(s)))
        for s in mo_list:
            result.moore.extend(normalize_mo(sent) for sent in segment_mo(s))
    else:
        result.french.extend(normalize_fr(replace_facilitateur_names_fr(s)) for s in fr_list if s.strip())
        result.moore.extend(normalize_mo(s) for s in mo_list if s.strip())

    return result


def flatten_simple_parser(
    entries: list,
    include_examples: bool = True,
    include_entries: bool = False,
) -> ParallelText:
    """Flatten output of :func:`moore_web.simple_parser.parse_doc` into parallel text.

    Args:
        entries:          Output of ``parse_doc`` — list of :class:`DictionaryEntry`.
        include_examples: Add pre-aligned example triplets. All three languages must
                          be present for a triplet to be included.
        include_entries:  Add definition pairs: mooré headword + French + English.
    """
    from moore_web.models import DictionaryEntry

    result = ParallelText(source="simple")

    def _clean(text: str | None) -> str:
        text = text or ""
        text = re.sub(r"\n+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    for entry in entries:
        if not isinstance(entry, DictionaryEntry):
            continue
        moore_headword = _clean(entry.lemma)

        for sense in entry.senses:
            if include_entries:
                fr = _clean(sense.french)
                en = _clean(sense.english)
                if fr or en or moore_headword:
                    result.french.append(normalize_fr(fr) if fr else "")
                    result.moore.append(normalize_mo(moore_headword) if moore_headword else "")
                    result.english.append(en)

            if include_examples:
                for example in sense.examples:
                    mo_ex = _clean(example.moore)
                    fr_ex = _clean(example.french)
                    en_ex = _clean(example.english)
                    if mo_ex and fr_ex and en_ex:
                        result.moore.append(mo_ex)
                        result.french.append(fr_ex)
                        result.english.append(en_ex)

    return result


def flatten_conseils(
    corpus: list[dict],
    segment: bool = True,
) -> list[tuple[str, "ParallelText"]]:
    """Flatten conseil-des-ministres corpus into per-date ParallelText lists.

    Each entry in *corpus* represents one council session.  Only entries that
    have non-empty ``src_sections`` **and** ``tgt_sections`` are included.
    ``src_lang`` / ``tgt_lang`` identify which side is French vs Mooré.

    Each section has a ``title`` string and a ``sentences`` list (already
    sentence-split by the parser).  Titles are optionally re-segmented;
    individual sentences are added directly.

    Args:
        corpus:  List of session dicts with ``date``, ``src_lang``,
                 ``tgt_lang``, ``src_sections``, ``tgt_sections`` keys.
                 Each section dict has ``number``, ``title``, ``sentences``,
                 and ``subsections`` fields.
        segment: If True, run sentence segmentation on section titles.

    Returns:
        List of ``(date, ParallelText)`` pairs, one per bilingual session.
    """
    results: list[tuple[str, ParallelText]] = []

    for entry in corpus:
        date = entry.get("date", "")
        src_lang = entry.get("src_lang", "fr")
        src_sections = entry.get("src_sections") or []
        tgt_sections = entry.get("tgt_sections") or []

        if not src_sections or not tgt_sections:
            continue

        # Map src/tgt to french/moore based on declared language codes
        if src_lang == "fr":
            fr_sections, mo_sections = src_sections, tgt_sections
        else:
            mo_sections, fr_sections = src_sections, tgt_sections

        parallel = ParallelText(source=f"conseils/{date}")

        def _add_section(sections: list[dict], target: list[str], normalize_fn, segment_fn) -> None:
            for sec in sections:
                title = _join_lines(sec.get("title", ""))
                if title and not _NUMBER_ONLY_RE.match(title):
                    if segment:
                        target.extend(normalize_fn(s) for s in segment_fn(title))
                    else:
                        target.append(normalize_fn(title))
                for sent in sec.get("sentences") or []:
                    s = _join_lines(sent)
                    if s and not _NUMBER_ONLY_RE.match(s):
                        target.append(normalize_fn(s))

        _add_section(fr_sections, parallel.french, normalize_fr, segment_fr)
        _add_section(mo_sections, parallel.moore, normalize_mo, segment_mo)

        if parallel.french and parallel.moore:
            results.append((date, parallel))

    return results


def flatten_news_entries(
    entries: list[dict],
    segment: bool = True,
) -> ParallelText:
    """Flatten segmented news entries into a single parallel text.

    Each entry must have ``entry["segments"]["french"]`` and
    ``entry["segments"]["moore"]`` (lists of text units, as produced by
    :func:`moore_web.segment_news_data.segment_entries`).

    .. warning::
        All articles are merged into one flat list.  Use
        :func:`flatten_news_per_entry` when aligning, so that FastDTW
        operates within article boundaries rather than across them.

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


def flatten_news_per_entry(
    entries: list[dict],
    segment: bool = True,
) -> list[tuple[str, ParallelText]]:
    """Flatten segmented news entries into one ParallelText per article.

    Returns a list of ``(url, ParallelText)`` pairs so that alignment can be
    run independently per article.  This preserves the monotonic ordering
    assumption required by FastDTW and prevents sentences from different
    articles being aligned to each other.

    Entries without both French and Mooré content are skipped.

    Args:
        entries: Annotated corpus entries (output of
                 :func:`moore_web.segment_news_data.segment_entries`).
        segment: If True, run sentence segmentation on each entry's text.
    """
    results: list[tuple[str, ParallelText]] = []

    for i, item in enumerate(entries):
        segs = item.get("segments", {})
        fr_text = _join_lines(" ".join(segs.get("french") or []))
        mo_text = _join_lines(" ".join(segs.get("moore") or []))

        if not fr_text or not mo_text:
            continue

        parallel = ParallelText(source=f"news/{i}")

        if segment:
            parallel.french.extend(normalize_fr(s) for s in segment_fr(fr_text))
            parallel.moore.extend(normalize_mo(s) for s in segment_mo(mo_text))
        else:
            parallel.french.append(normalize_fr(fr_text))
            parallel.moore.append(normalize_mo(mo_text))

        if parallel.french and parallel.moore:
            url = item.get("url", str(i))
            results.append((url, parallel))

    return results
