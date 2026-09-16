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
_PAGE_REF_RE = re.compile(r"\([^)]*\bp\.?\s*\d+(?:\s*[-–]\s*\d+)?\)", re.IGNORECASE)
_STANDALONE_NUM_RE = re.compile(r"^\s*\d+(?:[-–]\d+)?(?:[.,;:\s]+\d+(?:[-–]\d+)?)*[.,;:]?\s*$")
_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
_COPYRIGHT_RE = re.compile(r"©")


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


# Original (not-translated) language for each known ``source`` tag, used by
# AlignedCorpus.to_jsonl_rows to set ``is_source_orig``. Kadé and sida-bilingual-book
# are both translations of an uncaptured English original (the underlying
# Shellbook Publishing Systems story), so French isn't truly "original" there
# either -- but Mooré was translated FROM the French text, which is what this
# field tracks: translation direction within this corpus, not ultimate
# authorship. conseils/raamde-news are drafted in French then translated to
# Mooré (confirmed: conseils via sig.gov.bf's file naming, raamde-news via
# explicit "Kibarã yii <French source>" attribution lines in ~half the
# articles). niggli-dictionary-mos-fra-eng and digital-postal-glossary
# (-term)(-definition) are lexical: a Mooré headword/term with French/English
# glosses, so Mooré is the "original" side.
ORIGINAL_LANGUAGE: dict[str, str] = {
    "sida-bilingual-book": "fra",
    "kade": "fra",
    "raamde-news": "fra",
    "conseils": "fra",
    "niggli-dictionary-mos-fra-eng": "mos",
    "digital-postal-glossary": "mos",
    "digital-postal-glossary-term": "mos",
    "digital-postal-glossary-term-definition": "mos",
}


def flat_rows_to_long(rows: list[dict], source: str) -> list[dict]:
    """Convert flat ``{french, moore, english?, laser_score?, doc_id?}`` rows
    into the long-format HF schema: one row per (src_lang, tgt_lang) pair.

    A pairwise score (e.g. LASER cosine similarity) belongs to exactly one
    language pair, so a french+moore+english row becomes *two* output rows
    sharing one ``id`` -- a fra/mos-eng row alongside the fra-mos row --
    instead of one row whose single score can't say which pair it's scoring.

    ``src_lang``/``tgt_lang`` follow :data:`ORIGINAL_LANGUAGE`: the original
    (not-translated) side is always ``src_lang``, so ``is_source_orig`` is
    ``True`` for every row of a source with a known direction, and ``None``
    when the source isn't in that mapping (direction not yet determined --
    e.g. a future source where it varies per row would need per-row handling
    here rather than the constant used today).

    ``id`` carries the source document, not just a flat position, when a row
    has a ``doc_id`` (the per-unit key several sources already align by --
    page/enum for sida, article URL for news, session date for conseils --
    but previously discarded after concatenating aligned units together):
    ``"{source}-{nth distinct doc_id}-{nth row within that doc}"``, e.g.
    ``"conseils-000042-003"``. The raw ``doc_id`` value itself (a date, a
    URL, ...) is also kept as its own field for exact grouping/lookup, since
    the ordinal in ``id`` alone doesn't let you filter by it. Rows without a
    ``doc_id`` keep the previous flat ``"{source}-{i:06d}"`` scheme and get
    ``doc_id: None``.
    """
    orig_lang = ORIGINAL_LANGUAGE.get(source)
    is_orig = True if orig_lang is not None else None

    def _pair_row(
        row_id: str, src_lang: str, tgt_lang: str, source_text: str, target_text: str, score, doc_id
    ) -> dict:
        return {
            "id": row_id,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "source_text": source_text,
            "target_text": target_text,
            "is_source_orig": is_orig,
            "doc_id": doc_id,
            "source": source,
            # Always present, even when every row's score is None (e.g.
            # definition pairs, which aren't LASER-scored) -- a key that's
            # sometimes missing and sometimes present across different
            # rows/files/configs can cause a schema mismatch for HF/Arrow
            # consumers (e.g. concatenate_datasets); a null value in an
            # always-present column is the normal, well-handled case.
            "laser_score": round(score, 4) if score is not None else None,
        }

    doc_ordinal: dict[str, int] = {}
    doc_row_count: dict[str, int] = {}

    long_rows: list[dict] = []
    for i, row in enumerate(rows):
        fr, mo, score = row.get("french", ""), row.get("moore", ""), row.get("laser_score")
        doc_id = row.get("doc_id")
        if doc_id is not None:
            if doc_id not in doc_ordinal:
                doc_ordinal[doc_id] = len(doc_ordinal)
            local_idx = doc_row_count.get(doc_id, 0)
            doc_row_count[doc_id] = local_idx + 1
            row_id = f"{source}-{doc_ordinal[doc_id]:06d}-{local_idx:03d}"
        else:
            row_id = f"{source}-{i:06d}"

        if orig_lang == "mos":
            src_lang, tgt_lang, source_text, target_text = "mos", "fra", mo, fr
        else:
            src_lang, tgt_lang, source_text, target_text = "fra", "mos", fr, mo
        long_rows.append(_pair_row(row_id, src_lang, tgt_lang, source_text, target_text, score, doc_id))

        en = row.get("english")
        if en:
            # The english score isn't computed separately today (english only
            # occurs for exact-match dictionary entries, which score 1.0 for
            # the primary pair above); carry the same score value rather than
            # fabricate a distinct one.
            long_rows.append(_pair_row(row_id, src_lang, "eng", source_text, en, score, doc_id))

    return long_rows


class AlignedCorpus(ParallelText):
    """Aligned parallel corpus where every list has the same length.

    Inherits ``french``, ``moore``, ``source`` from :class:`ParallelText`
    and adds a ``scores`` list (LASER cosine similarity per pair).
    ``__post_init__`` enforces the length invariant.
    """

    scores: list[float | None] = msgspec.field(default_factory=list)
    # Per-pair source-document key (page/enum id, article URL, session date,
    # ...) for sources that align per unit and concatenate. Optional -- like
    # `english`, either empty (not tracked) or one entry per pair.
    doc_ids: list[str] = msgspec.field(default_factory=list)

    def __post_init__(self) -> None:
        n_fr, n_mo, n_sc = len(self.french), len(self.moore), len(self.scores)
        if not (n_fr == n_mo == n_sc):
            raise ValueError(
                f"AlignedCorpus requires equal-length lists, got french={n_fr}, moore={n_mo}, scores={n_sc}"
            )
        if self.doc_ids and len(self.doc_ids) != n_fr:
            raise ValueError(f"doc_ids must be empty or match french/moore length, got {len(self.doc_ids)}")

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
        """Return one row per (src_lang, tgt_lang) translation pair, ready to write as JSONL.

        See :func:`flat_rows_to_long` for the schema and rationale.
        """
        flat = [{"french": f, "moore": m, "laser_score": s} for f, m, s in zip(self.french, self.moore, self.scores)]
        if self.english:
            for row, en in zip(flat, self.english):
                row["english"] = en
        if self.doc_ids:
            for row, doc_id in zip(flat, self.doc_ids):
                row["doc_id"] = doc_id
        return flat_rows_to_long(flat, self.source)

    def write_jsonl(self, path: str) -> list[str]:
        """Write aligned pairs to JSONL file(s).

        Split into one file per distinct (src_lang, tgt_lang) pair when more
        than one is present (e.g. a trilingual dictionary's mos-fra rows
        mixed with its mos-eng rows) -- one clean bitext per pair instead of
        one file a consumer has to filter first, matching the convention
        used on the HF-push path (see ``moore_web.annotate.save_data``).
        A single-pair (or empty) corpus is written to ``path`` unchanged.

        Returns the list of file paths written.
        """
        import json
        from pathlib import Path as _Path

        def _write(dest: str, subset: list[dict]) -> None:
            with open(dest, "w", encoding="utf-8") as f:
                for row in subset:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

        rows = self.to_jsonl_rows()
        pairs = sorted({(r["src_lang"], r["tgt_lang"]) for r in rows})

        if len(pairs) <= 1:
            _write(path, rows)
            return [path]

        base = _Path(path)
        written: list[str] = []
        for src_lang, tgt_lang in pairs:
            subset = [r for r in rows if (r["src_lang"], r["tgt_lang"]) == (src_lang, tgt_lang)]
            sub_path = base.with_name(f"{base.stem}.{src_lang}-{tgt_lang}{base.suffix}")
            _write(str(sub_path), subset)
            written.append(str(sub_path))
        return written


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

    closing_chars = {'"', "»", "\u201d"}

    def _starts_new_sentence(text: str) -> bool:
        """First cased letter in `text` is uppercase -> a new sentence, not a
        continuation. Distinguishes a second dialogue turn ('"Foo?" "Bar."')
        from a trailing attribution clause ('"Foo!", she said.'), which starts
        lowercase and must stay merged into the quote it follows."""
        for ch in text:
            if ch.isalpha():
                return ch.isupper()
        return False

    for s in sentences:
        stripped = s.strip()
        is_lone_closing = stripped in closing_chars

        if (
            result
            and in_quote
            and not is_lone_closing
            and stripped
            and stripped[0] in closing_chars
            and _starts_new_sentence(stripped[1:])
        ):
            # `s` opens with the closing partner of the still-open quote,
            # immediately followed by a new, self-contained sentence -- a
            # back-to-back dialogue turn ("...first turn." "Second turn..."),
            # not a continuation. Close the open quote with just that leading
            # character instead of swallowing the whole fragment into it.
            result[-1] += " " + stripped[0]
            remainder = stripped[1:].strip()
            in_quote = False
            if remainder:
                result.append(remainder)
                in_quote = _is_open(remainder)
            continue

        if result and (in_quote or is_lone_closing):
            result[-1] += " " + s
        else:
            result.append(s)
        in_quote = _is_open(result[-1])

    return result


def _syntok_sentences(text: str) -> tuple[str, list[str]]:
    """Tokenize text into sentences with syntok. Returns (joined_text, sentences)."""
    import syntok.segmenter as segmenter

    joined = _join_lines(text)
    sentences = []
    for paragraph in segmenter.process(joined):
        for sentence in paragraph:
            s = "".join(str(t) for t in sentence).strip()
            if s:
                sentences.append(s)
    return joined, sentences


def segment_fr(text: str) -> list[str]:
    """Segment French text into sentences using syntok."""
    joined, sentences = _syntok_sentences(text)
    return _merge_open_quotes(sentences) or ([joined] if joined else [])


def segment_mo(text: str) -> list[str]:
    """Segment Mooré text by punctuation boundaries.

    syntok does not support Mooré, so we only use it for its punctuation
    tokenizer. We skip `_merge_open_quotes`: this source's Mooré dialogue
    doesn't reliably pair quote marks the way French does (an opening `"`
    often has no matching close), so the French quote-balance merge collapses
    whole multi-sentence dialogue passages into one — e.g. an 8-sentence
    passage on page 7 of the SIDA book merges down to 2 with that heuristic.
    """
    joined, sentences = _syntok_sentences(text)
    return sentences or ([joined] if joined else [])


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

    Chapter 5's enum-question pages (from ``enum.start_page`` onward) are
    skipped in the page loop below — their content is already covered by
    ``chapter.enums``, so including both would duplicate every sentence in
    that page range.

    Args:
        chapters: Output of :func:`moore_web.book_parser.parse_pdf_to_json`.
        segment:  If True, run sentence segmentation on each text block.
    """
    result = ParallelText(source="sida-bilingual-book")
    # FIXME: normalization add extra spaces.@critical

    for chapter in chapters:
        enum_start_page = min((e.start_page for e in chapter.enums), default=None)
        for page in chapter.pages:
            if enum_start_page is not None and page.page_number >= enum_start_page:
                continue
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

        # Chapter 5 enum items: title as its own unit, body segmented into sentences
        for enum in chapter.enums:
            fr_title = normalize_fr(_join_lines(enum.french_title))
            mo_title = normalize_mo(_join_lines(enum.moore_title))
            if fr_title:
                result.french.append(fr_title)
            if mo_title:
                result.moore.append(mo_title)

            fr_body = _join_lines(enum.french_text)
            mo_body = _join_lines(enum.moore_text)
            if segment:
                result.french.extend(normalize_fr(s) for s in segment_fr(fr_body) if s)
                result.moore.extend(normalize_mo(s) for s in segment_mo(mo_body) if s)
            else:
                if fr_body:
                    result.french.append(normalize_fr(fr_body))
                if mo_body:
                    result.moore.append(normalize_mo(mo_body))

    return result


def flatten_sida_book_per_unit(
    chapters: list[SidaChapter],
    segment: bool = True,
) -> list[tuple[str, ParallelText]]:
    """Flatten the SIDA book into one ParallelText per page / enum item.

    The PDF is laid out as strict left/right columns (Mooré / French) on
    every content page, so a page's French and Mooré text are already known
    to correspond — unlike a whole-book flatten, alignment doesn't need to
    guess correspondence across page boundaries. Returns a list of
    ``(unit_id, ParallelText)`` pairs so alignment can run independently per
    page (and per Chapter-5 enum item), the same pattern used for per-article
    news alignment and per-date conseils alignment.

    Args:
        chapters: Output of :func:`moore_web.book_parser.parse_pdf_to_json`.
        segment:  If True, run sentence segmentation on each text block.
    """
    results: list[tuple[str, ParallelText]] = []

    for chapter in chapters:
        enum_start_page = min((e.start_page for e in chapter.enums), default=None)
        for page in chapter.pages:
            if enum_start_page is not None and page.page_number >= enum_start_page:
                continue
            fr_raw = _join_lines(page.french_text)
            mo_raw = _join_lines(page.moore_text)
            if not fr_raw or not mo_raw:
                continue

            parallel = ParallelText(source="sida-bilingual-book")
            if segment:
                parallel.french.extend(normalize_fr(s) for s in segment_fr(fr_raw))
                parallel.moore.extend(normalize_mo(s) for s in segment_mo(mo_raw))
            else:
                parallel.french.append(normalize_fr(fr_raw))
                parallel.moore.append(normalize_mo(mo_raw))

            if parallel.french and parallel.moore:
                results.append((f"page-{page.page_number}", parallel))

        for enum in chapter.enums:
            parallel = ParallelText(source="sida-bilingual-book")

            fr_title = normalize_fr(_join_lines(enum.french_title))
            mo_title = normalize_mo(_join_lines(enum.moore_title))
            if fr_title:
                parallel.french.append(fr_title)
            if mo_title:
                parallel.moore.append(mo_title)

            fr_body = _join_lines(enum.french_text)
            mo_body = _join_lines(enum.moore_text)
            if segment:
                parallel.french.extend(normalize_fr(s) for s in segment_fr(fr_body) if s)
                parallel.moore.extend(normalize_mo(s) for s in segment_mo(mo_body) if s)
            else:
                if fr_body:
                    parallel.french.append(normalize_fr(fr_body))
                if mo_body:
                    parallel.moore.append(normalize_mo(mo_body))

            if parallel.french and parallel.moore:
                results.append((f"enum-{enum.enum_number}", parallel))

    return results


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
    from moore_web.book_parser_facilitateur import (
        atomic_item_texts,
        flatten_book_to_list,
        replace_facilitateur_names_fr,
    )

    result = ParallelText(source="kade")

    def _clean_title_fr(t: str) -> str:
        return normalize_fr(replace_facilitateur_names_fr(_PAGE_REF_RE.sub("", t).strip()))

    def _clean_title_mo(t: str) -> str:
        return normalize_mo(_PAGE_REF_RE.sub("", t).strip())

    # Titles as alignment anchors
    for ch in fr_book.chapters:
        if ch.title.strip():
            result.french.append(_clean_title_fr(ch.title))
        for sec in ch.sections:
            if sec.title.strip():
                result.french.append(_clean_title_fr(sec.title))
            for sub in sec.subsections:
                if sub.title.strip():
                    result.french.append(_clean_title_fr(sub.title))

    for ch in mo_book.chapters:
        if ch.title.strip():
            result.moore.append(_clean_title_mo(ch.title))
        for sec in ch.sections:
            if sec.title.strip():
                result.moore.append(_clean_title_mo(sec.title))

    # Section content
    fr_list = flatten_book_to_list(fr_book)
    mo_list = flatten_book_to_list(mo_book)
    mo_atomic = atomic_item_texts(mo_book)

    def _keep(s: str) -> bool:
        return (
            bool(s.strip())
            and not _STANDALONE_NUM_RE.match(s)
            and not _URL_RE.search(s)
            and not _COPYRIGHT_RE.search(s)
            and any(ch.isalpha() for ch in s)
        )

    if segment:
        for s in fr_list:
            result.french.extend(
                normalize_fr(sent)
                for sent in segment_fr(replace_facilitateur_names_fr(_PAGE_REF_RE.sub("", s)))
                if _keep(sent)
            )
        for s in mo_list:
            cleaned = _PAGE_REF_RE.sub("", s)
            if s in mo_atomic:
                # A "Zãmsog a N soaba" (Lesson N) scripture-reference entry:
                # its internal period isn't a real sentence boundary, so keep
                # it as one corpus line instead of sentence-splitting it.
                text = normalize_mo(cleaned)
                if _keep(text):
                    result.moore.append(text)
            else:
                result.moore.extend(normalize_mo(sent) for sent in segment_mo(cleaned) if _keep(sent))
    else:
        result.french.extend(
            normalize_fr(replace_facilitateur_names_fr(_PAGE_REF_RE.sub("", s))) for s in fr_list if _keep(s)
        )
        result.moore.extend(normalize_mo(_PAGE_REF_RE.sub("", s)) for s in mo_list if _keep(s))

    return result


def flatten_simple_parser(
    entries: list,
    include_examples: bool = True,
    include_entries: bool = False,
) -> ParallelText:
    """Flatten output of :func:`moore_web.one_column_dict_parser.parse_doc` into parallel text.

    Args:
        entries:          Output of ``parse_doc`` — list of :class:`DictionaryEntry`.
        include_examples: Add pre-aligned example triplets. All three languages must
                          be present for a triplet to be included.
        include_entries:  Add definition pairs: mooré headword + French + English.
    """
    from moore_web.models import DictionaryEntry

    result = ParallelText(source="niggli-dictionary-mos-fra-eng")

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
    result = ParallelText(source="raamde-news")

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
