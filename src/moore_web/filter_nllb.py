"""Multi-dimensional filtering of the NLLB eng↔mos dataset.

Filters applied
---------------
1. ``target_sentence_lid >= threshold``  — NLLB built-in LID confidence
2. ``target_glotlid_prob >= threshold``  — GlotLID confidence on Mooré target
3. ``source_glotlid_prob >= threshold``  — GlotLID confidence on English source
4. ``comet_qe >= threshold``             — COMET-QE translation quality
5. Emoji filter                          — drop rows where Mooré text contains emoji
6. Dots asymmetry                        — ".." at start/end in one side but not both
7. Bullet/special char asymmetry         — leading bullet chars (●, •, ^) in one side only
8. Parenthesis asymmetry                 — parenthetical content in English absent in Mooré
9. Number mismatch                       — digit sequences present on one side but not the other
10. Foreign word list                    — Mooré contains words from non-Mooré GlotLID wordlists

Quality warnings added per row (before hard filtering)
-------------------------------------------------------
``has_emoji``, ``has_dots_asymmetry``, ``has_number_mismatch``,
``has_parenthesis_asymmetry``, ``has_bullet_asymmetry``,
``has_foreign_words``, ``identification_inconsistency``

Usage
-----
    # Annotate warnings only (no hard filter, no push)
    uv run python -m moore_web.filter_nllb --source-repo madoss/nllb-mos-lid --no-push

    # Filter and push to Hub
    uv run python -m moore_web.filter_nllb --source-repo madoss/nllb-mos-lid --hub-repo madoss/nllb-mos-filtered

    # Custom thresholds
    uv run python -m moore_web.filter_nllb \\
        --source-repo madoss/nllb-mos \\
        --hub-repo madoss/nllb-mos-filtered \\
        --lid-threshold 0.9 \\
        --glotlid-threshold 0.9 \\
        --comet-threshold 0.5
"""

# TODO; Improve filtering with https://huggingface.co/datasets/cis-lmu/glotlid-wordlists/blob/main/filter.py
from __future__ import annotations

import argparse
import re

from dotenv import load_dotenv

try:
    from datatrove.utils.text import TextNormConfig, simplify_text as _simplify_text

    _NORM_CONFIG = TextNormConfig(
        lowercase=False,
        norm_numbers=False,
        norm_weekdays=False,
        norm_monthnames=False,
        remove_punctuation=True,
        norm_unicode_diacritics=False,
        norm_whitespace=True,
    )
    _HAS_DATATROVE = True
except ImportError:
    _HAS_DATATROVE = False

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Unicode emoji ranges (covers most common emoji blocks)
_EMOJI_RE = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f680-\U0001f6ff"  # transport & map
    "\U0001f1e0-\U0001f1ff"  # flags
    "\U00002702-\U000027b0"
    "\U000024c2-\U0001f251"
    "]+",
    flags=re.UNICODE,
)

# Consecutive dots that act as ellipsis / page-reference markers
_DOTS_RE = re.compile(r"\.{2,}")

# Numbers (digit sequences, possibly separated by . or ,)
_NUMBERS_RE = re.compile(r"\b\d[\d.,]*\b")

# Parenthetical content: (…) or [...]
_PARENS_RE = re.compile(r"[\(\[][^\)\]]{2,}[\)\]]")

# Bullet / list markers that may legitimately start a line
_BULLET_CHARS = frozenset("●•◦▪▸►‣⁃◆◇→−–—^*")

# GlotLID language codes we expect for this dataset
_EXPECTED_SOURCE_LANG = "eng_Latn"
_EXPECTED_TARGET_LANG = "mos_Latn"

# Column names used by upstream scripts (keep in sync with glotlid.py / score_nllb_mos.py)
_COL_ENG = "eng_Latn"
_COL_MOS = "mos_Latn"
_COL_TARGET_LID = "target_sentence_lid"
_COL_SOURCE_LID = "source_sentence_lid"
_COL_TARGET_GLOTLID_LANG = "target_glotlid_lang"
_COL_TARGET_GLOTLID_PROB = "target_glotlid_prob"
_COL_SOURCE_GLOTLID_LANG = "source_glotlid_lang"
_COL_SOURCE_GLOTLID_PROB = "source_glotlid_prob"
_COL_COMET_QE = "comet_qe_en_mos"

# ---------------------------------------------------------------------------
# Warning detectors (pure functions, operate on a single row dict)
# ---------------------------------------------------------------------------


def _has_emoji(text: str) -> bool:
    return bool(_EMOJI_RE.search(text))


def _has_dots_asymmetry(src: str, tgt: str) -> bool:
    """True when '..'-style dots appear at start/end of one side but not the other."""
    src_stripped = src.strip()
    tgt_stripped = tgt.strip()
    src_has = bool(_DOTS_RE.match(src_stripped) or (src_stripped and _DOTS_RE.search(src_stripped[-5:])))
    tgt_has = bool(_DOTS_RE.match(tgt_stripped) or (tgt_stripped and _DOTS_RE.search(tgt_stripped[-5:])))
    return src_has != tgt_has


def _has_number_mismatch(src: str, tgt: str) -> bool:
    """True when digit sequences found in source are missing from target (or vice-versa)."""
    src_nums = set(_NUMBERS_RE.findall(src))
    tgt_nums = set(_NUMBERS_RE.findall(tgt))
    # Flag only when one side has numbers and the other has none, or sets are disjoint and both non-empty
    if not src_nums and not tgt_nums:
        return False
    if src_nums and not tgt_nums:
        return True
    if not src_nums and tgt_nums:
        return True
    # Both have numbers — flag when the intersection is empty (completely different numbers)
    return src_nums.isdisjoint(tgt_nums)


def _has_parenthesis_asymmetry(src: str, tgt: str) -> bool:
    """True when source (English) has parenthetical content that target lacks."""
    src_parens = _PARENS_RE.findall(src)
    tgt_parens = _PARENS_RE.findall(tgt)
    return bool(src_parens) and not bool(tgt_parens)


def _has_bullet_asymmetry(src: str, tgt: str) -> bool:
    """True when a bullet/special char leads one side but not the other."""
    src_first = src.strip()[:1]
    tgt_first = tgt.strip()[:1]
    src_bullet = src_first in _BULLET_CHARS
    tgt_bullet = tgt_first in _BULLET_CHARS
    return src_bullet != tgt_bullet


def _lang_consistency_score(text: str, foreign_words: set[str]) -> float:
    """Fraction of tokens in *text* that do NOT appear in the foreign wordlist.

    Uses datatrove's ``simplify_text`` when available for punctuation stripping,
    then falls back to regex tokenisation.

    A high score means the text is mostly non-foreign (consistent with Mooré).
    A low score means many tokens are French/English words.

    score = len(words - foreign_words) / len(words) if words else 0
    """
    if _HAS_DATATROVE:
        cleaned = _simplify_text(str(text).strip(), _NORM_CONFIG)
    else:
        cleaned = text
    # Exclude tokens shorter than 3 characters: single letters, abbreviations and
    # random noise are absent from any wordlist and would inflate the score.
    words = {w for w in re.findall(r"\b\w+\b", cleaned.lower()) if len(w) >= 3}
    # If too few meaningful tokens remain the score is unreliable (noise / gibberish).
    if len(words) < 2:
        return 0.0
    non_foreign = len(words - foreign_words)
    return round(non_foreign / len(words), 4)


# ---------------------------------------------------------------------------
# Word-list filter
# ---------------------------------------------------------------------------


def _load_glotlid_wordlists(languages: list[str]) -> set[str]:
    """Load word lists from ``madoss/mos-eng-fra-wordlists`` for the given language configs.

    Each language is a separate dataset config (e.g. ``"fra_Latn"``).
    Returns a flat union of all words across the requested languages.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Warning: 'datasets' not installed — skipping wordlist filter.")
        return set()

    words: set[str] = set()
    for lang in languages:
        try:
            ds = load_dataset("madoss/mos-eng-fra-wordlists", split=lang)
            for row in ds:
                word = row.get("text") or row.get("word", "")
                if word:
                    words.add(word.lower())
            print(f"  Loaded {lang} wordlist ({len(words):,} words total so far).")
        except Exception as e:
            print(f"Warning: could not load glotlid-wordlists[{lang}] ({e}) — skipping.")
    return words


# GlotLID language code → pyspellchecker language code
_SPELLCHECKER_LANG_MAP = {"fra_Latn": "fr", "eng_Latn": "en", "spa_Latn": "es", "deu_Latn": "de"}


def _load_spellchecker_words(languages: list[str]) -> set[str]:
    """Load full vocabulary from pyspellchecker for the given GlotLID language codes.

    pyspellchecker ships with comprehensive word-frequency dictionaries (~100k+ words),
    unlike the GlotLID discriminative lists (~14–25k n-grams).  This catches common
    words like 'comment', 'game', 'pesée' that GlotLID lists omit.
    """
    try:
        from spellchecker import SpellChecker
    except ImportError:
        print("Warning: 'pyspellchecker' not installed — falling back to GlotLID wordlists only.")
        return set()

    words: set[str] = set()
    for lang in languages:
        sc_lang = _SPELLCHECKER_LANG_MAP.get(lang)
        if sc_lang is None:
            continue
        try:
            sc = SpellChecker(language=sc_lang)
            words.update(sc.word_frequency.keys())
            print(f"  Loaded spellchecker[{sc_lang}] ({len(words):,} words total so far).")
        except Exception as e:
            print(f"Warning: could not load spellchecker[{sc_lang}] ({e}) — skipping.")
    return words


def _has_foreign_words(text: str, wordlist: set[str]) -> bool:
    """True when *text* contains at least one token (length >= 3) present in the foreign wordlist."""
    if not wordlist:
        return False
    tokens = (t for t in re.findall(r"\b\w+\b", text.lower()) if len(t) >= 3)
    return any(t in wordlist for t in tokens)


# ---------------------------------------------------------------------------
# Annotation: add warning columns to a batch
# ---------------------------------------------------------------------------


def annotate_warnings(
    batch: dict[str, list],
    foreign_wordlist: set[str],
) -> dict[str, list]:
    """Add ``quality_warnings`` and ``identification_consistency`` columns.

    ``quality_warnings`` is a ``list[str]`` of active warning labels per row:
      ``"emoji"``, ``"dots_asymmetry"``, ``"number_mismatch"``,
      ``"parenthesis_asymmetry"``, ``"bullet_asymmetry"``, ``"foreign_words"``

    ``identification_consistency`` is a float in [0, 1]: fraction of Mooré
    tokens that do NOT appear in the foreign (French/English) word list.
    """
    src_texts = batch[_COL_ENG]
    tgt_texts = batch[_COL_MOS]
    n = len(src_texts)

    quality_warnings = []
    id_consistency = []

    for i in range(n):
        src = src_texts[i] or ""
        tgt = tgt_texts[i] or ""

        warnings: list[str] = []
        if _has_emoji(tgt):
            warnings.append("emoji")
        if _has_dots_asymmetry(src, tgt):
            warnings.append("dots_asymmetry")
        if _has_number_mismatch(src, tgt):
            warnings.append("number_mismatch")
        if _has_parenthesis_asymmetry(src, tgt):
            warnings.append("parenthesis_asymmetry")
        if _has_bullet_asymmetry(src, tgt):
            warnings.append("bullet_asymmetry")
        if _has_foreign_words(tgt, foreign_wordlist):
            warnings.append("foreign_words")

        quality_warnings.append(warnings)
        id_consistency.append(_lang_consistency_score(tgt, foreign_wordlist))

    batch["quality_warnings"] = quality_warnings
    batch["identification_consistency"] = id_consistency

    return batch


# ---------------------------------------------------------------------------
# Hard filtering
# ---------------------------------------------------------------------------


def apply_hard_filters(
    dataset,
    lid_threshold: float = 0.9,
    glotlid_threshold: float = 0.9,
    comet_threshold: float = 0.5,
    filter_emoji: bool = True,
    filter_dots: bool = True,
    filter_foreign_words: bool = True,
    filter_parenthesis: bool = False,
    filter_number_mismatch: bool = False,
    consistency_threshold: float = 0.0,
):
    """Remove rows that fail any hard quality criterion.

    Soft warnings (``has_*`` columns) are left in the dataset for downstream
    inspection regardless of whether hard filtering is applied.

    Args:
        dataset:                  HF Dataset with warning columns already added.
        lid_threshold:            Minimum ``target_sentence_lid`` (NLLB built-in).
        glotlid_threshold:        Minimum ``target_glotlid_prob`` and ``source_glotlid_prob``.
        comet_threshold:          Minimum COMET-QE score.
        filter_emoji:             Drop rows where Mooré text has emoji.
        filter_dots:              Drop rows with dots asymmetry.
        filter_foreign_words:     Drop rows where Mooré contains foreign words.
        filter_parenthesis:       Drop rows with parenthesis asymmetry (off by default).
        filter_number_mismatch:   Drop rows with number mismatch (off by default).
        consistency_threshold:    Minimum ``identification_consistency`` score (fraction of
                                  Mooré tokens found in the Mooré word list).  0.0 disables
                                  this filter.

    Returns:
        Filtered HF Dataset.
    """
    before = len(dataset)
    stats: dict[str, int] = {}

    def _track(ds, name: str, condition):
        nonlocal stats
        after = ds.filter(condition, desc=f"filter:{name}")
        stats[name] = len(ds) - len(after)
        return after

    # --- numeric thresholds ---
    if _COL_TARGET_LID in dataset.column_names:
        dataset = _track(
            dataset,
            "target_lid",
            lambda r: r[_COL_TARGET_LID] is not None and r[_COL_TARGET_LID] >= lid_threshold,
        )

    if _COL_TARGET_GLOTLID_PROB in dataset.column_names:
        dataset = _track(
            dataset,
            "target_glotlid",
            lambda r: (
                r[_COL_TARGET_GLOTLID_PROB] is not None
                and r[_COL_TARGET_GLOTLID_PROB] >= glotlid_threshold
                and r.get(_COL_TARGET_GLOTLID_LANG) == _EXPECTED_TARGET_LANG
            ),
        )

    if _COL_SOURCE_GLOTLID_PROB in dataset.column_names:
        dataset = _track(
            dataset,
            "source_glotlid",
            lambda r: (
                r[_COL_SOURCE_GLOTLID_PROB] is not None
                and r[_COL_SOURCE_GLOTLID_PROB] >= glotlid_threshold
                and r.get(_COL_SOURCE_GLOTLID_LANG) == _EXPECTED_SOURCE_LANG
            ),
        )

    if _COL_COMET_QE in dataset.column_names:
        dataset = _track(
            dataset,
            "comet_qe_en_mos",
            lambda r: r[_COL_COMET_QE] is not None and r[_COL_COMET_QE] >= comet_threshold,
        )

    # --- warning-based hard filters (check quality_warnings list) ---
    qw_col = "quality_warnings"
    if qw_col in dataset.column_names:
        active: list[str] = []
        if filter_emoji:
            active.append("emoji")
        if filter_dots:
            active.append("dots_asymmetry")
        if filter_foreign_words:
            active.append("foreign_words")
        if filter_parenthesis:
            active.append("parenthesis_asymmetry")
        if filter_number_mismatch:
            active.append("number_mismatch")
        if active:
            active_set = frozenset(active)
            dataset = _track(
                dataset,
                "quality_warnings",
                lambda r: not bool(active_set & set(r[qw_col] or [])),
            )

    if consistency_threshold > 0.0 and "identification_consistency" in dataset.column_names:
        dataset = _track(
            dataset,
            "identification_consistency",
            lambda r: (r["identification_consistency"] or 0.0) >= consistency_threshold,
        )

    after = len(dataset)
    print(f"\nFiltering summary ({before:,} → {after:,} rows kept, {before - after:,} dropped):")
    for name, dropped in stats.items():
        print(f"  {name}: dropped {dropped:,}")

    return dataset


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def filter_nllb(
    source_repo: str,
    hub_repo: str | None = None,
    output: str | None = None,
    lid_threshold: float = 0.9,
    glotlid_threshold: float = 0.9,
    comet_threshold: float = 0.5,
    filter_emoji: bool = True,
    filter_dots: bool = True,
    filter_foreign_words: bool = True,
    filter_parenthesis: bool = False,
    filter_number_mismatch: bool = False,
    consistency_threshold: float = 0.0,
    load_wordlists: bool = True,
    batch_size: int = 1000,
    private: bool = False,
) -> None:
    """Full annotation + filtering pipeline for the NLLB eng↔mos dataset.

    Args:
        source_repo:              HF Hub dataset to load (must have GlotLID columns).
        hub_repo:                 HF Hub repo to push filtered results to.  If None,
                                  results are not pushed.
        output:                   Local JSONL path to write filtered results to.
                                  If None, no local file is written.
        lid_threshold:            Minimum ``target_sentence_lid``.
        glotlid_threshold:        Minimum GlotLID probability for both sides.
        comet_threshold:          Minimum COMET-QE score.
        filter_emoji:             Drop rows with emoji in Mooré text.
        filter_dots:              Drop rows with dots asymmetry.
        filter_foreign_words:     Drop rows where Mooré has foreign words.
        filter_parenthesis:       Drop rows with parenthesis asymmetry.
        filter_number_mismatch:   Drop rows with number mismatch.
        consistency_threshold:    Minimum ``identification_consistency`` score. 0.0 disables.
        load_wordlists:           Whether to load GlotLID wordlists.
        batch_size:               Rows per batch for dataset.map.
        private:                  Whether to make the HF Hub dataset private.
    """
    from datasets import load_dataset

    print(f"Loading dataset from '{source_repo}'…")
    ds = load_dataset(source_repo, split="train")
    print(f"Loaded {len(ds):,} rows.")

    # Load wordlists
    foreign_wordlist: set[str] = set()
    if load_wordlists:
        # GlotLID discriminative n-grams (~14–25k, misses common words like 'comment', 'game')
        glotlid_foreign = _load_glotlid_wordlists(["fra_Latn", "eng_Latn"])
        # pyspellchecker full dictionaries (~100k+, covers common vocabulary)
        spell_foreign = _load_spellchecker_words(["fra_Latn", "eng_Latn"])
        raw_foreign = glotlid_foreign | spell_foreign
        moore_wordlist = _load_glotlid_wordlists(["mos_Latn"])
        # NOTE: the GlotLID Mooré wordlist is very small (~1 000 discriminative n-grams),
        # so it cannot be used to positively identify Mooré words.  We use it only to
        # subtract any overlap from the foreign wordlist, avoiding false positives for
        # loanwords or short tokens shared between languages.
        foreign_wordlist = raw_foreign - moore_wordlist
        print(
            f"  Exclusive foreign wordlist: {len(foreign_wordlist):,} words ({len(raw_foreign) - len(foreign_wordlist):,} removed as Mooré overlap)."
        )

    # Annotate quality warnings
    print("Annotating quality warnings…")
    ds = ds.map(
        lambda batch: annotate_warnings(batch, foreign_wordlist),
        batched=True,
        batch_size=batch_size,
        desc="annotate warnings",
        load_from_cache_file=False,
    )

    # Print warning summary before filtering
    n_total = len(ds)
    print("\nWarning counts (before hard filtering):")
    if "quality_warnings" in ds.column_names:
        rows_with_warnings = sum(1 for w in ds["quality_warnings"] if w)
        print(f"  quality_warnings (any): {rows_with_warnings:,} ({100 * rows_with_warnings / n_total:.1f}%)")
        from collections import Counter

        label_counts: Counter = Counter(label for w in ds["quality_warnings"] for label in (w or []))
        for label, cnt in label_counts.most_common():
            print(f"    {label}: {cnt:,} ({100 * cnt / n_total:.1f}%)")
    if "identification_consistency" in ds.column_names:
        scores = ds["identification_consistency"]
        mean_score = sum(scores) / len(scores) if scores else 0.0
        print(f"  identification_consistency (mean): {mean_score:.3f}")

    # Apply hard filters
    ds = apply_hard_filters(
        ds,
        lid_threshold=lid_threshold,
        glotlid_threshold=glotlid_threshold,
        comet_threshold=comet_threshold,
        filter_emoji=filter_emoji,
        filter_dots=filter_dots,
        filter_foreign_words=filter_foreign_words,
        filter_parenthesis=filter_parenthesis,
        filter_number_mismatch=filter_number_mismatch,
        consistency_threshold=consistency_threshold,
    )

    # Write local output
    if output:
        import json
        from pathlib import Path

        print(f"\nWriting {len(ds):,} rows → {output}")
        with Path(output).open("w", encoding="utf-8") as f:
            for row in ds:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print("Done (local).")

    # Push to Hub
    if hub_repo:
        from datasets import DatasetDict

        print(f"\nPushing {len(ds):,} rows → '{hub_repo}'…")
        DatasetDict({"train": ds}).push_to_hub(hub_repo, private=private)
        print(f"Done. https://huggingface.co/datasets/{hub_repo}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multi-dimensional NLLB eng↔mos filtering pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source-repo",
        required=True,
        help="HF Hub dataset to load (should already have GlotLID and COMET-QE columns).",
    )
    parser.add_argument(
        "--hub-repo",
        default=None,
        help="HF Hub repo to push filtered dataset to.  Omit to skip pushing.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Local JSONL path to write filtered rows to.  Omit to skip local write.",
    )
    parser.add_argument(
        "--lid-threshold",
        type=float,
        default=0.9,
        help="Minimum target_sentence_lid (NLLB built-in, default: %(default)s).",
    )
    parser.add_argument(
        "--glotlid-threshold",
        type=float,
        default=0.9,
        help="Minimum GlotLID probability for source and target (default: %(default)s).",
    )
    parser.add_argument(
        "--comet-threshold",
        type=float,
        default=0.5,
        help="Minimum COMET-QE score (default: %(default)s).",
    )
    parser.add_argument(
        "--no-filter-emoji",
        dest="filter_emoji",
        action="store_false",
        help="Do not hard-filter rows with emoji in Mooré.",
    )
    parser.add_argument(
        "--no-filter-dots",
        dest="filter_dots",
        action="store_false",
        help="Do not hard-filter rows with dots asymmetry.",
    )
    parser.add_argument(
        "--no-filter-foreign-words",
        dest="filter_foreign_words",
        action="store_false",
        help="Do not hard-filter rows with foreign words in Mooré.",
    )
    parser.add_argument(
        "--filter-parenthesis",
        dest="filter_parenthesis",
        action="store_true",
        help="Hard-filter rows with parenthesis asymmetry (default: warn only).",
    )
    parser.add_argument(
        "--filter-number-mismatch",
        dest="filter_number_mismatch",
        action="store_true",
        help="Hard-filter rows with number mismatch (default: warn only).",
    )
    parser.add_argument(
        "--consistency-threshold",
        type=float,
        default=0.0,
        help="Minimum identification_consistency score (fraction of Mooré tokens in the Mooré "
        "word list). 0.0 disables this filter (default: %(default)s).",
    )
    parser.add_argument(
        "--no-push",
        dest="push",
        action="store_false",
        help="Skip pushing to HF Hub (useful with --output for local-only runs).",
    )
    parser.add_argument(
        "--no-wordlists",
        dest="load_wordlists",
        action="store_false",
        help="Skip loading GlotLID wordlists (faster, skips foreign-word detection).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Rows per batch for dataset.map (default: %(default)s).",
    )
    parser.add_argument("--private", action="store_true", help="Make the HF Hub dataset private.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    hub_repo = args.hub_repo if args.push else None

    filter_nllb(
        source_repo=args.source_repo,
        hub_repo=hub_repo,
        output=args.output,
        lid_threshold=args.lid_threshold,
        glotlid_threshold=args.glotlid_threshold,
        comet_threshold=args.comet_threshold,
        filter_emoji=args.filter_emoji,
        filter_dots=args.filter_dots,
        filter_foreign_words=args.filter_foreign_words,
        filter_parenthesis=args.filter_parenthesis,
        filter_number_mismatch=args.filter_number_mismatch,
        consistency_threshold=args.consistency_threshold,
        load_wordlists=args.load_wordlists,
        batch_size=args.batch_size,
        private=args.private,
    )


if __name__ == "__main__":
    main()
