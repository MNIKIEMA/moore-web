"""Word-list loading utilities for language-quality filtering.

Provides helpers to fetch GlotLID discriminative n-gram lists and
pyspellchecker full-vocabulary dictionaries, plus a high-level builder
that constructs the foreign-word exclusion set used by quality-warning
checks.
"""

from __future__ import annotations

# GlotLID language code → pyspellchecker language code
_SPELLCHECKER_LANG_MAP = {"fra_Latn": "fr", "eng_Latn": "en", "spa_Latn": "es", "deu_Latn": "de"}


def load_glotlid_wordlists(languages: list[str]) -> set[str]:
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


def load_spellchecker_words(languages: list[str]) -> set[str]:
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


# TODO: split build_foreign_wordlist into two sets:
#   - glotlid_only (GlotLID discriminative lists minus Mooré) → used for the hard
#     foreign_words flag; avoids false positives from short tokens like 'zap' that
#     appear in broad spellchecker dictionaries but not in GlotLID lists.
#   - full_foreign (GlotLID + spellchecker minus Mooré) → used for the soft
#     identification_consistency score, which benefits from broader coverage.
#   Verified via check_wordlists.py: 'zap' is eng_spell=1 but eng_glotlid=0,
#   confirming the spellchecker is the source of the false positive.
def build_foreign_wordlist(
    foreign_langs: list[str] | None = None,
    moore_langs: list[str] | None = None,
) -> set[str]:
    """Build a foreign-word exclusion set for quality-warning checks.

    Combines GlotLID word lists and pyspellchecker dictionaries for the given
    *foreign_langs*, then removes any words that also appear in *moore_langs*
    (so genuine Mooré loanwords are not flagged).

    Args:
        foreign_langs: GlotLID language codes to treat as foreign
                       (default: ``["fra_Latn", "eng_Latn"]``).
        moore_langs:   GlotLID language codes for the target language used to
                       subtract overlap (default: ``["mos_Latn"]``).

    Returns:
        A ``set[str]`` of lowercase foreign words, with Mooré overlap removed.
    """
    if foreign_langs is None:
        foreign_langs = ["fra_Latn", "eng_Latn"]
    if moore_langs is None:
        moore_langs = ["mos_Latn"]

    glotlid_foreign = load_glotlid_wordlists(foreign_langs)
    spell_foreign = load_spellchecker_words(foreign_langs)
    raw_foreign = glotlid_foreign | spell_foreign
    moore_wordlist = load_glotlid_wordlists(moore_langs)
    foreign = raw_foreign - moore_wordlist
    print(
        f"  Exclusive foreign wordlist: {len(foreign):,} words "
        f"({len(raw_foreign) - len(foreign):,} removed as Mooré overlap)."
    )
    return foreign
