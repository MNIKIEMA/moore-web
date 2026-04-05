"""Build and push language wordlists to HuggingFace Hub.

Loads French and English wordlists from ``cis-lmu/glotlid-wordlists``, then
builds the Mooré wordlist by merging the GlotLID Mooré entries with lemmas
extracted from the local lexicon (``final_data_hf/lexicon_entries.jsonl``).

The result is pushed as three dataset configs on the same HF Hub repo:
  - ``fra_Latn`` — French words
  - ``eng_Latn`` — English words
  - ``mos_Latn`` — Mooré words (GlotLID + lexicon)

Usage
-----
    uv run python -m moore_web.build_wordlists --hub-repo madoss/moore-wordlists
    uv run python -m moore_web.build_wordlists --hub-repo madoss/moore-wordlists --lexicon path/to/lexicon_entries.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_DEFAULT_LEXICON = Path(__file__).parents[2] / "final_data_hf" / "lexicon_entries.jsonl"


def _load_glotlid(lang: str) -> set[str]:
    """Load one language config from ``cis-lmu/glotlid-wordlists``."""
    from datasets import load_dataset

    words: set[str] = set()
    try:
        ds = load_dataset("cis-lmu/glotlid-wordlists", name=lang, split="train")
        for row in ds:
            w = row.get("text") or row.get("word", "")
            if w:
                words.add(w.lower())
        print(f"  glotlid[{lang}]: {len(words):,} words")
    except Exception as e:
        print(f"  Warning: could not load glotlid-wordlists[{lang}]: {e}")
    return words


def _load_lexicon_moore(path: Path) -> set[str]:
    """Extract all Mooré tokens from the lexicon JSONL.

    Each entry has a ``moore`` field (e.g. ``"kenge [è]"``).  We take the raw
    lemma and also split on whitespace/brackets to capture individual tokens.
    """
    words: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            raw = entry.get("moore", "")
            if not raw:
                continue
            # Strip IPA/pronunciation annotations in brackets e.g. "kenge [è]" → "kenge"
            clean = re.sub(r"\s*[\[\(][^\]\)]*[\]\)]", "", raw).strip()
            # Split on whitespace and process each token individually
            for token in clean.lower().split():
                # Drop placeholder/noise tokens
                if token.startswith("@"):  # e.g. "@email"
                    continue
                # Strip leading dots (ellipsis markers e.g. "...ye" → "ye")
                token = token.lstrip(".")
                # Strip leading hyphens (morpheme markers e.g. "-ame" → "ame")
                token = token.lstrip("-")
                if len(token) >= 2:
                    words.add(token)
    print(f"  lexicon[mos_Latn]: {len(words):,} words (from {path.name})")
    return words


def build_and_push(
    hub_repo: str,
    lexicon: Path = _DEFAULT_LEXICON,
    private: bool = False,
) -> None:
    from datasets import Dataset, DatasetDict

    print("Loading wordlists…")
    fra_words = _load_glotlid("fra_Latn")
    eng_words = _load_glotlid("eng_Latn")
    mos_glotlid = _load_glotlid("mos_Latn")
    mos_lexicon = _load_lexicon_moore(lexicon)
    mos_words = mos_glotlid | mos_lexicon
    print(
        f"  mos_Latn total: {len(mos_words):,} (glotlid={len(mos_glotlid):,} + lexicon={len(mos_lexicon):,}, overlap={len(mos_glotlid & mos_lexicon):,})"
    )

    def _to_dataset(words: set[str]) -> Dataset:
        return Dataset.from_dict({"word": sorted(words)})

    dataset_dict = DatasetDict(
        {
            "fra_Latn": _to_dataset(fra_words),
            "eng_Latn": _to_dataset(eng_words),
            "mos_Latn": _to_dataset(mos_words),
        }
    )

    print(f"\nPushing to '{hub_repo}'…")
    dataset_dict.push_to_hub(hub_repo, private=private)
    print(f"Done. https://huggingface.co/datasets/{hub_repo}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and push language wordlists to HuggingFace Hub.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--hub-repo",
        required=True,
        help="HF Hub dataset repo to push wordlists to (e.g. 'madoss/moore-wordlists').",
    )
    parser.add_argument(
        "--lexicon",
        type=Path,
        default=_DEFAULT_LEXICON,
        help=f"Path to lexicon_entries.jsonl (default: {_DEFAULT_LEXICON}).",
    )
    parser.add_argument("--private", action="store_true", help="Make the dataset private.")
    args = parser.parse_args()
    build_and_push(hub_repo=args.hub_repo, lexicon=args.lexicon, private=args.private)


if __name__ == "__main__":
    main()
