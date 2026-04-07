"""Segment bilingual Moore/French news entries by language.

Each entry is expected to have a ``text_units`` list.  This module
provides the pure segmentation logic (no ML dependencies).  Pair it
with :mod:`moore_web.lang_id` to first annotate entries with
``text_unit_langs`` / ``text_unit_probs``.

Usage (CLI):
    uv run python -m moore_web.segment_news_data -j raamde_corpus.json -o out.json
    uv run python -m moore_web.segment_news_data -j raamde_corpus.json --no-lang-id
"""

import re

MOORE_END_MARKERS = [
    r"^Kibar[aã]\s+yii",
    r"^Z[ɩi]ll\s+KAFAANDO",
]

MOORE_LANG = "mos_Latn"
FRENCH_LANG = "fra_Latn"

MOORE_END_PATTERN = re.compile("|".join(MOORE_END_MARKERS), re.IGNORECASE)


def find_marker_boundary(text_units: list[str]) -> int | None:
    """Return the index of the text unit containing an end-of-mooré marker, inclusive."""
    for i, text in enumerate(text_units):
        if MOORE_END_PATTERN.search(text):
            return i
    return None


def segment_by_language(
    text_units: list[str],
    predicted_langs: list[str],
) -> dict:
    """Segment text units into mooré and French sections.

    Strategy:
      1. Look for an explicit end-of-mooré marker first.
      2. Fall back to the first text unit predicted as French.
      3. If neither found, treat everything as mooré.
    """
    marker_idx = find_marker_boundary(text_units)

    if marker_idx is not None:
        split_idx = marker_idx + 1
    else:
        french_indices = [i for i, lang in enumerate(predicted_langs) if lang == FRENCH_LANG]
        split_idx = french_indices[0] if french_indices else len(text_units)

    moore_units = text_units[:split_idx]
    french_units = text_units[split_idx:]

    return {
        "moore": moore_units if moore_units else None,
        "french": french_units if french_units else None,
    }


def segment_entries(entries: list[dict]) -> list[dict]:
    """Apply language segmentation to all entries."""
    for item in entries:
        text_units = item.get("text_units", [])
        langs = list(item.get("text_unit_langs") or [])

        if not text_units:
            item["segments"] = {"moore": None, "french": None}
            continue

        item["segments"] = segment_by_language(text_units, langs)

    return entries


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Annotate and segment bilingual Moore/French news entries.",
    )
    parser.add_argument(
        "--json",
        "-j",
        required=True,
        help="Path to JSON file containing list of entries with text_units.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="corpus_segmented.json",
        help="Output path for annotated JSON (default: %(default)s).",
    )
    parser.add_argument(
        "--no-lang-id",
        action="store_true",
        help="Skip lang ID prediction (use if text_unit_langs already present).",
    )
    parser.add_argument(
        "--drop-debug",
        action="store_true",
        help="Remove text_unit_langs and text_unit_probs from the final output.",
    )
    args = parser.parse_args()

    with open(args.json, encoding="utf-8") as f:
        corpus = json.load(f)

    if not args.no_lang_id:
        from moore_web.glotlid import annotate_text_units

        corpus = annotate_text_units(corpus)

    corpus = segment_entries(corpus)

    if args.drop_debug:
        for item in corpus:
            item.pop("text_unit_langs", None)
            item.pop("text_unit_probs", None)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(corpus)} entries → {args.output}")
