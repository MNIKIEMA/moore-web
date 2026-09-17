#!/usr/bin/env python3
"""Export a bilingual source into the review JSONL schema.

Each output line is one JSON object: ``{"<unit_id>": {"fra": [...], "mos": [...]}}``
-- one line per reviewable unit (a page, an enum item, a news article...),
sentence-segmented. ``unit_id`` is just an opaque, stable, unique string --
it doesn't need to be numeric or follow any convention, and different
sources should use whatever label is natural for them (``page-3``,
``enum-4``, a news article's URL, ...). This is the schema
``notebooks/merge_review.py`` expects, so any source exported through it can
be reviewed with that same notebook, regardless of which parser produced it.

Usage:
    uv run python scripts/export_review_units.py --source sida \
        --input "data/2 SIDA mooré - français.pdf" \
        -o data/review/sida_units.jsonl

    uv run python scripts/export_review_units.py --source raamde \
        --input data/raamde/raamde_corpus_with_lang_id.json \
        -o data/review/raamde_units.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _write_units(units: list[tuple[str, object]], output_path: str) -> int:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for uid, parallel in units:
            row = {uid: {"fra": parallel.french, "mos": parallel.moore}}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(units)


def export_sida(input_path: str, output_path: str) -> int:
    from moore_web.book_parser import parse_pdf_to_json
    from moore_web.flatten import flatten_sida_book_per_unit

    chapters = parse_pdf_to_json(input_path)
    units = flatten_sida_book_per_unit(chapters, segment=True)
    return _write_units(units, output_path)


def export_raamde(input_path: str, output_path: str) -> int:
    from moore_web.flatten import flatten_news_per_entry
    from moore_web.segment_news_data import segment_entries

    corpus = json.loads(Path(input_path).read_text(encoding="utf-8"))
    corpus = segment_entries(corpus)
    units = flatten_news_per_entry(corpus, segment=True)
    return _write_units(units, output_path)


EXPORTERS = {
    "sida": export_sida,
    "raamde": export_raamde,
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--source",
        required=True,
        choices=sorted(EXPORTERS),
        help="Which document family to export. Adding another source means writing an "
        "export_<source>(input_path, output_path) -> int that yields the same "
        "{uid: {fra, mos}} shape per line and registering it in EXPORTERS.",
    )
    p.add_argument("--input", "-i", required=True, help="Path to the source file (PDF, JSON, ...).")
    p.add_argument("--output", "-o", required=True, help="Output JSONL path.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    n = EXPORTERS[args.source](args.input, args.output)
    print(f"Wrote {n} units -> {args.output}")


if __name__ == "__main__":
    main()
