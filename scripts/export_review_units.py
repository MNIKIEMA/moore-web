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

    uv run python scripts/export_review_units.py --source kade \
        --fr-input "kadé_fr.pdf" --mo-input "kadé_mos.pdf" \
        -o data/review/kade_units.jsonl

    uv run python scripts/export_review_units.py --source messages-nouvel-an \
        --input ../faso-web-docs/messages-nouvel-an \
        -o data/review/messages-nouvel-an_units.jsonl

    uv run python scripts/export_review_units.py --source udhr \
        --fr-input ../faso-web-docs/universal-declaration-human-rights/udhr-fra.txt \
        --mo-input ../faso-web-docs/universal-declaration-human-rights/udhr-mos.txt \
        -o data/review/udhr_units.jsonl

    ``--source facilitateur`` is accepted as an alias for ``kade``.
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


def export_kade(fr_input: str, mo_input: str, output_path: str) -> int:
    from moore_web.cli import KadeLang, _parse_kade_file
    from moore_web.flatten import flatten_facilitateur_pair_per_unit

    fr_book = _parse_kade_file(Path(fr_input), KadeLang.french)
    mo_book = _parse_kade_file(Path(mo_input), KadeLang.moore)
    units = flatten_facilitateur_pair_per_unit(fr_book, mo_book, segment=True)
    return _write_units(units, output_path)


def export_new_year(input_path: str, output_path: str) -> int:
    """One unit for the whole address: its curated blocks don't line up across languages."""
    from moore_web.flatten import ParallelText, segment_fr, segment_mo
    from moore_web.new_year_message import prepare_new_year_pair

    collection_dir = Path(input_path)
    parallel = prepare_new_year_pair(collection_dir)
    manifest = json.loads((collection_dir / "manifest.json").read_text(encoding="utf-8"))
    unit = ParallelText(
        french=[s for block in parallel.french for s in segment_fr(block)],
        moore=[s for block in parallel.moore for s in segment_mo(block)],
        source=parallel.source,
    )
    return _write_units([(f"new-year-{manifest.get('subject_year', 'message')}", unit)], output_path)


def export_udhr(fr_input: str, mo_input: str, output_path: str) -> int:
    from moore_web.udhr import udhr_review_units

    units, skipped = udhr_review_units(Path(fr_input), Path(mo_input))
    for reason in skipped:
        print(f"Skipped {reason}")
    return _write_units(units, output_path)


# Sources needing one input (PDF, JSON, directory, ...).
EXPORTERS = {
    "sida": export_sida,
    "raamde": export_raamde,
    "messages-nouvel-an": export_new_year,
}

# Sources needing a French and a Mooré input.
PAIR_EXPORTERS = {
    "facilitateur": export_kade,
    "kade": export_kade,
    "udhr": export_udhr,
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--source",
        required=True,
        choices=sorted([*EXPORTERS, *PAIR_EXPORTERS]),
        help="Which document family to export. Adding another single-input source means "
        "writing an export_<source>(input_path, output_path) -> int that yields the same "
        "{uid: {fra, mos}} shape per line and registering it in EXPORTERS.",
    )
    p.add_argument(
        "--input", "-i", help="Path to the source file (PDF, JSON, ...) -- single-input sources only."
    )
    p.add_argument("--fr-input", help="Path to the French PDF/TXT (paired sources only).")
    p.add_argument("--mo-input", help="Path to the Mooré PDF/TXT (paired sources only).")
    p.add_argument("--output", "-o", required=True, help="Output JSONL path.")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.source in PAIR_EXPORTERS:
        if not args.fr_input or not args.mo_input:
            raise SystemExit(f"--fr-input and --mo-input are both required for --source {args.source}")
        n = PAIR_EXPORTERS[args.source](args.fr_input, args.mo_input, args.output)
    else:
        if not args.input:
            raise SystemExit(f"--input is required for --source {args.source}")
        n = EXPORTERS[args.source](args.input, args.output)

    print(f"Wrote {n} units -> {args.output}")


if __name__ == "__main__":
    main()
