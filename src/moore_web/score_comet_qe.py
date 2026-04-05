"""Add COMET-QE scores to aligned JSONL files.

Input
-----
One or more ``*_aligned.jsonl`` files where each line has::

    {"french": "...", "moore": "...", "laser_score": 0.82, "source": "..."}

Output
------
Each input file is re-written in-place (or to ``--output-dir``) with an extra
``comet_qe`` field per row.

Usage
-----
    # Score a single file in-place
    uv run python -m moore_web.score_comet_qe final_data_hf/conseils_ministres_aligned.jsonl

    # Score all aligned files into a separate directory
    uv run python -m moore_web.score_comet_qe final_data_hf/*_aligned.jsonl --output-dir final_data_hf_comet

    # French source instead of English (experimental)
    uv run python -m moore_web.score_comet_qe final_data_hf/*.jsonl --src-field french

Notes
-----
- Model: ``McGill-NLP/ssa-comet-qe`` (reference-free, ~1.5 GB download on first run).
- Scores are calibrated for English sources; French sources give *relative* quality
  signals useful for filtering but absolute values are less meaningful.
- Use --gpus 0 to force CPU (slow on large files).
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_file(
    path: Path,
    output_path: Path,
    src_field: str,
    mt_field: str,
    batch_size: int,
    gpus: int,
) -> None:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        print(f"[skip] {path} — empty file")
        return

    from comet import download_model, load_from_checkpoint

    model_path = download_model("McGill-NLP/ssa-comet-qe")
    model = load_from_checkpoint(model_path)

    data = [{"src": r[src_field], "mt": r[mt_field]} for r in rows]

    print(f"Scoring {len(data)} pairs from {path.name} …")
    output = model.predict(data, batch_size=batch_size, gpus=gpus)

    for row, qe_score in zip(rows, output.scores):
        row["comet_qe"] = round(float(qe_score), 6)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    scores = output.scores
    print(
        f"  → {output_path.name}  "
        f"mean={statistics.mean(scores):.4f}  "
        f"median={statistics.median(scores):.4f}  "
        f"min={min(scores):.4f}  max={max(scores):.4f}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Add COMET-QE scores to aligned JSONL files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        metavar="FILE",
        help="One or more *_aligned.jsonl files to score.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        metavar="DIR",
        help="Write scored files here (default: overwrite inputs in-place).",
    )
    parser.add_argument(
        "--src-field",
        default="french",
        help="JSONL field to use as translation source (default: %(default)s).",
    )
    parser.add_argument(
        "--mt-field",
        default="moore",
        help="JSONL field to use as MT output (default: %(default)s).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for COMET inference (default: %(default)s).",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs to use; set 0 for CPU (default: %(default)s).",
    )
    args = parser.parse_args()

    for input_str in args.inputs:
        input_path = Path(input_str)
        if not input_path.exists():
            print(f"[warn] {input_path} not found, skipping.")
            continue

        if args.output_dir:
            out_path = Path(args.output_dir) / input_path.name
        else:
            out_path = input_path  # in-place

        score_file(
            path=input_path,
            output_path=out_path,
            src_field=args.src_field,
            mt_field=args.mt_field,
            batch_size=args.batch_size,
            gpus=args.gpus,
        )
