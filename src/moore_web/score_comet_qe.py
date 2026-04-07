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


def load_model():
    from comet import download_model, load_from_checkpoint

    print("Loading McGill-NLP/ssa-comet-qe …")
    return load_from_checkpoint(download_model("McGill-NLP/ssa-comet-qe"))


def score_dataset(
    dataset,
    src_field: str = "french",
    tgt_field: str = "moore",
    output_field: str | None = None,
    batch_size: int = 8,
    gpus: int = 1,
    model=None,
):
    """Add COMET-QE scores to every row of a HuggingFace ``Dataset``.

    Args:
        dataset:      Input ``datasets.Dataset``.
        src_field:    Source column name (default: ``"french"``).
        tgt_field:    Target column name (default: ``"moore"``).
        output_field: Name of the new score column. Defaults to
                      ``"comet_qe_{src_field}_{tgt_field}"`` when ``None``.
        batch_size:   Rows per inference batch.
        gpus:         Number of GPUs to use (0 = CPU).
        model:        Pre-loaded COMET model; loaded automatically if ``None``.

    Returns:
        Annotated ``datasets.Dataset`` with an added score column.
    """
    if output_field is None:
        output_field = f"comet_qe_{src_field}_{tgt_field}"
    if model is None:
        model = load_model()

    def _score_batch(batch: dict) -> dict:
        data = [{"src": s, "mt": t} for s, t in zip(batch[src_field], batch[tgt_field])]
        output = model.predict(data, batch_size=batch_size, gpus=gpus)
        batch[output_field] = [round(float(s), 4) for s in output.scores]
        return batch

    print(f"Scoring {len(dataset):,} pairs with COMET-QE…")
    return dataset.map(_score_batch, batched=True, desc="comet-qe")


def score_file(
    path: Path,
    output_path: Path,
    src_field: str,
    mt_field: str,
    batch_size: int,
    gpus: int,
    model,
    output_field: str | None = None,
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

    if output_field is None:
        output_field = f"comet_qe_{src_field}_{mt_field}"

    data = [{"src": r[src_field], "mt": r[mt_field]} for r in rows]

    print(f"Scoring {len(data)} pairs from {path.name} …")
    output = model.predict(data, batch_size=batch_size, gpus=gpus)

    for row, qe_score in zip(rows, output.scores):
        row[output_field] = round(float(qe_score), 4)

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
        "--output",
        "-o",
        default=None,
        metavar="PATH",
        help=(
            "Output path. With a single input: treated as a file path. "
            "With multiple inputs: treated as a directory (files keep their original names). "
            "Default: overwrite inputs in-place."
        ),
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

    model = load_model()
    single_input = len(args.inputs) == 1

    for input_str in args.inputs:
        input_path = Path(input_str)
        if not input_path.exists():
            print(f"[warn] {input_path} not found, skipping.")
            continue

        if args.output:
            out = Path(args.output)
            # Single input + explicit path → use it directly as the output file.
            # Multiple inputs → treat as a directory.
            out_path = out if single_input else out / input_path.name
        else:
            out_path = input_path  # in-place

        score_file(
            path=input_path,
            output_path=out_path,
            src_field=args.src_field,
            mt_field=args.mt_field,
            batch_size=args.batch_size,
            gpus=args.gpus,
            model=model,
        )
