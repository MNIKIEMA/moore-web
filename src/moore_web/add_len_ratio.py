"""Add length-ratio scores to aligned JSONL files or HuggingFace datasets.

The length ratio is defined as::

    min(len(src), len(tgt)) / max(len(src), len(tgt))

A value of 1.0 means both sides are the same character length; values close to
0.0 indicate a strong imbalance (e.g. one side is much shorter than the other).

Input
-----
Either local JSONL files or a HuggingFace Hub dataset repo ID.

Output (JSONL mode)
-------------------
Each input file is re-written in-place (or to ``--output``) with an extra
``len_ratio`` field per row.

Output (HF mode)
----------------
The annotated dataset is pushed to ``--hub-repo`` and/or saved locally to
``--output`` as JSONL.

Usage
-----
    # Annotate a single file in-place
    uv run python -m moore_web.add_len_ratio final_data_hf/conseils_ministres_aligned.jsonl

    # Annotate all aligned files into a separate directory
    uv run python -m moore_web.add_len_ratio final_data_hf/*_aligned.jsonl --output final_data_hf_ratio

    # HuggingFace dataset (annotate and push back)
    uv run python -m moore_web.add_len_ratio --hf-dataset madoss/nllb-mos-lid --hub-repo madoss/nllb-mos-lid-ratio

    # HuggingFace dataset, save locally only
    uv run python -m moore_web.add_len_ratio --hf-dataset madoss/nllb-mos-lid --output out.jsonl

    # Custom field names
    uv run python -m moore_web.add_len_ratio data.jsonl --src-field english --tgt-field moore
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------


def _len_ratio(src: str, tgt: str) -> float:
    """Return min(len(src), len(tgt)) / max(len(src), len(tgt)).

    Returns 0.0 when either side is empty.
    """
    a, b = len(src), len(tgt)
    if not a or not b:
        return 0.0
    return round(min(a, b) / max(a, b), 4)


def _print_stats(scores: list[float], label: str) -> None:
    print(
        f"  → {label}  "
        f"mean={statistics.mean(scores):.4f}  "
        f"median={statistics.median(scores):.4f}  "
        f"min={min(scores):.4f}  max={max(scores):.4f}"
    )


# ---------------------------------------------------------------------------
# JSONL mode
# ---------------------------------------------------------------------------


def annotate_file(
    path: Path,
    output_path: Path,
    src_field: str,
    tgt_field: str,
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

    print(f"Annotating {len(rows)} pairs from {path.name} …")
    scores = []
    for row in rows:
        ratio = _len_ratio(row.get(src_field, "") or "", row.get(tgt_field, "") or "")
        row["len_ratio"] = ratio
        scores.append(ratio)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    _print_stats(scores, output_path.name)


# ---------------------------------------------------------------------------
# HuggingFace dataset mode
# ---------------------------------------------------------------------------


def _annotate_batch(batch: dict[str, list], src_field: str, tgt_field: str) -> dict[str, list]:
    src_texts = batch[src_field]
    tgt_texts = batch[tgt_field]
    batch["len_ratio"] = [_len_ratio(s or "", t or "") for s, t in zip(src_texts, tgt_texts)]
    return batch


def annotate_hf_dataset(
    source_repo: str,
    hub_repo: str | None,
    output: str | None,
    src_field: str,
    tgt_field: str,
    split: str,
    batch_size: int,
    private: bool,
) -> None:
    from datasets import DatasetDict, load_dataset

    print(f"Loading '{source_repo}' (split={split}) …")
    ds = load_dataset(source_repo, split=split)
    print(f"Loaded {len(ds):,} rows.")

    for field in (src_field, tgt_field):
        if field not in ds.column_names:
            raise ValueError(f"Column '{field}' not found. Available: {ds.column_names}")

    print("Computing len_ratio …")
    ds = ds.map(
        lambda batch: _annotate_batch(batch, src_field, tgt_field),
        batched=True,
        batch_size=batch_size,
        desc="len_ratio",
    )

    scores = ds["len_ratio"]
    _print_stats(scores, source_repo)

    if output:
        print(f"\nWriting {len(ds):,} rows → {output}")
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for row in ds:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print("Done (local).")

    if hub_repo:
        print(f"\nPushing {len(ds):,} rows → '{hub_repo}' …")
        DatasetDict({split: ds}).push_to_hub(hub_repo, private=private)
        print(f"Done. https://huggingface.co/datasets/{hub_repo}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Add length-ratio scores to aligned JSONL files or HF datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "inputs",
        nargs="*",
        metavar="FILE",
        help="One or more local JSONL files to annotate.",
    )
    parser.add_argument(
        "--hf-dataset",
        default=None,
        metavar="REPO_ID",
        help="HuggingFace Hub dataset repo ID to load (e.g. madoss/nllb-mos-lid). "
        "Mutually exclusive with FILE positional arguments.",
    )

    parser.add_argument(
        "--output",
        "-o",
        default=None,
        metavar="PATH",
        help=(
            "Output path. "
            "JSONL mode — single input: file path; multiple inputs: directory. "
            "HF mode — local JSONL file to write. "
            "Default: overwrite inputs in-place (JSONL mode) or skip local write (HF mode)."
        ),
    )
    parser.add_argument(
        "--hub-repo",
        default=None,
        metavar="REPO_ID",
        help="HF Hub repo to push annotated dataset to (HF mode only).",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to load (HF mode only, default: %(default)s).",
    )
    parser.add_argument(
        "--src-field",
        default="french",
        help="Field for the source text (default: %(default)s).",
    )
    parser.add_argument(
        "--tgt-field",
        default="moore",
        help="Field for the target text (default: %(default)s).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Rows per batch for dataset.map (HF mode only, default: %(default)s).",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the pushed HF Hub dataset private (HF mode only).",
    )
    args = parser.parse_args()

    if args.hf_dataset and args.inputs:
        parser.error("FILE arguments and --hf-dataset are mutually exclusive.")
    if not args.hf_dataset and not args.inputs:
        parser.error("Provide at least one FILE or use --hf-dataset.")

    if args.hf_dataset:
        annotate_hf_dataset(
            source_repo=args.hf_dataset,
            hub_repo=args.hub_repo,
            output=args.output,
            src_field=args.src_field,
            tgt_field=args.tgt_field,
            split=args.split,
            batch_size=args.batch_size,
            private=args.private,
        )
    else:
        single_input = len(args.inputs) == 1
        for input_str in args.inputs:
            input_path = Path(input_str)
            if not input_path.exists():
                print(f"[warn] {input_path} not found, skipping.")
                continue
            if args.output:
                out = Path(args.output)
                out_path = out if single_input else out / input_path.name
            else:
                out_path = input_path  # in-place
            annotate_file(
                path=input_path,
                output_path=out_path,
                src_field=args.src_field,
                tgt_field=args.tgt_field,
            )
