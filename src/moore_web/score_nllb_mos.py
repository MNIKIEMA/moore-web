"""Add COMET-QE scores to the raw eng↔mos dataset and upload to HF.

Assumes the raw dataset has already been pushed to HF Hub by
``moore_web.upload_nllb_raw``.

Usage
-----
    # Score a split from HF Hub and push results
    uv run python -m moore_web.score_nllb_mos score --source-repo madoss/nllb-mos-raw
    uv run python -m moore_web.score_nllb_mos score --source-repo madoss/nllb-mos-raw --min-laser 0.5

    # Translate English segments to French and push
    uv run python -m moore_web.score_nllb_mos translate --source-repo madoss/nllb-mos-raw

    # Download raw TSV and score in one step (no separate upload)
    uv run python -m moore_web.score_nllb_mos full
"""

from __future__ import annotations

import argparse
import statistics

from dotenv import load_dotenv

from moore_web.translation import translate_and_upload
from moore_web.upload_nllb_raw import _COLS as _TSV_COLS, _load_nllb_tsv

load_dotenv()

# ---------------------------------------------------------------------------
# COMET-QE scoring
# ---------------------------------------------------------------------------


def _comet_scores(
    eng_sentences: list[str],
    mos_sentences: list[str],
    batch_size: int = 8,
    accelerator: str = "auto",
    chunk_size: int = 4096,
    num_workers: int = 0,
) -> list[float]:
    """Return per-pair COMET-QE scores using McGill-NLP/ssa-comet-qe.

    For large datasets, ``chunk_size`` controls how many pairs are passed to
    ``model.predict`` at once to avoid OOM errors.  COMET already handles
    mini-batching internally (``batch_size``), so chunking is only needed when
    the full dataset is too large to hold in GPU memory all at once.  Set
    ``chunk_size=0`` to disable chunking and process everything in one call.
    """
    from comet import download_model, load_from_checkpoint
    from tqdm import tqdm

    print("Loading COMET-QE model (McGill-NLP/ssa-comet-qe)…")
    model_path = download_model("McGill-NLP/ssa-comet-qe")
    model = load_from_checkpoint(model_path)

    data = [{"src": src, "mt": mt} for src, mt in zip(eng_sentences, mos_sentences)]
    n = len(data)
    print(f"Scoring {n} pairs with COMET-QE (batch_size={batch_size}, accelerator={accelerator})…")

    effective_chunk = chunk_size if chunk_size and chunk_size < n else n
    all_scores: list[float] = []
    for i in tqdm(range(0, n, effective_chunk), desc="COMET chunks", unit="chunk"):
        chunk = data[i : i + effective_chunk]
        output = model.predict(chunk, batch_size=batch_size, accelerator=accelerator, num_workers=num_workers)
        all_scores.extend(round(float(s), 4) for s in output["scores"])

    print(
        f"COMET-QE scores — mean: {statistics.mean(all_scores):.3f}  "
        f"median: {statistics.median(all_scores):.3f}  "
        f"min: {min(all_scores):.3f}  max: {max(all_scores):.3f}"
    )
    return all_scores


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def score_and_upload(
    hub_repo: str = "madoss/nllb-mos",
    source_repo: str | None = None,
    min_laser: float = 0.0,
    comet_batch_size: int = 8,
    accelerator: str = "auto",
    private: bool = False,
    rows_slice: slice | None = None,
) -> None:
    """Add COMET-QE scores to an existing HF dataset and push the result.

    Loads from ``source_repo`` if provided, otherwise re-downloads the raw TSV.

    Args:
        hub_repo:          HuggingFace repo to push the scored dataset to.
        source_repo:       HuggingFace repo to load the raw dataset from.
                           If None, the raw TSV is downloaded directly.
        min_laser:         Drop pairs whose NLLB laser_score is below this.
        comet_batch_size:  Batch size for COMET-QE inference.
        accelerator:       PyTorch Lightning accelerator (``"auto"``, ``"gpu"``, ``"cpu"``).
        private:           Whether to make the HF Hub dataset private.
        rows_slice:        Optional ``slice`` to select a subset of rows *after*
                           laser filtering, e.g. ``slice(0, 1000)`` for the first
                           1 000 pairs.  ``None`` keeps all rows.
    """
    from datasets import Dataset, DatasetDict, load_dataset

    if source_repo:
        # When no laser filtering is needed we can push the slice directly into
        # the split string so only the requested rows are downloaded.
        if rows_slice is not None and min_laser == 0.0:
            start = rows_slice.start or ""
            stop = rows_slice.stop or ""
            split = f"train[{start}:{stop}]"
            print(f"Loading dataset from HuggingFace Hub: '{source_repo}' (split='{split}')…")
            ds = load_dataset(source_repo, split=split)
            rows_slice = None  # already applied
        else:
            print(f"Loading dataset from HuggingFace Hub: '{source_repo}'…")
            ds = load_dataset(source_repo, split="train")
        rows = [dict(zip(ds.column_names, vals)) for vals in zip(*[ds[col] for col in ds.column_names])]
    else:
        rows = _load_nllb_tsv()

    if min_laser > 0.0:
        before = len(rows)
        rows = [r for r in rows if (r["laser_score"] or 0.0) >= min_laser]
        print(f"Dropped {before - len(rows)} pairs with laser_score < {min_laser}. Keeping {len(rows)}.")

    if rows_slice is not None:
        before = len(rows)
        rows = rows[rows_slice]
        print(f"Row selection {rows_slice}: using {len(rows)} of {before} pairs.")

    eng_sentences = [r["eng_Latn"] for r in rows]
    mos_sentences = [r["mos_Latn"] for r in rows]

    comet_scores = _comet_scores(
        eng_sentences, mos_sentences, batch_size=comet_batch_size, accelerator=accelerator
    )

    cols = list(rows[0].keys()) if rows else _TSV_COLS
    dataset = Dataset.from_dict(
        {
            **{col: [r[col] for r in rows] for col in cols},
            "comet_qe": comet_scores,
        }
    )

    dataset_dict = DatasetDict({"train": dataset})

    print(f"\nPushing scored dataset to HuggingFace Hub as '{hub_repo}'…")
    dataset_dict.push_to_hub(hub_repo, private=private)
    print(f"Done. Dataset available at https://huggingface.co/datasets/{hub_repo}")



def full_pipeline(
    hub_repo: str = "madoss/nllb-mos",
    min_laser: float = 0.0,
    comet_batch_size: int = 8,
    accelerator: str = "auto",
    private: bool = False,
    rows_slice: slice | None = None,
) -> None:
    """Download, score, and upload in one step (original behaviour)."""
    score_and_upload(
        hub_repo=hub_repo,
        source_repo=None,
        min_laser=min_laser,
        comet_batch_size=comet_batch_size,
        accelerator=accelerator,
        private=private,
        rows_slice=rows_slice,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_rows(value: str | None) -> slice | None:
    if value is None:
        return None
    parts = value.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--rows must be in START:END format, e.g. '0:1000'")
    start = int(parts[0]) if parts[0] else None
    end = int(parts[1]) if parts[1] else None
    return slice(start, end)


def cli():
    """Build and return the argument parser."""
    parser = argparse.ArgumentParser(
        description="Score or translate the NLLB eng↔mos dataset with COMET-QE / TranslateGemma.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- score subcommand ----------------------------------------------------
    sc_parser = subparsers.add_parser(
        "score",
        help="Add COMET-QE scores to a (raw) HF dataset and push to HF Hub.",
    )
    sc_parser.add_argument(
        "--hub-repo",
        default="madoss/nllb-mos",
        help="HuggingFace Hub repo to push the scored dataset to (default: %(default)s).",
    )
    sc_parser.add_argument(
        "--source-repo",
        default=None,
        help="HF Hub repo to load the raw dataset from. If omitted, re-downloads the TSV.",
    )
    sc_parser.add_argument(
        "--min-laser",
        type=float,
        default=0.0,
        help="Drop pairs with LASER score below this value (default: keep all).",
    )
    sc_parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="COMET-QE inference batch size (default: %(default)s).",
    )
    sc_parser.add_argument(
        "--accelerator",
        default="auto",
        help="PyTorch Lightning accelerator: 'auto', 'gpu', or 'cpu' (default: %(default)s).",
    )
    sc_parser.add_argument(
        "--rows",
        default=None,
        metavar="START:END",
        help="Select a slice of rows to score, e.g. '0:1000' or '500:' (default: all).",
    )
    sc_parser.add_argument("--private", action="store_true", help="Make the dataset private.")

    # -- translate subcommand ------------------------------------------------
    tr_parser = subparsers.add_parser(
        "translate",
        help="Translate English segments to French with TranslateGemma and push to HF Hub.",
    )
    tr_parser.add_argument(
        "--hub-repo",
        default="madoss/nllb-mos-fr",
        help="HuggingFace Hub repo to push the translated dataset to (default: %(default)s).",
    )
    tr_parser.add_argument(
        "--source-repo",
        default=None,
        help="HF Hub repo to load the source dataset from. If omitted, re-downloads the TSV.",
    )
    tr_parser.add_argument(
        "--model",
        default="google/translategemma-12b-it",
        help="TranslateGemma model ID (default: %(default)s).",
    )
    tr_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Sentences per vLLM generation call (default: %(default)s).",
    )
    tr_parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate per sentence (default: %(default)s).",
    )
    tr_parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: %(default)s).",
    )
    tr_parser.add_argument("--private", action="store_true", help="Make the dataset private.")

    # -- full subcommand (legacy) --------------------------------------------
    full_parser = subparsers.add_parser(
        "full",
        help="Download, score, and upload in one step.",
    )
    full_parser.add_argument(
        "--hub-repo",
        default="madoss/nllb-mos",
        help="HuggingFace Hub repo to push to (default: %(default)s).",
    )
    full_parser.add_argument(
        "--min-laser",
        type=float,
        default=0.0,
        help="Drop pairs with LASER score below this value (default: keep all).",
    )
    full_parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="COMET-QE inference batch size (default: %(default)s).",
    )
    full_parser.add_argument(
        "--accelerator",
        default="auto",
        help="PyTorch Lightning accelerator: 'auto', 'gpu', or 'cpu' (default: %(default)s).",
    )
    full_parser.add_argument(
        "--rows",
        default=None,
        metavar="START:END",
        help="Select a slice of rows to score, e.g. '0:1000' or '500:' (default: all).",
    )
    full_parser.add_argument("--private", action="store_true", help="Make the dataset private.")

    args = parser.parse_args()
    return args


def main() -> None:
    """Dispatch parsed CLI arguments to the appropriate pipeline function."""
    args = cli()
    slice_args = _parse_rows(args.rows) if args.command in ("score", "full") else None
    if args.command == "translate":
        translate_and_upload(
            hub_repo=args.hub_repo,
            source_repo=args.source_repo,
            model=args.model,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            tensor_parallel_size=args.tensor_parallel_size,
            private=args.private,
        )
    elif args.command == "score":
        score_and_upload(
            hub_repo=args.hub_repo,
            source_repo=args.source_repo,
            min_laser=args.min_laser,
            comet_batch_size=args.batch_size,
            accelerator=args.accelerator,
            private=args.private,
            rows_slice=slice_args,
        )
    elif args.command == "full":
        full_pipeline(
            hub_repo=args.hub_repo,
            min_laser=args.min_laser,
            comet_batch_size=args.batch_size,
            accelerator=args.accelerator,
            private=args.private,
            rows_slice=slice_args,
        )


if __name__ == "__main__":
    main()
