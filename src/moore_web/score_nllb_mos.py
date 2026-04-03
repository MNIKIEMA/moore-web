"""Add COMET-QE scores to the raw eng↔mos dataset and upload to HF.

Assumes the raw dataset has already been pushed to HF Hub by
``moore_web.upload_nllb_raw``.

Usage
-----
    # Score a split from HF Hub and push results
    uv run python -m moore_web.score_nllb_mos score --source-repo madoss/nllb-mos-raw
    uv run python -m moore_web.score_nllb_mos score --source-repo madoss/nllb-mos-raw --min-laser 0.5

    # Score only rows that pass the LID quality filter (glotlid + sentence-lid)
    uv run python -m moore_web.score_nllb_mos score --source-repo madoss/nllb-mos-raw --filter-lid

    # Translate English segments to French and push
    uv run python -m moore_web.score_nllb_mos translate --source-repo madoss/nllb-mos-raw

    # Download raw TSV and score in one step (no separate upload)
    uv run python -m moore_web.score_nllb_mos full
"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv
from functools import partial
from moore_web.translation import translate_and_upload
from moore_web.upload_nllb_raw import _load_nllb_tsv

load_dotenv()

# ---------------------------------------------------------------------------
# LID quality filter predicate
# ---------------------------------------------------------------------------


def _passes_lid(row: dict) -> bool:
    return (
        (row.get("target_glotlid_prob") or 0.0) > 0.9
        and (row.get("target_sentence_lid") or 0.0) > 0.9
        and "mos_Latn" in (row.get("target_glotlid_lang") or "")
    )


# ---------------------------------------------------------------------------
# COMET-QE scoring
# ---------------------------------------------------------------------------


def _score_batch(
    batch: dict,
    model,
    comet_batch_size: int,
    accelerator: str,
    apply_lid_filter: bool,
) -> dict:
    n = len(batch["eng_Latn"])
    if apply_lid_filter:
        idx = [i for i in range(n) if _passes_lid({k: batch[k][i] for k in batch})]
        scores: list[float | None] = [None] * n
        if idx:
            data = [{"src": batch["eng_Latn"][i], "mt": batch["mos_Latn"][i]} for i in idx]
            output = model.predict(data, batch_size=comet_batch_size, accelerator=accelerator, num_workers=0)
            for i, score in zip(idx, output["scores"]):
                scores[i] = round(float(score), 4)
    else:
        data = [{"src": src, "mt": mt} for src, mt in zip(batch["eng_Latn"], batch["mos_Latn"])]
        output = model.predict(data, batch_size=comet_batch_size, accelerator=accelerator, num_workers=0)
        scores = [round(float(s), 4) for s in output["scores"]]
    return {"comet_qe_en_mos": scores}


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
    apply_lid_filter: bool = False,
) -> None:
    """Add COMET-QE scores to an existing HF dataset and push the result.

    Loads from ``source_repo`` if provided, otherwise re-downloads the raw TSV.
    The output column is named ``comet_qe_en_mos``; rows that are skipped by
    the LID filter (when ``apply_lid_filter=True``) keep ``None`` for that
    column.

    Args:
        hub_repo:          HuggingFace repo to push the scored dataset to.
        source_repo:       HuggingFace repo to load the raw dataset from.
                           If None, the raw TSV is downloaded directly.
        min_laser:         Drop pairs whose NLLB laser_score is below this.
        comet_batch_size:  COMET-QE internal DataLoader mini-batch size.
        accelerator:       PyTorch Lightning accelerator (``"auto"``, ``"gpu"``, ``"cpu"``).
        private:           Whether to make the HF Hub dataset private.
        rows_slice:        Optional ``slice`` to select a subset of rows *after*
                           laser filtering, e.g. ``slice(0, 1000)`` for the first
                           1 000 pairs.  ``None`` keeps all rows.
        apply_lid_filter:  When True, only score rows that pass the LID quality
                           filter (target_glotlid_prob > 0.9,
                           target_sentence_lid > 0.9, target_glotlid_lang
                           contains ``mos_Latn``).  Other rows are kept in the
                           dataset with ``comet_qe_en_mos=None``.
    """
    from comet import download_model, load_from_checkpoint
    from datasets import Dataset, DatasetDict, load_dataset

    if source_repo:
        ds = load_dataset(source_repo, split="train")
    else:
        ds = Dataset.from_list(_load_nllb_tsv())

    if min_laser > 0.0:
        before = len(ds)
        ds = ds.filter(lambda r: (r["laser_score"] or 0.0) >= min_laser)
        print(f"Dropped {before - len(ds)} pairs with laser_score < {min_laser}. Keeping {len(ds)}.")

    if rows_slice is not None:
        ds = ds.select(range(*rows_slice.indices(len(ds))))

    print("Loading COMET-QE model (McGill-NLP/ssa-comet-qe)…")
    model = load_from_checkpoint(download_model("McGill-NLP/ssa-comet-qe"))

    score_fn = partial(
        _score_batch,
        model=model,
        comet_batch_size=comet_batch_size,
        accelerator=accelerator,
        apply_lid_filter=apply_lid_filter,
    )
    ds = ds.map(score_fn, batched=True, desc="COMET-QE scoring")

    print(f"\nPushing scored dataset to HuggingFace Hub as '{hub_repo}'…")
    DatasetDict({"train": ds}).push_to_hub(hub_repo, private=private)
    print(f"Done. Dataset available at https://huggingface.co/datasets/{hub_repo}")


def full_pipeline(
    hub_repo: str = "madoss/nllb-mos",
    min_laser: float = 0.0,
    comet_batch_size: int = 8,
    accelerator: str = "auto",
    private: bool = False,
    rows_slice: slice | None = None,
    apply_lid_filter: bool = False,
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
        apply_lid_filter=apply_lid_filter,
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
    sc_parser.add_argument(
        "--filter-lid",
        action="store_true",
        help=(
            "Only score rows that pass the LID quality filter "
            "(target_glotlid_prob > 0.9, target_sentence_lid > 0.9, "
            "target_glotlid_lang contains 'mos_Latn'). "
            "Other rows are kept with comet_qe_en_mos=null."
        ),
    )

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
    full_parser.add_argument(
        "--filter-lid",
        action="store_true",
        help=(
            "Only score rows that pass the LID quality filter "
            "(target_glotlid_prob > 0.9, target_sentence_lid > 0.9, "
            "target_glotlid_lang contains 'mos_Latn'). "
            "Other rows are kept with comet_qe_en_mos=null."
        ),
    )

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
            apply_lid_filter=args.filter_lid,
        )
    elif args.command == "full":
        full_pipeline(
            hub_repo=args.hub_repo,
            min_laser=args.min_laser,
            comet_batch_size=args.batch_size,
            accelerator=args.accelerator,
            private=args.private,
            rows_slice=slice_args,
            apply_lid_filter=args.filter_lid,
        )


if __name__ == "__main__":
    main()
