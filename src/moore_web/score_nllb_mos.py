"""Load allenai/nllb (eng_Latn-mos_Latn), add COMET-QE scores, upload to HF.

Pipeline
--------
1. Download the NLLB eng↔mos gzipped TSV from AllenAI's storage.
   (The dataset already contains an original NLLB ``laser_score``.)
2. Push the raw dataset to HuggingFace Hub (``download`` subcommand).
3. Score every pair with ``McGill-NLP/ssa-comet-qe`` and update the HF dataset
   (``score`` subcommand).

Usage
-----
    # Step 1 — download raw dataset and push to HF
    uv run python -m moore_web.score_nllb_mos download
    uv run python -m moore_web.score_nllb_mos download --hub-repo YOUR_USERNAME/nllb-mos-raw

    # Step 2 — add COMET-QE scores and push to HF
    uv run python -m moore_web.score_nllb_mos score
    uv run python -m moore_web.score_nllb_mos score --hub-repo YOUR_USERNAME/nllb-mos --min-laser 0.5

    # Legacy — run both steps in one go (original behaviour)
    uv run python -m moore_web.score_nllb_mos full
"""

from __future__ import annotations

from dotenv import load_dotenv

import statistics

load_dotenv()

# ---------------------------------------------------------------------------
# COMET-QE scoring
# ---------------------------------------------------------------------------


def _comet_scores(
    eng_sentences: list[str],
    mos_sentences: list[str],
    batch_size: int = 8,
    accelerator: str = "auto",
) -> list[float]:
    """Return per-pair COMET-QE scores using McGill-NLP/ssa-comet-qe."""
    from comet import download_model, load_from_checkpoint

    print("Loading COMET-QE model (McGill-NLP/ssa-comet-qe)…")
    model_path = download_model("McGill-NLP/ssa-comet-qe")
    model = load_from_checkpoint(model_path)

    comet_data = [{"src": src, "mt": mt} for src, mt in zip(eng_sentences, mos_sentences)]

    print(f"Scoring {len(comet_data)} pairs with COMET-QE (batch_size={batch_size}, accelerator={accelerator})…")
    output = model.predict(comet_data, batch_size=batch_size, accelerator=accelerator)

    scores = [float(s) for s in output.scores]
    print(
        f"COMET-QE scores — mean: {statistics.mean(scores):.3f}  "
        f"median: {statistics.median(scores):.3f}  "
        f"min: {min(scores):.3f}  max: {max(scores):.3f}"
    )
    return scores


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


_NLLB_URL = "https://storage.googleapis.com/allennlp-data-bucket/nllb/eng_Latn-mos_Latn.gz"

# Column order in the AllenAI TSV:
# src  tgt  laser_score  src_lid  tgt_lid  src_source  src_url  tgt_source  tgt_url
_TSV_COLS = [
    "eng_Latn",
    "mos_Latn",
    "laser_score",
    "source_sentence_lid",
    "target_sentence_lid",
    "source_sentence_source",
    "source_sentence_url",
    "target_sentence_source",
    "target_sentence_url",
]


def _load_nllb_tsv(url: str = _NLLB_URL) -> list[dict]:
    """Download and parse the NLLB eng↔mos gzipped TSV.

    Returns a list of dicts with keys matching ``_TSV_COLS``.
    The ``laser_score`` field is the original NLLB LASER score (float).
    """
    import gzip
    import io
    import urllib.request

    print(f"Downloading {url} …")
    with urllib.request.urlopen(url) as response:
        raw = response.read()

    rows = []
    with gzip.open(io.BytesIO(raw), "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            row: dict = {}
            for i, col in enumerate(_TSV_COLS):
                val: str | float | None = parts[i] if i < len(parts) else None
                if col in ("laser_score", "source_sentence_lid", "target_sentence_lid"):
                    try:
                        val = float(val) if val else None  # type: ignore[arg-type]
                    except ValueError:
                        val = None
                row[col] = val or None
            rows.append(row)

    print(f"Loaded {len(rows)} pairs from NLLB tsv.")
    return rows


def download_and_upload(
    hub_repo: str = "madoss/nllb-mos-raw",
    private: bool = False,
) -> None:
    """Download NLLB eng↔mos TSV and push the raw dataset to HF Hub.

    Args:
        hub_repo:  HuggingFace repo name, e.g. ``username/nllb-mos-raw``.
        private:   Whether to make the HF Hub dataset private.
    """
    from datasets import Dataset, DatasetDict

    rows = _load_nllb_tsv()

    dataset = Dataset.from_dict({col: [r[col] for r in rows] for col in _TSV_COLS})
    dataset_dict = DatasetDict({"train": dataset})

    print(f"\nPushing raw dataset to HuggingFace Hub as '{hub_repo}'…")
    dataset_dict.push_to_hub(hub_repo, private=private)
    print(f"Done. Dataset available at https://huggingface.co/datasets/{hub_repo}")


def score_and_upload(
    hub_repo: str = "madoss/nllb-mos",
    source_repo: str | None = None,
    min_laser: float = 0.0,
    comet_batch_size: int = 8,
    accelerator: str = "auto",
    private: bool = False,
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
    """
    from datasets import Dataset, DatasetDict, load_dataset

    if source_repo:
        print(f"Loading dataset from HuggingFace Hub: '{source_repo}'…")
        ds = load_dataset(source_repo, split="train")
        rows = [dict(zip(ds.column_names, vals)) for vals in zip(*[ds[col] for col in ds.column_names])]
    else:
        rows = _load_nllb_tsv()

    if min_laser > 0.0:
        before = len(rows)
        rows = [r for r in rows if (r["laser_score"] or 0.0) >= min_laser]
        print(f"Dropped {before - len(rows)} pairs with laser_score < {min_laser}. Keeping {len(rows)}.")

    eng_sentences = [r["eng_Latn"] for r in rows]
    mos_sentences = [r["mos_Latn"] for r in rows]

    comet_scores = _comet_scores(eng_sentences, mos_sentences, batch_size=comet_batch_size, accelerator=accelerator)

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
) -> None:
    """Download, score, and upload in one step (original behaviour)."""
    score_and_upload(
        hub_repo=hub_repo,
        source_repo=None,
        min_laser=min_laser,
        comet_batch_size=comet_batch_size,
        accelerator=accelerator,
        private=private,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download NLLB eng↔mos data and/or score it with COMET-QE.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- download subcommand -------------------------------------------------
    dl_parser = subparsers.add_parser(
        "download",
        help="Download raw NLLB TSV and push to HF Hub (no scoring).",
    )
    dl_parser.add_argument(
        "--hub-repo",
        default="madoss/nllb-mos-raw",
        help="HuggingFace Hub repo to push to (default: %(default)s).",
    )
    dl_parser.add_argument("--private", action="store_true", help="Make the dataset private.")

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
    sc_parser.add_argument("--private", action="store_true", help="Make the dataset private.")

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
    full_parser.add_argument("--private", action="store_true", help="Make the dataset private.")

    args = parser.parse_args()

    if args.command == "download":
        download_and_upload(hub_repo=args.hub_repo, private=args.private)
    elif args.command == "score":
        score_and_upload(
            hub_repo=args.hub_repo,
            source_repo=args.source_repo,
            min_laser=args.min_laser,
            comet_batch_size=args.batch_size,
            accelerator=args.accelerator,
            private=args.private,
        )
    elif args.command == "full":
        full_pipeline(
            hub_repo=args.hub_repo,
            min_laser=args.min_laser,
            comet_batch_size=args.batch_size,
            accelerator=args.accelerator,
            private=args.private,
        )
