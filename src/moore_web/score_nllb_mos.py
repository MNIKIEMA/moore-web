"""Load allenai/nllb (eng_Latn-mos_Latn), add COMET-QE scores, upload to HF.

Pipeline
--------
1. Download the NLLB eng↔mos gzipped TSV from AllenAI's storage.
   (The dataset already contains an original NLLB ``laser_score``.)
2. Score every pair with ``McGill-NLP/ssa-comet-qe`` (reference-free QE).
3. Push the enriched dataset to HuggingFace Hub as ``nllb-mos``.

Usage
-----
    uv run python -m moore_web.score_nllb_mos
    uv run python -m moore_web.score_nllb_mos --hub-repo YOUR_USERNAME/nllb-mos
    uv run python -m moore_web.score_nllb_mos --min-laser 0.5 --batch-size 16
"""

from __future__ import annotations

import statistics


# ---------------------------------------------------------------------------
# COMET-QE scoring
# ---------------------------------------------------------------------------


def _comet_scores(
    eng_sentences: list[str],
    mos_sentences: list[str],
    batch_size: int = 8,
    gpus: int = 0,
) -> list[float]:
    """Return per-pair COMET-QE scores using McGill-NLP/ssa-comet-qe."""
    from comet import download_model, load_from_checkpoint

    print("Loading COMET-QE model (McGill-NLP/ssa-comet-qe)…")
    model_path = download_model("McGill-NLP/ssa-comet-qe")
    model = load_from_checkpoint(model_path)

    comet_data = [{"src": src, "mt": mt} for src, mt in zip(eng_sentences, mos_sentences)]

    print(f"Scoring {len(comet_data)} pairs with COMET-QE (batch_size={batch_size})…")
    output = model.predict(comet_data, batch_size=batch_size, gpus=gpus)

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


def score_and_upload(
    hub_repo: str = "nllb-mos",
    min_laser: float = 0.0,
    comet_batch_size: int = 8,
    gpus: int = 0,
    private: bool = False,
) -> None:
    """Load NLLB eng↔mos, add COMET-QE scores, push to HF Hub.

    The dataset already contains an original NLLB ``laser_score``.  We keep it
    and add ``comet_qe`` from ``McGill-NLP/ssa-comet-qe``.

    Args:
        hub_repo:          HuggingFace repo name, e.g. ``username/nllb-mos``.
        min_laser:         Drop pairs whose original NLLB laser_score is below this.
        comet_batch_size:  Batch size for COMET-QE inference.
        gpus:              Number of GPUs for COMET-QE (0 = CPU).
        private:           Whether to make the HF Hub dataset private.
    """
    from datasets import Dataset, DatasetDict

    rows = _load_nllb_tsv()

    if min_laser > 0.0:
        before = len(rows)
        rows = [r for r in rows if (r["laser_score"] or 0.0) >= min_laser]
        print(f"Dropped {before - len(rows)} pairs with laser_score < {min_laser}. Keeping {len(rows)}.")

    eng_sentences = [r["eng_Latn"] for r in rows]
    mos_sentences = [r["mos_Latn"] for r in rows]

    comet_scores = _comet_scores(eng_sentences, mos_sentences, batch_size=comet_batch_size, gpus=gpus)

    dataset = Dataset.from_dict(
        {
            **{col: [r[col] for r in rows] for col in _TSV_COLS},
            "comet_qe": comet_scores,
        }
    )

    dataset_dict = DatasetDict({"train": dataset})

    print(f"\nPushing dataset to HuggingFace Hub as '{hub_repo}'…")
    dataset_dict.push_to_hub(hub_repo, private=private)
    print(f"Done. Dataset available at https://huggingface.co/datasets/{hub_repo}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Score allenai/nllb (eng↔mos) with LASER + COMET-QE and upload to HF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--hub-repo",
        default="madoss/nllb-mos",
        help="HuggingFace Hub repo to push to, e.g. username/nllb-mos (default: %(default)s).",
    )
    parser.add_argument(
        "--min-laser",
        type=float,
        default=0.0,
        help="Drop pairs with LASER cosine similarity below this value before COMET-QE (default: keep all).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="COMET-QE inference batch size (default: %(default)s).",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=0,
        help="Number of GPUs for COMET-QE inference (default: 0 = CPU).",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the HuggingFace dataset private.",
    )
    args = parser.parse_args()

    score_and_upload(
        hub_repo=args.hub_repo,
        min_laser=args.min_laser,
        comet_batch_size=args.batch_size,
        gpus=args.gpus,
        private=args.private,
    )
