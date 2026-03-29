"""Build and upload the raw eng↔mos dataset to HuggingFace Hub.

Two subsets are combined into a single DatasetDict and pushed as named splits:

* ``nllb``  — downloaded directly from AllenAI's storage (gzipped TSV).
              Contains full metadata (source, URL, LID scores).
* ``opus``  — local OPUS NLLB package (three parallel text files).
              Cleaner deduplication by OPUS; no URL/source metadata.

Both subsets share the same schema.  Columns absent in OPUS are set to None.
Laser scores are kept at their original scale (not normalised across subsets).

Usage
-----
    uv run python -m moore_web.upload_nllb_raw \\
        --opus-dir data/en-mos.txt \\
        --hub-repo madoss/nllb-mos-raw

    # push only the AllenAI NLLB split (skip OPUS)
    uv run python -m moore_web.upload_nllb_raw --hub-repo madoss/nllb-mos-raw
"""

from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Shared schema
# ---------------------------------------------------------------------------

_COLS = [
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

_NLLB_URL = "https://storage.googleapis.com/allennlp-data-bucket/nllb/eng_Latn-mos_Latn.gz"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_nllb_tsv(url: str = _NLLB_URL) -> list[dict]:
    """Download and parse the AllenAI NLLB gzipped TSV."""
    import gzip
    import io
    import urllib.request

    print(f"Downloading AllenAI NLLB TSV from {url} …")
    with urllib.request.urlopen(url) as response:
        raw = response.read()

    rows: list[dict] = []
    with gzip.open(io.BytesIO(raw), "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            row: dict = {}
            for i, col in enumerate(_COLS):
                val: str | float | None = parts[i] if i < len(parts) else None
                if col in ("laser_score", "source_sentence_lid", "target_sentence_lid"):
                    try:
                        val = float(val) if val else None  # type: ignore[arg-type]
                    except ValueError:
                        val = None
                row[col] = val or None
            rows.append(row)

    print(f"Loaded {len(rows):,} pairs from AllenAI NLLB TSV.")
    return rows


def _load_opus_files(opus_dir: str) -> list[dict]:
    """Load the OPUS NLLB parallel files (Moses format).

    Expected files inside ``opus_dir``:
        NLLB.en-mos.en     — one English sentence per line
        NLLB.en-mos.mos    — one Mooré sentence per line
        NLLB.en-mos.scores — one LASER score per line (raw, unnormalised)

    Metadata columns absent in the OPUS format are set to None.
    """
    en_path = os.path.join(opus_dir, "NLLB.en-mos.en")
    mos_path = os.path.join(opus_dir, "NLLB.en-mos.mos")
    scores_path = os.path.join(opus_dir, "NLLB.en-mos.scores")

    for p in (en_path, mos_path, scores_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Expected OPUS file not found: {p}")

    print(f"Loading OPUS NLLB files from {opus_dir} …")
    rows: list[dict] = []
    with open(en_path, encoding="utf-8") as f_en, \
         open(mos_path, encoding="utf-8") as f_mos, \
         open(scores_path, encoding="utf-8") as f_scores:
        for eng, mos, score_line in zip(f_en, f_mos, f_scores):
            eng = eng.rstrip("\n")
            mos = mos.rstrip("\n")
            try:
                laser_score: float | None = float(score_line.strip())
            except ValueError:
                laser_score = None
            if not eng or not mos:
                continue
            rows.append({
                "eng_Latn": eng,
                "mos_Latn": mos,
                "laser_score": laser_score,
                "source_sentence_lid": None,
                "target_sentence_lid": None,
                "source_sentence_source": None,
                "source_sentence_url": None,
                "target_sentence_source": None,
                "target_sentence_url": None,
            })

    print(f"Loaded {len(rows):,} pairs from OPUS NLLB files.")
    return rows


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


def upload(
    hub_repo: str = "madoss/nllb-mos-raw",
    opus_dir: str | None = None,
    private: bool = False,
) -> None:
    """Build and push the raw dataset to HF Hub.

    Args:
        hub_repo:  HuggingFace dataset repo to push to.
        opus_dir:  Path to the OPUS NLLB directory (contains NLLB.en-mos.*).
                   If None, only the ``nllb`` split is pushed.
        private:   Make the HF Hub dataset private.
    """
    from datasets import Dataset, DatasetDict

    splits: dict[str, Dataset] = {}

    nllb_rows = _load_nllb_tsv()
    splits["nllb"] = Dataset.from_dict({col: [r[col] for r in nllb_rows] for col in _COLS})

    if opus_dir:
        opus_rows = _load_opus_files(opus_dir)
        splits["opus"] = Dataset.from_dict({col: [r[col] for r in opus_rows] for col in _COLS})

    dataset_dict = DatasetDict(splits)

    split_names = ", ".join(f"'{k}' ({len(v):,} rows)" for k, v in splits.items())
    print(f"\nPushing {split_names} → '{hub_repo}' …")
    dataset_dict.push_to_hub(hub_repo, private=private)
    print(f"Done. https://huggingface.co/datasets/{hub_repo}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Upload raw NLLB eng↔mos data (AllenAI + OPUS) to HuggingFace Hub.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--hub-repo",
        default="madoss/nllb-mos-raw",
        help="HuggingFace Hub repo to push to (default: %(default)s).",
    )
    parser.add_argument(
        "--opus-dir",
        default=None,
        help="Path to the OPUS NLLB directory containing NLLB.en-mos.* files. "
             "If omitted, only the AllenAI 'nllb' split is pushed.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the HuggingFace dataset private.",
    )
    args = parser.parse_args()

    upload(hub_repo=args.hub_repo, opus_dir=args.opus_dir, private=args.private)
