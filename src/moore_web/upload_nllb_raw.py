"""Download the AllenAI NLLB eng↔mos TSV and push it to HuggingFace Hub.

Usage
-----
    uv run python -m moore_web.upload_nllb_raw
    uv run python -m moore_web.upload_nllb_raw --hub-repo madoss/nllb-mos-raw
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

_NLLB_URL = "https://storage.googleapis.com/allennlp-data-bucket/nllb/eng_Latn-mos_Latn.gz"

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


def _load_nllb_tsv(url: str = _NLLB_URL) -> list[dict]:
    import gzip
    import io
    import urllib.request

    print(f"Downloading {url} …")
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

    print(f"Loaded {len(rows):,} pairs.")
    return rows


def upload(hub_repo: str = "madoss/nllb-mos-raw", private: bool = False) -> None:
    from datasets import Dataset, DatasetDict

    rows = _load_nllb_tsv()
    dataset_dict = DatasetDict({"train": Dataset.from_dict({col: [r[col] for r in rows] for col in _COLS})})

    print(f"\nPushing to '{hub_repo}' …")
    dataset_dict.push_to_hub(hub_repo, private=private)
    print(f"Done. https://huggingface.co/datasets/{hub_repo}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Upload AllenAI NLLB eng↔mos dataset to HuggingFace Hub.")
    parser.add_argument("--hub-repo", default="madoss/nllb-mos-raw", help="HF Hub repo (default: %(default)s).")
    parser.add_argument("--private", action="store_true", help="Make the dataset private.")
    args = parser.parse_args()

    upload(hub_repo=args.hub_repo, private=args.private)
