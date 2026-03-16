"""Push the moore-web aligned corpora to the Hugging Face Hub.

Dataset layout
--------------
One dataset, one config (subset) per source:

    load_dataset("MNIKIEMA/moore-fr-mo")                         # all splits merged
    load_dataset("MNIKIEMA/moore-fr-mo", name="conseils")        # conseils only
    load_dataset("MNIKIEMA/moore-fr-mo", name="raamde")
    load_dataset("MNIKIEMA/moore-fr-mo", name="sida")
    load_dataset("MNIKIEMA/moore-fr-mo", name="sida-facilitateur")
    load_dataset("MNIKIEMA/moore-fr-mo", name="lexicon")

Each record:
    {
      "french":  "...",
      "moore":   "...",
      "score":   0.95,         # LASER cosine similarity (1.0 for lexicon)
      "source":  "sida",
      "english": "..."         # only in lexicon, else absent
    }

Usage
-----
    # dry run (no upload)
    python push_to_hf.py --dry-run

    # push everything
    python push_to_hf.py --repo MNIKIEMA/moore-fr-mo

    # push a single source
    python push_to_hf.py --repo MNIKIEMA/moore-fr-mo --only sida
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SOURCES: dict[str, str] = {
    "conseils": "final_data_hf/conseils_ministres_aligned.jsonl",
    "raamde": "final_data_hf/raamde_aligned.jsonl",
    "sida": "final_data_hf/sida_aligned.jsonl",
    "sida-facilitateur": "final_data_hf/sida-facilitateur_aligned.jsonl",
    "lexicon": "final_data_hf/lexicon.jsonl",
    "lexicon-entries": "final_data_hf/lexicon_entries.jsonl",
}

# Sources that carry English translations
HAS_ENGLISH = {"lexicon", "lexicon-entries"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_records(source: str, path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        print(f"  SKIP {source}: {p} not found.")
        return []

    # JSONL — records already have the right shape (from dict_to_jsonl.py)
    if p.suffix == ".jsonl":
        records = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        for rec in records:
            rec.setdefault("source", source)
        print(f"  {source}: {len(records)} pairs loaded from {p}")
        return records

    # JSON — AlignedCorpus format (from moore-web e2e pipeline)
    data = json.loads(p.read_text(encoding="utf-8"))
    french = data.get("french", [])
    moore = data.get("moore", [])
    scores = data.get("scores", [1.0] * len(french))
    english = data.get("english", []) if source in HAS_ENGLISH else []

    records = []
    for i, (fr, mo, sc) in enumerate(zip(french, moore, scores)):
        rec: dict = {"french": fr, "moore": mo, "score": float(sc), "source": source}
        if english and i < len(english) and english[i]:
            rec["english"] = english[i]
        records.append(rec)

    print(f"  {source}: {len(records)} pairs loaded from {p}")
    return records


def make_dataset(records: list[dict]):
    from datasets import Dataset

    return Dataset.from_list(records)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Push moore-web corpora to HF Hub.")
    parser.add_argument(
        "--repo",
        default="madoss/moore-fr-mo",
        help="HF Hub repo id (default: madoss/moore-fr-mo)",
    )
    parser.add_argument(
        "--only",
        metavar="SOURCE",
        help="Push only this source (e.g. --only sida).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and print stats without uploading.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository.",
    )
    args = parser.parse_args()

    sources = {k: v for k, v in SOURCES.items() if args.only is None or k == args.only}

    if not args.dry_run:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(
            repo_id=args.repo,
            repo_type="dataset",
            exist_ok=True,
            private=args.private,
        )
        print(f"Repository: https://huggingface.co/datasets/{args.repo}\n")

    all_records: list[dict] = []

    for source, path in sources.items():
        print(f"Loading {source}...")
        records = load_records(source, path)
        if not records:
            continue

        all_records.extend(records)

        if args.dry_run:
            continue

        ds = make_dataset(records)
        ds.push_to_hub(
            repo_id=args.repo,
            config_name=source,
            split="train",
            commit_message=f"Add {source} subset ({len(records)} pairs)",
        )
        print(f"  Pushed {source} subset.\n")

    if args.dry_run:
        print(f"\nDry run — {len(all_records)} total pairs across {len(sources)} sources.")
        print("Run without --dry-run to upload.")
        return

    if not args.only and all_records:
        print(f"\nPushing merged 'all' config ({len(all_records)} pairs)...")
        ds_all = make_dataset(all_records)
        ds_all.push_to_hub(
            repo_id=args.repo,
            config_name="all",
            split="train",
            commit_message=f"Add merged 'all' config ({len(all_records)} pairs)",
        )
        print("  Pushed 'all' subset.")

    print(f"\nDone. Dataset at https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
