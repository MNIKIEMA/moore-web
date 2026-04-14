"""Build a combined French–Mooré dataset.

Sources
-------
Moore-web collection (local JSONL files under ``--data-dir``):

  File                              Source tag      Rows   Eval-eligible
  ────────────────────────────────  ──────────────  ─────  ─────────────
  lexicon.jsonl                     lexicon         4 249  yes
  lexicon_entries.jsonl             lexicon_entries 19793  no  (dict entries)
  conseils_ministres_aligned.jsonl  conseils        7 596  yes
  raamde_aligned.jsonl              news            3 915  yes
  sida_aligned.jsonl                sida              216  yes
  sida-facilitateur_aligned.jsonl   kade              674  yes
  digital-terms.jsonl               digital-terms    ~300  no  (glossary terms)
  digital-defs.jsonl                digital-defs     ~300  yes

mafand-fr-mos (``--mafand-repo``, default: ``madoss/mafand-fr-mos``):
  The existing train / validation / test splits are used directly.

Output splits
-------------
  train  =  local train portion  +  mafand train
  dev    =  local dev portion    +  mafand validation
  test   =  local test portion   +  mafand test

The local dev/test are built by stratified sampling over eval-eligible
sources (use ``--train-only-sources`` to customise which sources stay
train-only).  Remaining local rows go to train.

Output schema:  french | moore | source | laser_score | comet_qe | len_ratio

Usage
-----
    # Write JSONL files to ./fr_mos_combined/
    python build_fr_mos_dataset.py --output-dir fr_mos_combined

    # Custom split sizes
    python build_fr_mos_dataset.py --output-dir fr_mos_combined \\
        --dev-size 600 --test-size 600

    # Also push to the Hub
    python build_fr_mos_dataset.py --output-dir fr_mos_combined \\
        --push-to-hub madoss/fr-mos-combined

    # Skip the mafand download (local data only)
    python build_fr_mos_dataset.py --output-dir fr_mos_combined \\
        --no-mafand
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Local file registry
# ---------------------------------------------------------------------------

# (filename, source_tag)  — order matters for reproducibility
_LOCAL_FILES: list[tuple[str, str]] = [
    ("lexicon.jsonl", "lexicon"),
    ("lexicon_entries.jsonl", "lexicon_entries"),
    ("conseils_ministres_aligned.jsonl", "conseils"),
    ("raamde_aligned.jsonl", "news"),
    ("sida_aligned.jsonl", "sida"),
    ("sida-facilitateur_aligned.jsonl", "kade"),
    ("digital-terms.jsonl", "digital-terms"),
    ("digital-defs.jsonl", "digital-defs"),
]

# Sources excluded from dev/test by default (raw dictionary entries or
# short keyword pairs not representative of sentence-level translation).
_DEFAULT_TRAIN_ONLY: tuple[str, ...] = ("lexicon_entries", "digital-terms")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path, source_override: str | None = None) -> list[dict]:
    """Read a JSONL file, keeping french/moore/source/laser_score/comet_qe/len_ratio."""
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            fr = obj.get("french", "").strip()
            mo = obj.get("moore", "").strip()
            if not fr or not mo:
                continue
            src = source_override if source_override is not None else obj.get("source", "unknown")
            rows.append({
                "french": fr,
                "moore": mo,
                "source": src,
                "laser_score": obj.get("laser_score"),
                "comet_qe": obj.get("comet_qe"),
                "len_ratio": obj.get("len_ratio"),
            })
    return rows


def _write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  wrote {len(rows):>6,} rows → {path}")


def _print_source_breakdown(rows: list[dict], label: str) -> None:
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        counts[r["source"]] += 1
    parts = "  ".join(f"{s}={n:,}" for s, n in sorted(counts.items()))
    print(f"  {label}: {len(rows):,} rows  [{parts}]")


# ---------------------------------------------------------------------------
# Quality filter
# ---------------------------------------------------------------------------

_FILTERS: dict[str, tuple[str, float]] = {
    "len_ratio":   (">",  0.1),
    "comet_qe":    (">=", 0.35),
    "laser_score": (">=", 0.5),
}


def _passes_filter(row: dict) -> bool:
    """Return True if the row passes all quality thresholds.

    A threshold is skipped when the value is None (field absent for that source).
    """
    for field, (op, threshold) in _FILTERS.items():
        val = row.get(field)
        if val is None:
            continue
        if op == ">=" and val < threshold:
            return False
        if op == ">" and val <= threshold:
            return False
    return True


def _apply_quality_filter(rows: list[dict]) -> list[dict]:
    kept = [r for r in rows if _passes_filter(r)]
    dropped = len(rows) - len(kept)
    if dropped:
        by_source: dict[str, int] = defaultdict(int)
        for r in rows:
            if not _passes_filter(r):
                by_source[r["source"]] += 1
        parts = "  ".join(f"{s}={n:,}" for s, n in sorted(by_source.items()))
        print(f"  quality filter: dropped {dropped:,} rows  [{parts}]")
    return kept


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------


def _stratified_split(
    rows: list[dict],
    dev_size: int,
    test_size: int,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split *rows* into (train, dev, test) stratified by source.

    Each source contributes proportionally to dev and test.  Sources
    with fewer than 3 rows are kept entirely in train.
    """
    rng = random.Random(seed)

    by_source: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)

    total = len(rows)
    dev_frac = dev_size / total
    test_frac = test_size / total

    train: list[dict] = []
    dev: list[dict] = []
    test: list[dict] = []

    for src, src_rows in sorted(by_source.items()):
        rng.shuffle(src_rows)
        n = len(src_rows)
        if n < 3:
            train.extend(src_rows)
            continue
        n_dev = max(1, round(n * dev_frac))
        n_test = max(1, round(n * test_frac))
        # Guard: never take more than half of a tiny source for eval
        n_dev = min(n_dev, n // 3)
        n_test = min(n_test, n // 3)
        dev.extend(src_rows[:n_dev])
        test.extend(src_rows[n_dev : n_dev + n_test])
        train.extend(src_rows[n_dev + n_test :])

    rng.shuffle(train)
    rng.shuffle(dev)
    rng.shuffle(test)
    return train, dev, test


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build(
    data_dir: Path,
    mafand_repo: str | None,
    output_dir: Path,
    dev_size: int,
    test_size: int,
    train_only_sources: tuple[str, ...],
    push_to_hub: str | None,
    hub_private: bool,
    seed: int,
) -> None:
    # ---- 1. Load local files ------------------------------------------------
    print("Loading local moore-web files …")
    local_all: list[dict] = []
    for filename, source_tag in _LOCAL_FILES:
        path = data_dir / filename
        if not path.exists():
            print(f"  [skip] {filename} not found")
            continue
        rows = _load_jsonl(path, source_override=source_tag)
        print(f"  {filename}: {len(rows):,} rows")
        local_all.extend(rows)

    # ---- 2. Deduplicate on (french, moore) across all local files -----------
    seen: set[tuple[str, str]] = set()
    deduped: list[dict] = []
    for r in local_all:
        key = (r["french"], r["moore"])
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    n_dropped = len(local_all) - len(deduped)
    if n_dropped:
        print(f"\n  dedup: dropped {n_dropped:,} duplicate (fr, mos) pairs "
              f"→ {len(deduped):,} unique rows")
    local_all = deduped

    # ---- Quality filter -----------------------------------------------------
    print("\nApplying quality filters …")
    local_all = _apply_quality_filter(local_all)

    # Separate train-only rows (dictionary entries etc.)
    train_only = [r for r in local_all if r["source"] in train_only_sources]
    splittable = [r for r in local_all if r["source"] not in train_only_sources]

    print(f"\nEval-eligible rows: {len(splittable):,}  "
          f"(train-only: {len(train_only):,})")

    # ---- 2. Stratified split of splittable rows ----------------------------
    print(f"\nBuilding stratified split  dev={dev_size}  test={test_size}  seed={seed} …")
    local_train, local_dev, local_test = _stratified_split(
        splittable, dev_size, test_size, seed
    )
    local_train = train_only + local_train  # re-attach train-only rows

    print("  Local split:")
    _print_source_breakdown(local_train, "train")
    _print_source_breakdown(local_dev,   "dev  ")
    _print_source_breakdown(local_test,  "test ")

    # ---- 3. mafand splits --------------------------------------------------
    mafand_train: list[dict] = []
    mafand_dev: list[dict] = []
    mafand_test: list[dict] = []

    if mafand_repo:
        print(f"\nLoading {mafand_repo} …")
        from datasets import load_dataset  # lazy import

        ds = load_dataset(mafand_repo)
        for split_name, target in [
            ("train", mafand_train),
            ("validation", mafand_dev),
            ("test", mafand_test),
        ]:
            if split_name not in ds:
                print(f"  [skip] split '{split_name}' not found in {mafand_repo}")
                continue
            for row in ds[split_name]:
                fr = (row.get("french") or "").strip()
                mo = (row.get("moore") or "").strip()
                src = row.get("source") or "mafand"
                if fr and mo:
                    target.append({
                        "french": fr,
                        "moore": mo,
                        "source": src,
                        "laser_score": row.get("laser_score"),
                        "comet_qe": row.get("comet_qe"),
                        "len_ratio": row.get("len_ratio"),
                    })
            print(f"  {split_name}: {len(target):,} rows")

    # ---- 4. Merge ----------------------------------------------------------
    final_train = local_train + mafand_train
    final_dev = local_dev + mafand_dev
    final_test = local_test + mafand_test

    random.Random(seed).shuffle(final_train)

    print("\nFinal dataset:")
    _print_source_breakdown(final_train, "train")
    _print_source_breakdown(final_dev,   "dev  ")
    _print_source_breakdown(final_test,  "test ")
    total = len(final_train) + len(final_dev) + len(final_test)
    print(f"  total: {total:,}")

    # ---- 5. Write JSONL ----------------------------------------------------
    if output_dir:
        print(f"\nWriting to {output_dir}/ …")
        _write_jsonl(final_train, output_dir / "train.jsonl")
        _write_jsonl(final_dev,   output_dir / "dev.jsonl")
        _write_jsonl(final_test,  output_dir / "test.jsonl")

    # ---- 6. Push to Hub ----------------------------------------------------
    if push_to_hub:
        from datasets import Dataset, DatasetDict

        print(f"\nPushing to {push_to_hub} …")
        _drop_cols = {"quality_warnings"}
        def _strip(rows: list[dict]) -> list[dict]:
            return [{k: v for k, v in r.items() if k not in _drop_cols} for r in rows]
        dataset_dict = DatasetDict({
            "train":      Dataset.from_list(_strip(final_train)),
            "validation": Dataset.from_list(_strip(final_dev)),
            "test":       Dataset.from_list(_strip(final_test)),
        })
        dataset_dict.push_to_hub(push_to_hub, private=hub_private)
        print(f"Done. https://huggingface.co/datasets/{push_to_hub}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a combined French–Mooré dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-dir",
        default="final_data_hf",
        metavar="DIR",
        help="Directory containing local moore-web JSONL files (default: %(default)s).",
    )
    parser.add_argument(
        "--mafand-repo",
        default="madoss/mafand-fr-mos",
        metavar="REPO_ID",
        help="HF Hub repo for mafand splits (default: %(default)s). "
             "Pass empty string or use --no-mafand to skip.",
    )
    parser.add_argument(
        "--no-mafand",
        action="store_true",
        help="Skip loading the mafand dataset (local data only).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="fr_mos_combined",
        metavar="DIR",
        help="Directory to write train/dev/test JSONL files (default: %(default)s).",
    )
    parser.add_argument(
        "--push-to-hub",
        default=None,
        metavar="REPO_ID",
        help="HF Hub repo to push the final DatasetDict to.",
    )
    parser.add_argument(
        "--hub-private",
        action="store_true",
        help="Make the pushed Hub dataset private.",
    )
    parser.add_argument(
        "--dev-size",
        type=int,
        default=500,
        help="Target number of local rows allocated to dev (default: %(default)s).",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=500,
        help="Target number of local rows allocated to test (default: %(default)s).",
    )
    parser.add_argument(
        "--train-only-sources",
        nargs="+",
        default=list(_DEFAULT_TRAIN_ONLY),
        metavar="SOURCE",
        help="Source tags that must stay in train only "
             "(default: %(default)s).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: %(default)s).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build(
        data_dir=Path(args.data_dir),
        mafand_repo=None if args.no_mafand else args.mafand_repo,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        dev_size=args.dev_size,
        test_size=args.test_size,
        train_only_sources=tuple(args.train_only_sources),
        push_to_hub=args.push_to_hub,
        hub_private=args.hub_private,
        seed=args.seed,
    )
