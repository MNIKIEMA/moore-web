"""Build a combined French–Mooré dataset.

Sources
-------
Local sources are declared in ``fr_mos_sources.toml`` (``--sources``): one
``[[sources]]`` entry per file with its dataset tag, per-source quality
thresholds, ``skip`` rules and ``train_only`` flag. Entries read either

- ``file``: a JSONL under ``data_dir`` (``final_data_hf``) -- parsed and
  automatically aligned outputs, expert translations;
- ``reviewed``: a file from the reviewed-export HF dataset repo, pinned by
  ``[reviewed].revision`` -- accepted review-app units, exported with
  ``moore-web export-reviewed --push``. ``--reviewed-dir`` reads a local
  export instead (e.g. before pushing it).

The build never opens the review DB, so the same sources file + revision
gives the same dataset.

mafand-fr-mos (``--mafand-repo``, default: ``madoss/mafand-fr-mos``):
  The existing train / validation / test splits are used directly.

Output splits
-------------
  train  =  local train portion  +  mafand train
  dev    =  local dev portion    +  mafand validation
  test   =  local test portion   +  mafand test

The local dev/test are built by stratified sampling over eval-eligible
sources (``train_only`` entries, or ``--train-only-sources``, stay in
train).  Remaining local rows go to train.

Output schema:  id | french | moore | source | original_lang | doc_id | reviewed
                | laser_score | comet_qe | len_ratio
(see docs/dataset-splits.md for what the metadata columns mean)

Usage
-----
    # Write JSONL files to ./fr_mos_combined/
    python build_fr_mos_dataset.py --output-dir fr_mos_combined

    # Custom split sizes
    python build_fr_mos_dataset.py --output-dir fr_mos_combined \\
        --dev-size 1500 --test-size 1500

    # Use a local reviewed export instead of the pinned Hub revision
    moore-web export-reviewed -o data/reviewed
    python build_fr_mos_dataset.py --reviewed-dir data/reviewed --no-mafand

    # Also push to the Hub
    python build_fr_mos_dataset.py --output-dir fr_mos_combined \\
        --push-to-hub madoss/fr-mos-combined

    # Skip the mafand download (local data only)
    python build_fr_mos_dataset.py --output-dir fr_mos_combined \\
        --no-mafand
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


# ---------------------------------------------------------------------------
# Sources file
# ---------------------------------------------------------------------------

Filters = dict[str, tuple[str, float]]


def _parse_filters(table: dict[str, str]) -> Filters:
    """``{"laser_score": ">= 0.5"}`` → ``{"laser_score": (">=", 0.5)}``."""
    filters: Filters = {}
    for field, expr in table.items():
        op, _, value = expr.strip().partition(" ")
        if op not in (">=", ">"):
            raise ValueError(f"Unsupported filter {field} = {expr!r}; use '>= x' or '> x'")
        filters[field] = (op, float(value))
    return filters


def load_sources(path: Path) -> dict:
    """Read the sources file; each entry gets its merged ``filters``."""
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    defaults = _parse_filters(config.get("filters", {}))
    for entry in config["sources"]:
        if ("file" in entry) == ("reviewed" in entry):
            raise ValueError(f"Source {entry.get('tag')!r} needs exactly one of 'file' or 'reviewed'")
        entry["filters"] = defaults | _parse_filters(entry.get("filters", {}))
    return config


def _reviewed_dir(config: dict, override: Path | None) -> Path | None:
    """Local directory of the reviewed export: the override, or the pinned Hub revision."""
    if override:
        return override
    reviewed = config.get("reviewed", {})
    if not reviewed.get("revision"):
        return None
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(reviewed["repo"], repo_type="dataset", revision=reviewed["revision"]))


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _text_pair(obj: dict) -> tuple[str, str]:
    """(french, moore) of a flat or long-format row; ``("", "")`` for other pairs.

    Long rows put the original language first, so a Mooré-original row has
    Mooré in ``source_text``.
    """
    if "source_text" not in obj:
        return (obj.get("french") or "").strip(), (obj.get("moore") or "").strip()
    src, tgt = (obj.get("source_text") or "").strip(), (obj.get("target_text") or "").strip()
    langs = (obj.get("src_lang", "fra"), obj.get("tgt_lang", "mos"))
    if langs == ("fra", "mos"):
        return src, tgt
    if langs == ("mos", "fra"):
        return tgt, src
    return "", ""  # e.g. the mos-eng rows of a trilingual source


def _row_id(obj: dict, tag: str, fr: str, mo: str) -> str:
    """Upstream id, else ``{source}-{unit}-{line}`` (review export), else a text hash."""
    if obj.get("id"):
        return str(obj["id"])
    if obj.get("unit") is not None and obj.get("line") is not None:
        return f"{obj.get('source', tag)}-{obj['unit']}-{obj['line']}"
    text = fr + "\t" + mo
    return f"{tag}-{hashlib.sha1(text.encode('utf-8')).hexdigest()[:12]}"


def _load_jsonl(
    path: Path,
    source_override: str | None = None,
    skip: dict | None = None,
    original_lang: str | None = None,
    reviewed: bool = False,
) -> list[dict]:
    """Read a JSONL file into dataset rows (see the output schema above).

    Reads the flat ``french``/``moore`` and the long ``source_text``/
    ``target_text`` schemas. ``original_lang`` is taken from the row when it
    records one (``is_source_orig``), else from the argument. Rows where any
    ``skip`` field equals its value are dropped.
    """
    skip = skip or {}
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if any(obj.get(field) == value for field, value in skip.items()):
                continue
            fr, mo = _text_pair(obj)
            if not fr or not mo:
                continue
            src = source_override if source_override is not None else obj.get("source", "unknown")
            rows.append(
                {
                    "id": _row_id(obj, src, fr, mo),
                    "french": fr,
                    "moore": mo,
                    "source": src,
                    "original_lang": obj["src_lang"] if obj.get("is_source_orig") is True else original_lang,
                    "doc_id": obj.get("doc_id") or obj.get("unit"),
                    "reviewed": reviewed,
                    "laser_score": obj.get("laser_score"),
                    "comet_qe": obj.get("comet_qe"),
                    "len_ratio": obj.get("len_ratio"),
                }
            )
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


def _passes_filter(row: dict, filters: Filters) -> bool:
    """Return True if the row passes all quality thresholds.

    A threshold is skipped when the value is None (field absent for that source),
    so human-reviewed rows, which carry no scores, always pass.
    """
    for field, (op, threshold) in filters.items():
        val = row.get(field)
        if val is None:
            continue
        if op == ">=" and val < threshold:
            return False
        if op == ">" and val <= threshold:
            return False
    return True


def load_local(config: dict, data_dir: Path, reviewed_dir: Path | None) -> list[dict]:
    """Load, filter and deduplicate every source entry, in sources-file order.

    Filters apply per entry, since entries sharing a tag (reviewed vs
    automatic ``news``) need different thresholds. Dedup keeps the first
    copy of a (french, moore) pair, so earlier entries win.
    """
    rows: list[dict] = []
    for entry in config["sources"]:
        if "reviewed" in entry:
            if reviewed_dir is None:
                print(f"  [skip] {entry['reviewed']}: no reviewed export pinned ([reviewed].revision)")
                continue
            path = reviewed_dir / entry["reviewed"]
        else:
            path = data_dir / entry["file"]
        if not path.exists():
            print(f"  [skip] {path} not found")
            continue
        loaded = _load_jsonl(
            path,
            source_override=entry["tag"],
            skip=entry.get("skip"),
            original_lang=entry.get("original_lang"),
            # Review-app exports are human-checked; `human` marks other human data.
            reviewed="reviewed" in entry or bool(entry.get("human")),
        )
        kept = [r for r in loaded if _passes_filter(r, entry["filters"])]
        dropped = f"  (quality filter dropped {len(loaded) - len(kept):,})" if len(kept) < len(loaded) else ""
        print(f"  {path.name} → {entry['tag']}: {len(kept):,} rows{dropped}")
        rows.extend(kept)

    seen: set[tuple[str, str]] = set()
    deduped: list[dict] = []
    for r in rows:
        key = (r["french"], r["moore"])
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    if len(deduped) < len(rows):
        print(
            f"\n  dedup: dropped {len(rows) - len(deduped):,} duplicate (fr, mos) pairs → {len(deduped):,} unique rows"
        )
    return deduped


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
    sources: Path,
    reviewed_dir: Path | None,
    mafand_repo: str | None,
    output_dir: Path,
    dev_size: int,
    test_size: int,
    train_only_sources: tuple[str, ...] | None,
    push_to_hub: str | None,
    hub_private: bool,
    seed: int,
) -> None:
    # ---- 1. Load, filter and deduplicate local sources ----------------------
    config = load_sources(sources)
    data_dir = sources.parent / config.get("data_dir", ".")
    reviewed_dir = _reviewed_dir(config, reviewed_dir)
    print(f"Loading sources from {sources} (reviewed export: {reviewed_dir}) …")
    local_all = load_local(config, data_dir, reviewed_dir)

    if train_only_sources is None:
        train_only_sources = tuple(sorted({e["tag"] for e in config["sources"] if e.get("train_only")}))

    # Separate train-only rows (dictionary entries etc.)
    train_only = [r for r in local_all if r["source"] in train_only_sources]
    splittable = [r for r in local_all if r["source"] not in train_only_sources]

    print(f"\nEval-eligible rows: {len(splittable):,}  (train-only: {len(train_only):,})")

    # ---- 2. Stratified split of splittable rows ----------------------------
    print(f"\nBuilding stratified split  dev={dev_size}  test={test_size}  seed={seed} …")
    local_train, local_dev, local_test = _stratified_split(splittable, dev_size, test_size, seed)
    local_train = train_only + local_train  # re-attach train-only rows

    print("  Local split:")
    _print_source_breakdown(local_train, "train")
    _print_source_breakdown(local_dev, "dev  ")
    _print_source_breakdown(local_test, "test ")

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
                    target.append(
                        {
                            "id": _row_id(row, src, fr, mo),
                            "french": fr,
                            "moore": mo,
                            "source": src,
                            # Professionally translated French news: no media
                            # publish original articles in Mooré.
                            "original_lang": "fra",
                            "doc_id": None,
                            "reviewed": True,
                            "laser_score": row.get("laser_score"),
                            "comet_qe": row.get("comet_qe"),
                            "len_ratio": row.get("len_ratio"),
                        }
                    )
            print(f"  {split_name}: {len(target):,} rows")

    # ---- 4. Merge ----------------------------------------------------------
    final_train = local_train + mafand_train
    final_dev = local_dev + mafand_dev
    final_test = local_test + mafand_test

    random.Random(seed).shuffle(final_train)

    id_counts = Counter(r["id"] for r in final_train + final_dev + final_test)
    dupes = sorted(i for i, n in id_counts.items() if n > 1)
    if dupes:
        raise ValueError(f"{len(dupes)} row ids are not unique, e.g. {dupes[:5]}")

    print("\nFinal dataset:")
    _print_source_breakdown(final_train, "train")
    _print_source_breakdown(final_dev, "dev  ")
    _print_source_breakdown(final_test, "test ")
    total = len(final_train) + len(final_dev) + len(final_test)
    print(f"  total: {total:,}")

    # ---- 5. Write JSONL ----------------------------------------------------
    if output_dir:
        print(f"\nWriting to {output_dir}/ …")
        _write_jsonl(final_train, output_dir / "train.jsonl")
        _write_jsonl(final_dev, output_dir / "dev.jsonl")
        _write_jsonl(final_test, output_dir / "test.jsonl")

    # ---- 6. Push to Hub ----------------------------------------------------
    if push_to_hub:
        from datasets import Dataset, DatasetDict

        print(f"\nPushing to {push_to_hub} …")
        _drop_cols = {"quality_warnings"}

        def _strip(rows: list[dict]) -> list[dict]:
            return [{k: v for k, v in r.items() if k not in _drop_cols} for r in rows]

        dataset_dict = DatasetDict(
            {
                "train": Dataset.from_list(_strip(final_train)),
                "validation": Dataset.from_list(_strip(final_dev)),
                "test": Dataset.from_list(_strip(final_test)),
            }
        )
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
        "--sources",
        default=str(Path(__file__).resolve().parent / "fr_mos_sources.toml"),
        metavar="TOML",
        help="Sources file; its data_dir is relative to it (default: fr_mos_sources.toml).",
    )
    parser.add_argument(
        "--reviewed-dir",
        default=None,
        metavar="DIR",
        help="Local reviewed export to use instead of the pinned Hub revision.",
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
        default=1000,
        help="Target number of local rows allocated to dev (default: %(default)s).",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=1000,
        help="Target number of local rows allocated to test (default: %(default)s).",
    )
    parser.add_argument(
        "--train-only-sources",
        nargs="+",
        default=None,
        metavar="SOURCE",
        help="Source tags that must stay in train only (default: train_only entries in the sources file).",
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
        sources=Path(args.sources),
        reviewed_dir=Path(args.reviewed_dir) if args.reviewed_dir else None,
        mafand_repo=None if args.no_mafand else args.mafand_repo,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        dev_size=args.dev_size,
        test_size=args.test_size,
        train_only_sources=tuple(args.train_only_sources) if args.train_only_sources else None,
        push_to_hub=args.push_to_hub,
        hub_private=args.hub_private,
        seed=args.seed,
    )
