#!/usr/bin/env python3
"""Migrate final_data_hf JSONL files: rename "score" → "laser_score".

Rewrites each file in-place (atomic via a temp file).  Files that already
use "laser_score" or have no "score" field are skipped.

Usage
-----
    uv run python migrate_score_field.py                      # default: final_data_hf/
    uv run python migrate_score_field.py --dir final_data_hf
    uv run python migrate_score_field.py --dry-run            # preview only
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


def migrate_file(path: Path, dry_run: bool) -> None:
    rows = []
    needs_migration = False

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if ("score" in row and "laser_score" not in row) or (
                "laser_score" in row and row["laser_score"] != round(row["laser_score"], 4)
            ):
                needs_migration = True
            rows.append(row)

    if not needs_migration:
        print(f"  [skip] {path.name} — already migrated or no 'score' field")
        return

    migrated = []
    for row in rows:
        if "score" in row and "laser_score" not in row:
            row["laser_score"] = round(row.pop("score"), 4)
        elif "laser_score" in row:
            row["laser_score"] = round(row["laser_score"], 4)
        migrated.append(row)

    if dry_run:
        print(f"  [dry-run] {path.name} — would rename 'score' → 'laser_score' in {len(migrated)} rows")
        return

    # Write atomically
    dir_ = path.parent
    with tempfile.NamedTemporaryFile("w", dir=dir_, delete=False, encoding="utf-8", suffix=".tmp") as tmp:
        for row in migrated:
            tmp.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp_path = tmp.name

    os.replace(tmp_path, path)
    print(f"  [done]  {path.name} — {len(migrated)} rows updated")


def main() -> None:
    parser = argparse.ArgumentParser(description='Rename "score" → "laser_score" in JSONL files.')
    parser.add_argument("--dir", "-d", default="final_data_hf", help="Directory to migrate (default: %(default)s).")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing.")
    args = parser.parse_args()

    target = Path(args.dir)
    if not target.is_dir():
        print(f"Error: {target} is not a directory.")
        raise SystemExit(1)

    jsonl_files = sorted(target.glob("*.jsonl"))
    if not jsonl_files:
        print(f"No .jsonl files found in {target}.")
        return

    print(f"Migrating {len(jsonl_files)} file(s) in {target}/{'  [DRY RUN]' if args.dry_run else ''}")
    for path in jsonl_files:
        migrate_file(path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
