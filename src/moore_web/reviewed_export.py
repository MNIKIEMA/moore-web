"""Snapshot accepted review-app units to per-source JSONL files.

The review DB is a workspace; datasets are built from these snapshots, which
are pushed to a private HF dataset repo so each export is a commit that a
build can pin (see ``fr_mos_sources.toml``).

Output directory layout::

    <source>.jsonl   one row per kept line pair of an accepted unit
    _export.json     row/unit counts per source and the latest review time

Rows are ordered by (source, unit position, line), and ``_export.json`` holds
no wall-clock time, so exporting an unchanged DB gives byte-identical files.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from moore_web.review_store import connect, kept_lines

DEFAULT_REPO = "madoss/moore-web-reviewed"


def export_reviewed(db_path: Path, output_dir: Path) -> dict:
    """Write accepted units to ``output_dir``; return the ``_export.json`` summary.

    Units whose kept lines don't pair up are skipped and counted. ``*.jsonl``
    files in ``output_dir`` for sources with no accepted rows are removed, so
    the directory mirrors the DB.
    """
    rows_by_source: dict[str, list[dict]] = defaultdict(list)
    units: dict[str, int] = defaultdict(int)
    skipped: dict[str, int] = defaultdict(int)
    latest = None
    with connect(db_path) as db:
        cursor = db.execute(
            """SELECT u.source, u.unit_uid, r.fra, r.mos, r.rejected_fra, r.rejected_mos,
                      r.reviewed_by, r.updated_at
            FROM units u JOIN reviews r ON r.unit_id = u.id
            WHERE r.reviewed = 1
            ORDER BY u.source, u.position, u.id"""
        )
        for row in cursor:
            fra = kept_lines(json.loads(row["fra"]), json.loads(row["rejected_fra"]))
            mos = kept_lines(json.loads(row["mos"]), json.loads(row["rejected_mos"]))
            if len(fra) != len(mos):
                skipped[row["source"]] += 1
                continue
            units[row["source"]] += 1
            if row["updated_at"] and (latest is None or row["updated_at"] > latest):
                latest = row["updated_at"]
            for line, (fr, mo) in enumerate(zip(fra, mos)):
                if fr.strip() and mo.strip():
                    rows_by_source[row["source"]].append(
                        {
                            "french": fr.strip(),
                            "moore": mo.strip(),
                            "source": row["source"],
                            "unit": row["unit_uid"],
                            "line": line,
                            "reviewed_by": row["reviewed_by"],
                            "updated_at": row["updated_at"],
                        }
                    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("*.jsonl"):
        if stale.stem not in rows_by_source:
            stale.unlink()
    for source, rows in rows_by_source.items():
        (output_dir / f"{source}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
        )

    summary = {
        "latest_review_at": latest,
        "sources": {
            source: {
                "rows": len(rows_by_source.get(source, [])),
                "units": units[source],
                "skipped_units": skipped[source],
            }
            for source in sorted(set(units) | set(skipped))
        },
    }
    (output_dir / "_export.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def push_reviewed(output_dir: Path, repo_id: str = DEFAULT_REPO, message: str | None = None) -> str:
    """Upload ``output_dir`` to a private HF dataset repo as one commit.

    Creates the repo (private) on first push. Returns the commit sha to pin in
    ``fr_mos_sources.toml``.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=output_dir,
        allow_patterns=["*.jsonl", "_export.json"],
        delete_patterns=["*.jsonl"],
        commit_message=message or "Export reviewed units",
    )
    return commit.oid
