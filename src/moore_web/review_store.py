"""SQLite storage for the bilingual unit review app."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence


class ReviewConflict(Exception):
    """The accepted unit changed since this reviewer opened it."""


@contextmanager
def connect(path: Path) -> Iterator[sqlite3.Connection]:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=10)
    try:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout = 10000")
        db.execute("PRAGMA foreign_keys = ON")
        with db:
            yield db
    finally:
        db.close()


def initialize(path: Path) -> None:
    with connect(path) as db:
        db.execute("PRAGMA journal_mode = WAL")
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS units (
                id INTEGER PRIMARY KEY,
                source TEXT NOT NULL,
                unit_uid TEXT NOT NULL,
                position INTEGER NOT NULL,
                original_fra TEXT NOT NULL,
                original_mos TEXT NOT NULL,
                UNIQUE (source, unit_uid)
            );
            CREATE TABLE IF NOT EXISTS reviews (
                unit_id INTEGER PRIMARY KEY REFERENCES units(id),
                fra TEXT NOT NULL,
                mos TEXT NOT NULL,
                version INTEGER NOT NULL DEFAULT 0,
                reviewed INTEGER NOT NULL DEFAULT 0,
                reviewed_by TEXT,
                updated_at TEXT
            );
            CREATE TABLE IF NOT EXISTS drafts (
                unit_id INTEGER NOT NULL REFERENCES units(id),
                reviewer TEXT NOT NULL,
                fra_text TEXT NOT NULL,
                mos_text TEXT NOT NULL,
                base_version INTEGER NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (unit_id, reviewer)
            );
            CREATE INDEX IF NOT EXISTS idx_units_source_position
                ON units(source, position, id);
            CREATE INDEX IF NOT EXISTS idx_reviews_reviewed
                ON reviews(reviewed, unit_id);
            """
        )
        _migrate(db)


def _migrate(db: sqlite3.Connection) -> None:
    """Add columns introduced after the first release to an existing database."""
    for table in ("reviews", "drafts"):
        columns = {row[1] for row in db.execute(f"PRAGMA table_info({table})")}
        for column in ("rejected_fra", "rejected_mos"):
            if column not in columns:
                db.execute(f"ALTER TABLE {table} ADD COLUMN {column} TEXT NOT NULL DEFAULT '[]'")


def import_units(path: Path, input_paths: list[Path]) -> int:
    """Import new units once; preserve existing reviews on subsequent starts."""
    added = 0
    with connect(path) as db:
        for input_path in sorted(input_paths):
            source = input_path.stem.removesuffix("_units")
            with input_path.open(encoding="utf-8") as handle:
                for position, line in enumerate(handle):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if len(row) != 1:
                        raise ValueError(f"Expected one unit per line in {input_path}:{position + 1}")
                    uid, sides = next(iter(row.items()))
                    fra, mos = sides["fra"], sides["mos"]
                    if not isinstance(fra, list) or not isinstance(mos, list):
                        raise ValueError(f"Expected sentence lists in {input_path}:{position + 1}")
                    cursor = db.execute(
                        """INSERT OR IGNORE INTO units
                        (source, unit_uid, position, original_fra, original_mos)
                        VALUES (?, ?, ?, ?, ?)""",
                        (
                            source,
                            uid,
                            position,
                            json.dumps(fra, ensure_ascii=False),
                            json.dumps(mos, ensure_ascii=False),
                        ),
                    )
                    if not cursor.rowcount:
                        continue
                    added += 1
                    db.execute(
                        "INSERT INTO reviews (unit_id, fra, mos) VALUES (?, ?, ?)",
                        (
                            cursor.lastrowid,
                            json.dumps(fra, ensure_ascii=False),
                            json.dumps(mos, ensure_ascii=False),
                        ),
                    )
    return added


# A unit is mismatched when the lines left after rejection don't pair up.
_KEPT_MISMATCH = """(json_array_length(r.fra) - json_array_length(r.rejected_fra))
            != (json_array_length(r.mos) - json_array_length(r.rejected_mos))"""


def _unit_filters(status: str, source: str | None) -> tuple[str, list[Any]]:
    clauses = []
    params: list[Any] = []
    if source:
        clauses.append("u.source = ?")
        params.append(source)
    if status == "reviewed":
        clauses.append("r.reviewed = 1")
    elif status == "pending":
        clauses.append("r.reviewed = 0")
    elif status == "mismatched":
        clauses.append(f"r.reviewed = 0 AND {_KEPT_MISMATCH}")
    elif status != "all":
        raise ValueError(f"Unknown review status: {status}")
    return (f" WHERE {' AND '.join(clauses)}" if clauses else ""), params


_JSON_COLUMNS = ("fra", "mos", "rejected_fra", "rejected_mos", "original_fra", "original_mos")


def _decode(row: sqlite3.Row) -> dict[str, Any]:
    unit = dict(row)
    for column in _JSON_COLUMNS:
        if column in unit:
            unit[column] = json.loads(unit[column])
    return unit


def list_units(
    path: Path,
    *,
    limit: int | None = None,
    offset: int = 0,
    status: str = "all",
    source: str | None = None,
) -> list[dict[str, Any]]:
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if offset < 0:
        raise ValueError("offset cannot be negative")
    where, params = _unit_filters(status, source)
    pagination = ""
    if limit is not None:
        pagination = " LIMIT ? OFFSET ?"
        params.extend((limit, offset))
    with connect(path) as db:
        rows = db.execute(
            f"""SELECT u.id, u.source, u.unit_uid, u.position, r.fra, r.mos,
            r.rejected_fra, r.rejected_mos, r.version, r.reviewed, r.reviewed_by
            FROM units u JOIN reviews r ON r.unit_id = u.id
            {where}
            ORDER BY u.source, u.position, u.id{pagination}""",
            params,
        ).fetchall()
    return [_decode(row) for row in rows]


def count_units(path: Path, *, status: str = "all", source: str | None = None) -> int:
    where, params = _unit_filters(status, source)
    with connect(path) as db:
        row = db.execute(
            f"SELECT COUNT(*) FROM units u JOIN reviews r ON r.unit_id = u.id{where}", params
        ).fetchone()
    return int(row[0])


def review_summary(path: Path) -> dict[str, int]:
    with connect(path) as db:
        row = db.execute(
            f"""SELECT COUNT(*) AS total,
            COALESCE(SUM(r.reviewed = 1), 0) AS reviewed,
            COALESCE(SUM(r.reviewed = 0 AND {_KEPT_MISMATCH}), 0) AS mismatched
            FROM reviews r"""
        ).fetchone()
    return {key: int(row[key]) for key in ("total", "reviewed", "mismatched")}


def list_sources(path: Path) -> list[str]:
    with connect(path) as db:
        rows = db.execute("SELECT DISTINCT source FROM units ORDER BY source").fetchall()
    return [str(row[0]) for row in rows]


def get_unit(path: Path, unit_id: int) -> dict[str, Any]:
    with connect(path) as db:
        row = db.execute(
            """SELECT u.id, u.source, u.unit_uid, u.original_fra, u.original_mos,
            r.fra, r.mos, r.rejected_fra, r.rejected_mos, r.version,
            r.reviewed, r.reviewed_by FROM units u
            JOIN reviews r ON r.unit_id = u.id WHERE u.id = ?""",
            (unit_id,),
        ).fetchone()
    if row is None:
        raise KeyError(unit_id)
    return _decode(row)


def get_draft(path: Path, unit_id: int, reviewer: str) -> dict[str, Any] | None:
    with connect(path) as db:
        row = db.execute(
            """SELECT fra_text, mos_text, rejected_fra, rejected_mos, base_version
            FROM drafts WHERE unit_id = ? AND reviewer = ?""",
            (unit_id, reviewer),
        ).fetchone()
    return _decode(row) if row else None


def save_draft(
    path: Path,
    unit_id: int,
    reviewer: str,
    fra_text: str,
    mos_text: str,
    base_version: int,
    rejected_fra: Sequence[int] = (),
    rejected_mos: Sequence[int] = (),
) -> None:
    """Keep the editor's text as typed; rejected indices point into its lines, blank ones included."""
    with connect(path) as db:
        db.execute(
            """INSERT INTO drafts
            (unit_id, reviewer, fra_text, mos_text, rejected_fra, rejected_mos, base_version)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (unit_id, reviewer) DO UPDATE SET
            fra_text = excluded.fra_text, mos_text = excluded.mos_text,
            rejected_fra = excluded.rejected_fra, rejected_mos = excluded.rejected_mos,
            base_version = excluded.base_version, updated_at = CURRENT_TIMESTAMP""",
            (
                unit_id,
                reviewer,
                fra_text,
                mos_text,
                json.dumps(sorted(set(rejected_fra))),
                json.dumps(sorted(set(rejected_mos))),
                base_version,
            ),
        )


def split_lines(value: str) -> list[str]:
    return [line.strip() for line in value.splitlines() if line.strip()]


def clean_lines(value: str, rejected: Sequence[int] = ()) -> tuple[list[str], list[int]]:
    """Drop blank lines like `split_lines`, re-pointing rejected indices at the kept list.

    `rejected` indexes the editor's lines (``value.split("\\n")``), where a blank
    line still takes a slot; the result indexes the cleaned lines.
    """
    skip = set(rejected)
    lines: list[str] = []
    kept_rejected: list[int] = []
    for index, raw in enumerate(value.replace("\r\n", "\n").split("\n")):
        if not raw.strip():
            continue
        if index in skip:
            kept_rejected.append(len(lines))
        lines.append(raw.strip())
    return lines, kept_rejected


def kept_lines(lines: list[str], rejected: Sequence[int]) -> list[str]:
    """The lines that pair up and get exported: everything not rejected."""
    skip = set(rejected)
    return [line for index, line in enumerate(lines) if index not in skip]


def accept_review(
    path: Path,
    unit_id: int,
    reviewer: str,
    fra_text: str,
    mos_text: str,
    expected_version: int,
    rejected_fra: Sequence[int] = (),
    rejected_mos: Sequence[int] = (),
) -> int:
    """Accept the editor's text; `rejected_*` index its lines as typed (see `clean_lines`)."""
    fra, fra_rejected = clean_lines(fra_text, rejected_fra)
    mos, mos_rejected = clean_lines(mos_text, rejected_mos)
    n_fra, n_mos = len(fra) - len(fra_rejected), len(mos) - len(mos_rejected)
    if not n_fra or n_fra != n_mos:
        raise ValueError(
            "French and Mooré must keep the same nonzero number of lines "
            f"after rejections ({n_fra} / {n_mos})."
        )
    with connect(path) as db:
        cursor = db.execute(
            """UPDATE reviews SET fra = ?, mos = ?, rejected_fra = ?, rejected_mos = ?,
            version = version + 1, reviewed = 1, reviewed_by = ?, updated_at = CURRENT_TIMESTAMP
            WHERE unit_id = ? AND version = ?""",
            (
                json.dumps(fra, ensure_ascii=False),
                json.dumps(mos, ensure_ascii=False),
                json.dumps(fra_rejected),
                json.dumps(mos_rejected),
                reviewer,
                unit_id,
                expected_version,
            ),
        )
        if cursor.rowcount != 1:
            raise ReviewConflict(
                "This unit was saved by another reviewer. Your draft is retained; reopen the unit to compare."
            )
        db.execute("DELETE FROM drafts WHERE unit_id = ? AND reviewer = ?", (unit_id, reviewer))
    return expected_version + 1


def iter_pairs(path: Path, reviewed_only: bool = False) -> Iterator[dict[str, str]]:
    where = " WHERE r.reviewed = 1" if reviewed_only else ""
    with connect(path) as db:
        rows = db.execute(
            f"""SELECT u.source, u.unit_uid, r.fra, r.mos, r.rejected_fra, r.rejected_mos
            FROM units u JOIN reviews r ON r.unit_id = u.id{where}
            ORDER BY u.source, u.position, u.id"""
        )
        for row in rows:
            fra_sentences = kept_lines(json.loads(row["fra"]), json.loads(row["rejected_fra"]))
            mos_sentences = kept_lines(json.loads(row["mos"]), json.loads(row["rejected_mos"]))
            if len(fra_sentences) != len(mos_sentences):
                continue
            for fra, mos in zip(fra_sentences, mos_sentences, strict=True):
                yield {
                    "french": fra,
                    "moore": mos,
                    "source": row["source"],
                    "unit": row["unit_uid"],
                }


def export_pairs(path: Path, reviewed_only: bool = False) -> list[dict[str, str]]:
    """Return all export pairs; use iter_pairs for streaming exports."""
    return list(iter_pairs(path, reviewed_only))
