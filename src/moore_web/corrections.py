"""One-off text corrections for specific rows, applied at dataset build time.

`normalize_moore` handles rules (a character that is always wrong); a typo that
needs judgement (ə standing for a, e or ɩ depending on the word) is fixed here,
row by row, from a TSV file referenced by `corrections` in
`fr_mos_sources.toml`:

    id <TAB> column <TAB> wrong <TAB> right <TAB> note

`wrong` is matched against the row after the orthography normalization and
replaced everywhere in that column. The build fails if a correction does not
apply (unknown id, or `wrong` not in the text), so a stale correction cannot
linger silently. Candidates come from `scripts/check_orthography.py`.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

COLUMNS = ("french", "moore")


@dataclass(frozen=True)
class Correction:
    row_id: str
    column: str
    wrong: str
    right: str
    note: str = ""


def load_corrections(path: Path) -> dict[str, list[Correction]]:
    by_id: dict[str, list[Correction]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as f:
        for n, rec in enumerate(csv.DictReader(f, delimiter="\t"), start=2):
            if rec["column"] not in COLUMNS:
                raise ValueError(f"{path}:{n}: column must be one of {COLUMNS}, got {rec['column']!r}")
            if not rec["wrong"] or rec["wrong"] == rec["right"]:
                raise ValueError(f"{path}:{n}: empty or no-op correction")
            c = Correction(rec["id"], rec["column"], rec["wrong"], rec["right"], rec.get("note") or "")
            by_id[c.row_id].append(c)
    return dict(by_id)


def apply_corrections(rows: list[dict], corrections: dict[str, list[Correction]]) -> int:
    """Apply in place; return the number applied. Raises if any does not apply."""
    applied = 0
    seen: set[str] = set()
    for row in rows:
        for c in corrections.get(row["id"], ()):
            seen.add(c.row_id)
            if c.wrong not in row[c.column]:
                raise ValueError(
                    f"Correction for {c.row_id} ({c.column}): {c.wrong!r} not in {row[c.column]!r}"
                )
            row[c.column] = row[c.column].replace(c.wrong, c.right)
            applied += 1
    missing = sorted(set(corrections) - seen)
    if missing:
        raise ValueError(f"Corrections for ids not in the dataset: {missing}")
    return applied
