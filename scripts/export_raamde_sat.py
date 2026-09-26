#!/usr/bin/env python3
"""Split SaT + DTW Raamde pairs into accepted pairs and review units.

Reads ``align_raamde_sat.py`` output and writes:

- ``--final``: pairs good enough to use unreviewed, in the
  ``french``/``moore``/``source``/``laser_score`` rows ``build_fr_mos_dataset.py``
  reads. Accepted: 1-1 pairs with ``laser_score >= 0.7``, and 2-sentence
  blocks (1-2 / 2-1) with ``>= 0.75`` -- merged blocks score higher partly
  from length, so they need a stricter bar. Blocks of 3+ sentences are never
  accepted unreviewed.
- ``--review``: every other pair with ``laser_score >= 0.6`` (mostly the
  0.6–0.7 band, plus 3+ blocks and 2-blocks below 0.75), one unit per article
  in the review schema. Lines are pre-aligned: line *i* on each side is one
  DTW pair, so reviewers reject or fix pairs instead of aligning from scratch.

Pairs below 0.6 are dropped: spot checks found them almost always wrong.

Usage:
    uv run python scripts/export_raamde_sat.py \
        --input data/raamde/raamde_sat_aligned.jsonl \
        --final final_data_hf/raamde_aligned.jsonl \
        --review data/review/raamde_units.jsonl
"""

import argparse
import json
from pathlib import Path

ACCEPT_1_1 = 0.7
ACCEPT_2_BLOCK = 0.75
REVIEW_MIN = 0.6


def accepted(row: dict) -> bool:
    size = max(len(row["fr_ids"]), len(row["mo_ids"]))
    if size == 1:
        return row["laser_score"] >= ACCEPT_1_1
    return size == 2 and row["laser_score"] >= ACCEPT_2_BLOCK


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", "-i", default="data/raamde/raamde_sat_aligned.jsonl")
    parser.add_argument("--final", default="final_data_hf/raamde_aligned.jsonl")
    parser.add_argument("--review", default="data/review/raamde_units.jsonl")
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line]

    final, units = [], {}
    for r in rows:
        if accepted(r):
            final.append(
                {
                    "french": r["french"],
                    "moore": r["moore"],
                    "source": "news",
                    "doc_id": r["doc_id"],
                    "laser_score": r["laser_score"],
                }
            )
        elif r["laser_score"] >= REVIEW_MIN:
            unit = units.setdefault(r["doc_id"], {"fra": [], "mos": []})
            unit["fra"].append(r["french"])
            unit["mos"].append(r["moore"])

    Path(args.final).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in final), encoding="utf-8")
    Path(args.review).write_text(
        "".join(json.dumps({uid: u}, ensure_ascii=False) + "\n" for uid, u in units.items()), encoding="utf-8"
    )
    n_review = sum(len(u["fra"]) for u in units.values())
    print(f"accepted {len(final)} → {args.final}")
    print(f"review   {n_review} pairs in {len(units)} articles → {args.review}")
    print(f"dropped  {len(rows) - len(final) - n_review} pairs below {REVIEW_MIN}")


if __name__ == "__main__":
    main()
