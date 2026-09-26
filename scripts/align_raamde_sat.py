#!/usr/bin/env python3
"""Align SaT-split Raamde news with LASER + FastDTW, one article at a time.

Input is the output of ``compare_raamde_splitters.py``; only its ``sat`` split
is used. Adapted from the ``news`` branch of ``moore-web align`` (cli.py) and
``moore_web.align_corpus``:

1. Encode every French sentence with LASER ``fra`` and every Mooré sentence
   with LASER3 ``mos``, in one pass each.
2. Run FastDTW per article (articles are never aligned across each other).
3. Merge the DTW path into blocks. Raamde's Mooré is a summary-style
   translation, so one French sentence often maps to two Mooré sentences or
   vice versa. ``align_from_embeddings`` emits every path cell as its own pair,
   which repeats the shared sentence. Here a diagonal step starts a new block,
   while horizontal/vertical steps extend the current one, giving 1-1, 1-n,
   n-1 and n-m pairs. ``--no-merge`` keeps the original per-cell behaviour.
4. Score each pair by cosine similarity. A merged block is re-encoded as
   joined text rather than averaging sentence embeddings.

Nothing is filtered by default; use ``--min-laser-score`` or filter the
output later on ``laser_score``.

Usage:
    uv run python scripts/align_raamde_sat.py \
        --input data/raamde/raamde_syntok_vs_sat.jsonl \
        -o data/raamde/raamde_sat_aligned.jsonl
"""

import argparse
import collections
import json
import statistics
from pathlib import Path

import numpy as np

from moore_web.align_corpus import dtw_align


def merge_path(path: list[tuple[int, int]]) -> list[tuple[list[int], list[int]]]:
    """Group a monotonic DTW path into (french_ids, moore_ids) blocks."""
    blocks: list[tuple[list[int], list[int]]] = []
    prev = None
    for i, j in path:
        if prev is None or (i != prev[0] and j != prev[1]):
            blocks.append(([i], [j]))
        else:
            fr_ids, mo_ids = blocks[-1]
            if i != fr_ids[-1]:
                fr_ids.append(i)
            if j != mo_ids[-1]:
                mo_ids.append(j)
        prev = (i, j)
    return blocks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", "-i", default="data/raamde/raamde_syntok_vs_sat.jsonl")
    parser.add_argument("--output", "-o", default="data/raamde/raamde_sat_aligned.jsonl")
    parser.add_argument("--min-laser-score", type=float, default=0.0)
    parser.add_argument("--no-merge", action="store_true", help="Emit one pair per DTW path cell.")
    args = parser.parse_args()

    from laser_encoders import LaserEncoderPipeline

    articles = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line]
    articles = [a for a in articles if a["sat"]["fra"] and a["sat"]["mos"]]
    print(f"{len(articles)} articles")

    laser_fr = LaserEncoderPipeline(lang="fra")
    laser_mo = LaserEncoderPipeline(lang="mos")

    all_fr = [s for a in articles for s in a["sat"]["fra"]]
    all_mo = [s for a in articles for s in a["sat"]["mos"]]
    print(f"Encoding {len(all_fr)} French / {len(all_mo)} Mooré sentences…")
    fr_embs = laser_fr.encode_sentences(all_fr, normalize_embeddings=True)
    mo_embs = laser_mo.encode_sentences(all_mo, normalize_embeddings=True)

    rows = []
    fr_off = mo_off = 0
    for a in articles:
        fr, mo = a["sat"]["fra"], a["sat"]["mos"]
        fe, me = fr_embs[fr_off : fr_off + len(fr)], mo_embs[mo_off : mo_off + len(mo)]
        fr_off += len(fr)
        mo_off += len(mo)

        path = dtw_align(fe, me)[0]
        blocks = [([i], [j]) for i, j in path] if args.no_merge else merge_path(path)
        for fr_ids, mo_ids in blocks:
            rows.append(
                {
                    "doc_id": a["url"],
                    "french": " ".join(fr[i] for i in fr_ids),
                    "moore": " ".join(mo[j] for j in mo_ids),
                    "fr_ids": fr_ids,
                    "mo_ids": mo_ids,
                    "_embs": (fe[fr_ids[0]], me[mo_ids[0]]) if len(fr_ids) == len(mo_ids) == 1 else None,
                }
            )

    # Single-sentence pairs reuse their embeddings; merged blocks are re-encoded as joined text.
    merged = [r for r in rows if r["_embs"] is None]
    if merged:
        print(f"Re-encoding {len(merged)} merged blocks…")
        mfe = laser_fr.encode_sentences([r["french"] for r in merged], normalize_embeddings=True)
        mme = laser_mo.encode_sentences([r["moore"] for r in merged], normalize_embeddings=True)
        for r, f, m in zip(merged, mfe, mme):
            r["_embs"] = (f, m)
    for r in rows:
        f, m = r.pop("_embs")
        r["laser_score"] = round(float(np.dot(f, m)), 4)

    kept = [r for r in rows if r["laser_score"] >= args.min_laser_score]
    Path(args.output).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8"
    )

    scores = [r["laser_score"] for r in kept]
    shapes = collections.Counter(f"{len(r['fr_ids'])}-{len(r['mo_ids'])}" for r in kept)
    print(f"Wrote {len(kept)} pairs → {args.output}")
    print(
        f"laser_score mean {statistics.mean(scores):.3f} median {statistics.median(scores):.3f}"
        + "".join(f" | >={t} {sum(s >= t for s in scores)}" for t in (0.6, 0.7, 0.8))
    )
    print("shapes:", ", ".join(f"{k} {v}" for k, v in shapes.most_common(8)))


if __name__ == "__main__":
    main()
