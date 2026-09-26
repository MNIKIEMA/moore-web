#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "msgspec>=0.20.0",
#     "onnxruntime-gpu[cuda,cudnn]>=1.23.2",
#     "syntok>=1.4.4",
#     "wtpsplit[onnx-gpu]>=2.2.2",
# ]
# ///
"""Compare syntok and SaT (wtpsplit) sentence splitting on Raamde news.

Raamde's Mooré is a summary-style translation of the French, so sentence
counts rarely match. This rebuilds each article's text the same way as
``export_review_units.py --source raamde``, splits it with both segmenters,
and prints per-splitter count statistics. Both splits are written side by side
for inspection.

Usage:
    uv run scripts/compare_raamde_splitters.py \
        --input data/raamde/raamde_corpus_with_lang_id.json \
        -o data/raamde/raamde_syntok_vs_sat.jsonl
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

# Import moore_web from src/ rather than installing it: the package pulls in
# torch/comet/laser, and the segmenters only need msgspec and syntok.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moore_web.flatten import flatten_news_per_entry, normalize_fr, normalize_mo
from moore_web.segment_news_data import segment_entries
import onnxruntime
from wtpsplit import SaT


def _stats(name: str, rows: list[dict], key: str) -> None:
    nf = [len(r[key]["fra"]) for r in rows]
    nm = [len(r[key]["mos"]) for r in rows]
    diff = [abs(f - m) for f, m in zip(nf, nm)]
    lf = [len(s.split()) for r in rows for s in r[key]["fra"]]
    lm = [len(s.split()) for r in rows for s in r[key]["mos"]]
    print(
        f"{name:7} fra={sum(nf):5} mos={sum(nm):5}"
        f" | equal counts {sum(d == 0 for d in diff)}/{len(rows)}"
        f" | |diff|<=1 {sum(d <= 1 for d in diff)}"
        f" | mean |diff| {statistics.mean(diff):.2f}"
        f" | median mos/fra {statistics.median(m / f for f, m in zip(nf, nm)):.2f}"
        f" | words/sent fra {statistics.mean(lf):.1f} mos {statistics.mean(lm):.1f}"
        f" | >60 words fra {sum(x > 60 for x in lf)} mos {sum(x > 60 for x in lm)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="data/raamde/raamde_corpus_with_lang_id.json")
    parser.add_argument("-o", "--output", default="data/raamde/raamde_syntok_vs_sat.jsonl")
    parser.add_argument("--model", default="sat-3l-sm")
    args = parser.parse_args()

    corpus = segment_entries(json.loads(Path(args.input).read_text(encoding="utf-8")))
    syntok = flatten_news_per_entry(corpus, segment=True)
    joined = dict(flatten_news_per_entry(corpus, segment=False))

    # Load the CUDA/cuDNN libraries shipped as nvidia-* wheels (no system CUDA or torch needed).
    onnxruntime.preload_dlls()
    sat = SaT(args.model, ort_providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    urls = [url for url, _ in syntok]
    sat_fr = sat.split([joined[u].french[0] for u in urls])
    sat_mo = sat.split([joined[u].moore[0] for u in urls])

    rows = []
    for (url, parallel), fr, mo in zip(syntok, sat_fr, sat_mo):
        rows.append(
            {
                "url": url,
                "syntok": {"fra": parallel.french, "mos": parallel.moore},
                "sat": {
                    "fra": [normalize_fr(s.strip()) for s in fr if s.strip()],
                    "mos": [normalize_mo(s.strip()) for s in mo if s.strip()],
                },
            }
        )

    Path(args.output).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    print(f"articles: {len(rows)} → {args.output}")
    _stats("syntok", rows, "syntok")
    _stats("sat", rows, "sat")


if __name__ == "__main__":
    main()
