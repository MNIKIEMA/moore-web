# Raamde bilingual news (`segment_news_data.py`, `flatten.py::flatten_news_per_entry`, `scripts/align_raamde_sat.py`)

raamde-bf.net articles scraped into `data/raamde/raamde_corpus.json` (429
entries; 409 with both languages). Each article is a Mooré run followed by a
French run; see `docs/raamde-news.md` for the record shape and language
boundary. Covers `mos` ↔ `fra`. The Mooré is a summary-style rewrite of the
French, not a sentence-by-sentence translation.

## 2026-09-26

- **`final_data_hf/raamde_aligned.jsonl` is mostly noise.** 3 915 pairs from
  syntok + LASER + FastDTW, never filtered on score (min 0.03). Spot checks:
  < 0.6 (945) wrong -- same story, different sentence; 0.6–0.7 (1 567) mostly
  wrong or partial; 0.7–0.8 (1 277) mostly right, loose paraphrase; ≥ 0.8
  (126) good. So ~1 400 usable rows. `comet_qe` (median 0.605) is present but
  was not used to filter either. Check whether `build_fr_mos_dataset.py`'s
  quality filter already drops these before relying on the file.
- **Splitter doesn't fix the mismatch.** `scripts/compare_raamde_splitters.py`
  (syntok vs SaT `sat-3l-sm`) → `data/raamde/raamde_syntok_vs_sat.jsonl`:

  | | fra / mos sents | equal counts | mean \|diff\| | >60-word fra sents |
  |---|---|---|---|---|
  | syntok | 4 892 / 5 925 | 53/409 | 4.43 | 368 |
  | SaT | 6 394 / 6 140 | 44/409 | 4.78 | 107 |

  SaT splits the long French run-ons (glued sentences with no space) that
  syntok can't; Mooré barely changes. Counts still match in ~11 % of articles,
  so manual review of raw sentence lists stays heavy → auto-align instead.
- **SaT on GPU without torch.** The script uses PEP 723 inline deps
  (`uv add --script`) and imports `moore_web` from `src/` instead of
  depending on the package, which drags in torch/comet/laser/marimo.
  `onnxruntime-gpu` alone fails with `libcublasLt.so.12` missing; the
  `onnxruntime-gpu[cuda,cudnn]` extras + `onnxruntime.preload_dlls()` load
  the CUDA libs from pip wheels.
- **SaT + LASER + FastDTW with merged blocks.**
  `scripts/align_raamde_sat.py` → `data/raamde/raamde_sat_aligned.jsonl`
  (5 204 pairs, mean 0.646). Adapted from `cli.py`'s news branch, with the
  change `moore-tales.md` asked for: consecutive DTW path steps sharing a
  sentence are merged into one 1:n / n:1 pair (diagonal step = new block),
  and merged blocks are re-encoded as joined text for scoring. Done in the
  script only; `align_from_embeddings` is unchanged. `--no-merge` gives the
  old per-cell output.
  Shapes: 1-1 4 066, 2-1 407, 1-2 302, 3-1 152, 1-3 82, 4-1 75.
- **Merged blocks score inflated.** Mean 0.70–0.74 for n:1 vs 0.625 for 1-1,
  partly from length; 3+ sentence blocks are often just same-topic (a 5-1 at
  0.72 was wrong). Proposed filter: 1-1 ≥ 0.7, 2-sentence blocks ≥ 0.75, drop
  3+ → **1 406** pairs (1 195 with 2-blocks ≥ 0.8). Not applied yet; the
  0.6–0.7 band (~1 600) could go to review but would mostly be rejected.
- **Next**: apply the filter and replace `final_data_hf/raamde_aligned.jsonl`
  (`french`/`moore`/`laser_score` keys for the build script); run the same
  score audit on the other LASER+DTW outputs, `conseils_ministres_aligned.jsonl`
  (7 596) first. The 409 raamde units in the review DB (unreviewed) are
  syntok-split and predate this.
