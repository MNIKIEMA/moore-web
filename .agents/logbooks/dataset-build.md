# Dataset build (`build_fr_mos_dataset.py`, `fr_mos_sources.toml`, `reviewed_export.py`)

Assembles the French–Mooré training dataset from every source: automatic
outputs in `final_data_hf/`, expert translations, and accepted review-app
units. Cross-source plumbing, so it has no single parser/source.

## 2026-09-26 (row metadata)

- **Rows now carry `id`, `original_lang`, `doc_id`, `reviewed`** (meaning in
  `docs/dataset-splits.md`, which also holds the not-yet-implemented split
  proposal). `id` must stay stable across rebuilds -- it will pin a frozen
  dev/test -- so it is the upstream id, `{source}-{unit}-{line}` for review
  rows, or a hash of the text, never a position; the build fails on a
  duplicate id. `original_lang` comes from the row's `is_source_orig`, else
  the entry's `original_lang` in `fr_mos_sources.toml`; left out where the
  direction is undocumented (udhr, du-moore, abcburkina-contes).
- **The loader assumed long rows were French-first.** `source_text` was always
  read as French, so a Mooré-original long file would have swapped the sides.
  It now follows `src_lang`/`tgt_lang` and skips non fra–mos rows. No current
  input was affected (conseils and expert are French-first).
- **raamde lost its article URL** in `export_raamde_sat.py`; it now keeps
  `doc_id` (file regenerated, same 1 406 pairs; backup
  `raamde_aligned.jsonl.bak-2026-09-26-sat-no-docid`).

## 2026-09-26

- **The build never reads the review DB.** The DB is a live workspace
  (drafts, the app writing to it), so a build from it isn't reproducible or
  diffable. `moore-web export-reviewed --push` snapshots accepted units to
  per-source JSONL in a private HF dataset repo, and `fr_mos_sources.toml`
  pins the export's commit (`[reviewed].revision`). Same sources file +
  revision → same dataset.
- **Why an HF dataset repo** (`madoss/moore-web-reviewed`, private):
  `faso-web-docs` is the raw-source archive and an HF *bucket* -- no history,
  and `just sync` is two-way, so a stale local copy can overwrite reviewed
  work. moore-web's git would be simplest but the repo is public and the
  books/news licences are unchecked. First export pinned: `a5e716a` (7
  sources, 2 904 rows; no `raamde.jsonl` until raamde units are accepted --
  the build skips missing reviewed files with a message).
- **Filters are per entry, not per tag.** Reviewed raamde units and the
  automatic `raamde_aligned.jsonl` share the `news` tag but need different
  thresholds (none vs `laser_score >= 0.7`). Reviewed rows carry no scores,
  so every threshold is skipped for them. Entry order is dedup priority:
  reviewed entries come first so their copy of a pair wins.
- **The old global `laser_score >= 0.5` was too loose** for summary-style
  sources: it let ~3 600 of the 3 915 old raamde rows through (see
  `raamde-news.md`). Audit a source's score bands before trusting the
  default; `conseils` (7 596 rows, LASER + DTW) hasn't been audited yet.
- **`ruff format` on `cli.py` rewrites unrelated code** (three spots
  predate the formatter). Format only new code there, or restore and
  re-insert; `ruff check` passes either way.
