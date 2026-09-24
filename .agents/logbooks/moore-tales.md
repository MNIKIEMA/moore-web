# Mooré tales, volume 5 (`moore_tales_parser.py`, `cli.py::e2e -s moore-tales`)

mooreburkina.com "Contes vol 5 avec français" app, archived in
`faso-web-docs/mooreburkina-priority/apps/mos-contes-volume-5/`: 60 HTML
pages (Mooré page, then French page, per tale), `text/` copies, and
`segments.jsonl` with audio timing for the Mooré pages only. Covers
`mos` ↔ `fra`; Mooré is the original.

## 2026-09-24

- **Tale is the only anchor.** Paragraph counts match for 3/30 tales and
  the French is a free translation (970 MOS vs 785 FR sentences), so no
  paragraph or position pairing. Hand anchors (abc-coepouses style) would
  mean 30 tales of manual work; LASER inside each tale is good enough to
  start from.
- **LASER + FastDTW per tale**: 987 pairs, 71 % ≥ 0.6, 18 % < 0.5. Content
  order is well preserved; the main defect is 1 FR : n MOS -- 337 rows
  (34 %) repeat a French sentence that merges several Mooré ones, e.g.
  `« N-ye sẽ !` + `Fo sã n yiis maam…` ↔ `« Bien sûr! Quand tu m'auras…`.
  Next step: merge consecutive rows sharing a sentence into one n:1 pair
  (`align_from_embeddings` is shared, so this touches every source).
- **HTML extraction traps**: adjacent `div.txs` segments sometimes have no
  whitespace between them (`ye.Yaa` when joined naively), and tale 21's
  title is split across spans (`2` + `1 Kɩɩba`, read as tale 2 with a
  space join). Fix: join segments with a space, spans without. Verified
  word-for-word against `text/*.txt`.
- **Source text quality**: French has OCR-style slips (`tais`/`tut` for
  `fais`/`fut`, `27.Ue femme` title), Mooré has `Iʋɩ` for `lʋɩ`. Not
  corrected; visible in review.
- **Imported into the review app** as 30 per-tale units
  (`mos-contes-volume-5-01` … `-30`, title on row 1; backup
  `reviews.sqlite3.bak-2026-09-24-moore-tales`). 27/30 have uneven sides
  (up to 86 rows), so reviewers align from raw sentence lists -- heavy.
  Units don't update on re-export, so a later merged/pre-aligned export
  would need new unit ids.
