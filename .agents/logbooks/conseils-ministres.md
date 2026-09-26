# Council of Ministers (`conseils`: `flatten.py::flatten_conseils`, `cli.py::e2e -s conseils`)

Weekly council reports from the Burkina Faso government, one PDF per
language per session (`fra`, `mos`, `fuh`, `dyu`, `gux`). Archived in
`faso-web-docs/conseils-ministres/<YYYY-MM-DD>/`; parsed by the separate
`conseil-ministres` repo, whose output moore-web aligns (see
`docs/council-of-ministers.md`). French is the original.

## 2026-09-26

- **moore-web is a consumer, not the parser.** `conseil-ministres` owns
  crawling (GitHub Action every two months), the bucket sync and parsing for
  five languages. Its output is published to the private HF dataset repo
  `madoss/conseil-ministres-parsed` (`just publish`) and pinned in
  `docs/council-of-ministers.md`: revision `841e617`, parser `f88c870`, 92
  fra–mos sessions. `data/bilingual.json` was a hand copy from April (55
  sessions), which is why 40 sessions were never aligned.
- **Language-code rename broke the flattener silently.** The parser moved
  to ISO 639-3 (`fr` → `fra`); `flatten_conseils` treated anything but `fr`
  as Mooré-first and swapped the sides. Now it accepts `fr`/`fra` + `mos` in
  either order and raises on any other pair.
- **Source sites link the wrong session's translation.** Mooré in
  `2024-05-22` was N°013 and in `2024-11-06` N°025 -- each byte-identical to
  the correct Mooré of its real session; Fulfulde in `2024-02-28` was N°004.
  They were aligned with unrelated French: ~105 pairs in the current
  `conseils_ministres_aligned.jsonl`, median LASER ~0.65, 43 ≥ 0.7, so a
  score threshold can't catch them. Fixed in the archive; `conseil-ministres`
  now has `excluded.json` and a `PP-G N°` session check (crawler, parser,
  `check` command, failing workflow step). Folder dates can be wrong too
  (`2026-06-21` Mooré is a copy of `2026-06-25`).
- **Score audit** (old 7 596 pairs): median LASER 0.78, 84 % ≥ 0.7; 0.7+
  samples correct, 0.5–0.65 mostly wrong, 0.65–0.7 about half usable. Kept
  the default 0.5 threshold for now (7 519 rows pass).
- **Next**: re-align from the pinned `fra-mos.json` (+40 sessions, ≈5 500
  pairs; drops 2024-05-22, 2024-11-06 and the 2024-01-01 New Year address,
  which the parser doesn't include), then rebuild.
