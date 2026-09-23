# Presidential New Year messages (`new_year_message.py`, `prepare-new-year-message`)

French and Mooré texts of the 2024 New Year address, curated as UTF-8 files
in `faso-web-docs/messages-nouvel-an/2024/`. Each blank-line block is one
segment; `align` then runs LASER + FastDTW on the flat lists.

## 2026-09-23

- **No sentence segmentation yet.** Blocks are passed to `align` as-is:
  FR 47 blocks (max 487 chars), MO 55 blocks (max 285). Running
  `segment_fr`/`segment_mo` per block (not wired in) gives FR 56 / MO 59
  with no text lost, so it helps only a little.
- **Block boundaries are not reliable anchors across languages.** Two Mooré
  blocks span two French blocks each (MO block 32 = FR 25 + 26; MO 36 =
  FR 30 + 31), and the translator often splits one French sentence into two
  (FR 11b = MO 12+13, FR 18 = MO 22–24). Mooré also splits at `;` + capital
  where French does not. Fine for whole-list FastDTW; do not align block by
  block.
- **Curated-text leftovers** (in `faso-web-docs`, not this repo; editing
  them requires updating `text_sha256`/`text_size_bytes` in the manifest):
  `tʋʋm- gãnegdg` (MO block 22), `men-sekre ,` (MO block 26), missing final
  period on MO block 24. Not fixed yet.
