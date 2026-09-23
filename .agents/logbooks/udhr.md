# Universal Declaration of Human Rights (`udhr.py`, `cli.py::e2e -s udhr`)

Bilingual (`mos` ↔ `fra`) UDHR from the OHCHR translation collection, stored
as one text file per language in `faso-web-docs/universal-declaration-human-rights/`
(`udhr-fra.txt`, `udhr-mos.txt`; blank-line-delimited paragraphs, converted
from wooorm/udhr HTML). English and Dioula files exist but are not used.

## 2026-09-23

- **Paired by structure, no LASER.** Both texts have the same shape: title,
  preamble, then 30 article headings. Articles are matched by number and
  paragraphs by position; with segmentation on, a paragraph pair is split
  into sentence pairs only when both sides give the same sentence count.
  Output: 62 sentence pairs (57 paragraph pairs with `--no-segment`),
  `doc_id` = `title` / `preamble` / `article-NN`, `laser_score` null.
- **Mooré has no article 12.** The OHCHR Mooré PDF jumps from 11 to 13; the
  wooorm/udhr source fills it with an `&1` placeholder (documented in the
  manifest's `text_quality_note`). A naive paragraph count matched it 1:1
  with French article 12 — it must be dropped, not paired. Also missing in
  Mooré: the preamble's closing proclamation ("L'Assemblée générale /
  Proclame…"), absent from the Mooré PDF too. Both are skipped and reported.
- **Mooré heading spelling changes at 10:** "Koɛɛg a 9 soaba." but
  "Koɛɛg 10 soaba." — the regex must allow both.
- **CLI placement:** first added as a standalone `prepare-udhr` command,
  then moved into `e2e -s udhr` (same pattern as `digital`, the other
  structurally paired source) so annotation/HF push/dedup come for free.
- **Open:** `is_source_orig` is null — `udhr` is not in
  `ORIGINAL_LANGUAGE` because the original is English and it is unknown
  whether the Mooré was translated from French or English.
