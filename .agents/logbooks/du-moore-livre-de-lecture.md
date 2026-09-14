# Du Mooré au Français – livre de lecture (`parse_du_moore.py`)

Three-volume bilingual reading textbook published by CIER/AZN (2024), Burkina
Faso. Covers `mos` ↔ `fra`. Each lesson occupies two consecutive pages sharing
the same `Kaoreng … soaba` header: French page first, Mooré page second,
consistent across all three volumes.

Two lesson layouts across the volumes:

- **sectioned** (books 1–2, early book 3): lessons with ① (vocab) and ②
  (sentences) markers, plus a "D gom fãrende" / "Kʋmbɡo" drill section.
- **prose** (book 3 lessons 40–48): plain numbered vocab list followed by a
  reading passage.

Source PDFs in repo root:
- `Du_Moore_au_Francais_1_Noir_et_Blanc_pp_01-30_Lecons_1-16.pdf`
- `Du_Moore_au_Francais_2_Noir_et_Blanc_pp_31-60_Lecons_17-31.pdf`
- `Du_Moore_au_Francais_3_Noir_et_Blanc_pp.61-94_Lecons_32-48.pdf`

Output: `du_moore_parallel.jsonl` (716 pairs across sections: vocab, sentences,
key, passage).

## 2026-09-14

- **Bug found and fixed -- `conversation` section produced 225 misaligned
  pairs.** Root cause: `extract_conversation` extracted "D gom fãrende"
  examples from the French page and "Kʋmbɡo" examples from the Mooré page
  and zipped them as if they were translations. They are not: "D gom
  fãrende" gives French sentence templates for student writing practice;
  "Kʋmbɡo" gives French dictation/listening sentences for phonetic
  exercises. Both sides were French, so every pair was FR↔FR noise.
  Fix: removed `extract_conversation` entirely; the section stop condition
  previously embedded in `CONV_HEADERS` was inlined as `_SECTION2_STOP` to
  keep `extract_sentences_sectioned` correct. Result: 941 → 716 pairs,
  all genuinely parallel.
- **Structural insight confirmed from PDF inspection.** The French and Mooré
  pages are already positionally aligned: sentence N in section ② on the
  French page directly translates sentence N on the Mooré page. No
  embedding-based alignment is needed for sections ①/② -- `zip` by index
  is correct and sufficient.
- **Known limitation**: `extract_key` uses a positional heuristic (first
  long line after the subtitle). On a handful of lessons the subtitle spans
  two lines and the heuristic consumes the first sentence of ② as the key.
  Not fixed; affects a small number of pairs and is visible in the JSONL
  as an unusually short `key` entry followed by a shorter-than-expected
  `sentences` list for that lesson.
- **Known limitation**: `extract_sentences_sectioned` handles a drop-cap
  artefact where pdfplumber inserts a space after the first letter (e.g.
  `L e bébé`) via `_fix_dropcap_space`. Applied to French pages only;
  Mooré pages don't exhibit this artefact in the tested volumes.
