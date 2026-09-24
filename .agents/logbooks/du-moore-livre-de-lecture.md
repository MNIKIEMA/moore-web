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

Output: `du_moore_parallel.jsonl` (847 pairs across sections: vocab, sentences,
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

## 2026-09-24

- **Bug found and fixed -- most sectioned lessons lost vocab and misaligned
  sentences.** 31/35 sectioned lessons had vocab items missing on one side
  and 32/35 had different FR/MOS sentence counts, so `zip` shifted every pair
  after the first discrepancy (e.g. `"ade jolies ceintures."` ↔ `"Yibeoog-kãnɡa…"`).
  Three root causes:
  1. `page_lines` snapped `top` to a 5 px grid. Item numbers, bold names and
     drop caps sit 1–4 px off the rest of their line, so lines straddling a
     bucket boundary split (`'① 1 7 –'` / `'–les ceintures la peinture'`).
     Fix: cluster by chaining `bottom` edges (≤ 5 px between neighbours).
  2. Drop caps: pdfplumber leaves the tall capital alone (`C écile`), glues
     it to the previous token (`deC éline`) or both (`voisineCaroline`); the
     old 1–4-letter prefix logic missed `Alain` (5 letters) entirely. Fix:
     geometric detection in `_join_words` (capital > 1.15× taller than the
     next non-capital word, gap < 8 px). `_fix_dropcap_space` and the prefix
     logic are removed -- the text regex would have turned `A la` into `Ala`.
  3. Section ② treated each printed line as a sentence; FR and MOS wrap
     differently. Fix: `_merge_wrapped` joins lines lacking terminal
     punctuation and splits lines holding several sentences.
- **Earlier claim corrected**: "`zip` by index is correct and sufficient"
  only holds after lines are rebuilt into sentences. Lessons whose counts
  still differ are now skipped with a warning: book 2 lessons 3 (MOS has A/B
  dialogue lines) and 7 (one FR sentence → two MOS), book 3 lessons 6 (extra
  MOS sentence) and 7 (MOS missing a period).
- Result: 716 → 847 pairs. vocab 390 → 551 (2 unmatched items left),
  sentences 230 → 200 but now aligned, key 43 (one truncated key fixed),
  passage 53 unchanged.
- **Known limitation / follow-ups** (not fixed):
  - Book 3 lesson 7: MOS `… waooɡr ye Yaa wʋnɡã …` lacks a period, so two
    sentences merge. A split before a capital after `ye` would recover it.
  - Book 2 lesson 7: last FR sentence (`… Burkina Faso, je l’aime`) is two
    MOS sentences -- recoverable with 1:2 pairing.
  - Book 2 lesson 3 (A/B dialogue lines) and book 3 lesson 6 (extra MOS
    sentence) need manual review.
  - Unmatched vocab: book 2 lesson 14 item 14 (MOS only:
    `nii tɩ b yãk yiibu kella yoobe`) and book 3 lesson 10 item 8 (FR only:
    `une fleur`).
