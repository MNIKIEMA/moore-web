# SIDA bilingual book (`book_parser.py`, `flatten.py::flatten_sida_book*`)

Bilingual (`mos` ↔ `fra`) HIV/AIDS awareness book, two-column layout (Mooré
left, French right) on every content page, plus six enumerated Q&A items in
Chapter 5. See [`docs/sida-bilingual-book.md`](../../docs/sida-bilingual-book.md)
for the physical layout and parsing hierarchy.

## 2026-09-17

- **Tried OmegaT for human-reviewed alignment; abandoned it.** After the
  segmentation fixes below, 19/42 page+enum units already matched exactly and
  the rest had known, page-localized causes — looked like a good candidate
  for a manual alignment pass instead of trusting LASER+FastDTW blindly.
  Exported `data/omegat/sida_fr.txt` / `sida_mo.txt` (one sentence per line,
  book order, `==page-N==`/`==enum-N==` marker lines every unit) for
  OmegaT's Tools → Align Files.
- **Finding: OmegaT's language pickers in Align Files are NOT restricted to
  a whitelist.** `sourceLanguagePicker`/`targetLanguagePicker` are editable
  `JComboBox`es (confirmed in upstream source,
  `aligner/.../AlignFilePicker.form`: `editable=true`), and whatever you
  type is checked by `Language.verifySingleLangCode()`, which is just
  `Locale.forLanguageTag(code)` — a BCP-47 *syntax* check, not a lookup
  against known/supported languages. Typing the real ISO 639-3 code `mos`
  works fine (falls back to `DefaultTokenizer`, which is all this alignment
  needs). The initial failure with Arabic/Ukrainian tokenizers was just
  OmegaT's out-of-the-box template languages (`ar-LB`/`uk-UA`) never having
  been overwritten in the project, not a genuine language-support limit.
- **Finding: making OmegaT respect our pre-computed sentence boundaries
  needs two separate settings, not a custom SRX rule.** A hand-rolled "split
  on `\n`" SRX rule is insufficient by itself — the default punctuation-based
  break rules would still fire *inside* a line and undo fixes like the
  dialogue-tag merge. Confirmed via source (`Aligner.java`,
  `TextOptionsDialog.java`) that OmegaT already has purpose-built settings
  for this: (1) Options → Global File Filters → select `Text` → **Options...**
  (not Edit...) → **Line breaks** (makes each line its own paragraph), and
  (2) in the Align Files window's own Options menu, uncheck **Segment**
  (`Aligner.segment` field — when off, raw paragraphs are aligned directly,
  skipping SRX resegmentation entirely).
- **Decided against finishing the OmegaT pass.** Reasons: (a) real setup
  friction working through the above two settings (nested dialogs, "Edit..."
  vs "Options..." look identical at a glance); (b) the review UI treats
  every line pair as equally uncertain, when we already know page/enum
  boundaries are reliable and which specific units need a human look; (c)
  unrelated system memory pressure (swap 65% full from other running apps)
  made the Swing UI laggy in this environment. Decision: build a small
  custom side-by-side reviewer scoped to just the ~23 mismatched units
  instead of a general-purpose CAT tool over all ~280 lines. Not yet
  implemented as of this entry — pick this up before repeating the OmegaT
  attempt.

## 2026-09-14

- **Duplication bug found and fixed in `flatten_sida_book`.** Chapter 5's
  pages (from `enum.start_page` onward) were flattened twice: once via the
  plain `chapter.pages` loop, once via `chapter.enums` (which covers the same
  page range, restructured into title+body per question). Result: ~55–65
  exact-duplicate sentences per language feeding the aligner. Fix: skip pages
  `>= min(enum.start_page for enum in chapter.enums)` in the page loop.
- **Mooré quote-merge over-merging bug found and fixed in `segment_mo`.**
  `segment_mo` reused French's `segment_fr` wholesale, including
  `_merge_open_quotes`. That heuristic assumes quotes close in matching
  pairs (true for French dialogue in this book); Mooré dialogue in this
  source does not reliably pair quotes the same way, so an odd running quote
  count let the merge swallow entire dialogue passages into 1–2 sentences
  (page 7: 8 real sentences collapsed to 2). Fix: `segment_mo` now uses
  syntok's tokenizer only, skipping `_merge_open_quotes` entirely.
- **French dialogue-tag mis-split bug found and fixed in
  `_merge_open_quotes`.** Needed a way to split back-to-back dialogue turns
  (`"Turn one." "Turn two."`, two different quotes glued together by syntok)
  without also splitting off a trailing attribution clause that belongs to
  the sentence before it (`"Foo!", lui dit-elle.` — must stay one sentence).
  Fix: after a leading closing-quote character, only start a new sentence if
  the next cased letter is uppercase (new sentence); lowercase means it's a
  continuation and stays merged. Confirmed against page 3/19/20 (correct
  back-to-back splits) and page 6/11 (correct single-sentence merges,
  previously wrongly split into an orphan attribution clause).
- **Per-page/per-enum alignment wired into `e2e -s sida`.** Added
  `flatten_sida_book_per_unit`, returning one `ParallelText` per page/enum
  item instead of one flattened whole-book list. The PDF's left/right column
  layout means a page's two languages are already known to correspond, so
  there's no reason to make FastDTW guess a monotonic path across all 45
  pages at once — same pattern already used for per-article news alignment
  and per-date conseils alignment (`align_from_embeddings` per unit, then
  concatenate). `flatten_sida_book` (whole-book, unpaired) is kept for the
  `flatten`/`parse-flat` CLI commands, which don't align.
- **Residual per-unit sentence-count mismatch is real content, not bugs.**
  After the three fixes above: 19/42 page+enum units match exactly; the rest
  differ by 1–4 sentences. Manually checked several (enum-2, enum-4, page-4,
  page-20, page-25): the gap is translator expansion/compression (Mooré adds
  explanatory sentences not in French, e.g. around HIV transmission; French
  splits a scene into more sentences with attached dialogue tags than the
  Mooré retelling) and standalone Mooré interjections (`"Ayo!"`). Not
  fixable by boundary-detection changes — this is exactly what FastDTW's
  many-to-many alignment is for. Don't chase this further without evidence
  of a real bug.
- **Known environment gotcha, not a code bug**: `--drop-duplicate` on this
  source loads a COMET-QE model (`McGill-NLP/ssa-comet-qe`, ~2GB) from
  Hugging Face on top of two already-loaded LASER models: FR + Mooré LASER
  encoders and the COMET-QE model, is enough to OOM in a memory-constrained
  environment. The per-page LASER+FastDTW alignment itself completes fine
  (verified: every unit logged a successful `Aligned N pairs` with
  reasonable cosine scores, mostly 0.5–0.8 mean) — the OOM happened
  afterward, in dedup. If re-running `e2e -s sida --drop-duplicate` OOMs
  again, try without `--drop-duplicate` first to isolate.
