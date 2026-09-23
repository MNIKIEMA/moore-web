# Kadé facilitateur book (`book_parser_facilitateur.py`, `flatten.py::flatten_facilitateur_pair`)

Bilingual (`mos` ↔ `fra`) HIV/AIDS facilitator manual, two separate monolingual
PDFs (not two-column like the SIDA bilingual book), aligned by chapter/section
order. See [`docs/kade-facilitator-manual.md`](../../docs/kade-facilitator-manual.md)
for the physical layout and parsing hierarchy. Also referenced elsewhere as
"sida-facilitateur" (its `final_data/` output name).

## 2026-09-23

- **Mooré `•` bullets are flattened out of order** (issue #44, not fixed
  yet). `_classify_items_lines` keeps numbered items and bullets in two
  separate lists and `flatten_content` emits items, then bullets, then
  body, so bullets nested under a numbered item end up after *all* items of
  the container. Compounded in Mooré "Bũmb d sẽn tõe n zãmse" (ch5), which
  gets **no subsections**: question headings are absorbed as continuation
  lines of the preceding item or bullet. Net effect in `kade-ch5-learning`:
  the three "Tẽeb la mansem…" bullets plus the following question and its
  instruction (9 Mooré lines) landed at the end of the unit instead of after
  "…yãmb na yã :". French is unaffected (its sub-points are indented, not
  bulleted). The PDF also has an empty `2.` under the circumcision
  question; the parser opens item 2 and makes the next heading its text
  (order stays correct).
- **Verification technique:** diff the unit's tokens against
  `pdftotext` of the exact page range as a word multiset/sequence — shows
  in one pass that nothing is missing and exactly which block moved. Only
  other differences were line-break artefacts (`kʋ- ba`, `bi- bɩɩlema`,
  `ci- dessous`).
- The ch5 unit was fixed by hand in the review app (draft by `madoss`) and
  checked at 65/65 lines. Do not re-export the Kadé units while its
  annotation is in progress; see the SIDA logbook (2026-09-23) for why a
  re-export would not update existing units anyway.

## 2026-09-17

- **Added `flatten_facilitateur_pair_per_unit`**, a per-unit sibling of
  `flatten_facilitateur_pair` for `notebooks/merge_review.py` (built for the
  SIDA book, generalized to accept any source's `{uid: {fra, mos}}` JSONL —
  see that book's logbook entry from today). Reuses the same
  role-matching approach the 2026-09-16 session introduced (chapters matched
  by `Chapter.number`, sections matched by canonical bilingual role via
  `_FACILITATEUR_SECTION_ROLES`/`_FACILITATEUR_ROLE_ORDER`) but emits one
  `ParallelText` per chapter-title and per role instead of two long flat
  lists — 38 units total (6 chapter titles + roles per chapter) against the
  real Kadé PDFs, with `ValueError` on a chapter/role-set mismatch instead
  of silently pairing unrelated text. Shared cleaning helpers
  (`_facilitateur_clean_title_fr/mo`, `_facilitateur_keep`,
  `_facilitateur_flatten_fr/mo_parts`) were extracted out of
  `flatten_facilitateur_pair` itself to avoid duplicating logic between the
  two — behavior-preserving refactor, same regression tests still pass.
- **Resolves the 2026-09-16 "open, not investigated further" question**
  about chapter 0's Mooré intro having fewer recognized sections than
  French's: it's not a missing-title bug. Chapter 0's Mooré "Intro" section
  body is *entirely* a French-language colophon (`"Langue : Mooré parlée au
  Burkina Faso ... Traduit par : GANSAONRE Guillaume ... © SIL Région
  Afrique 2007. Utilisée avec autorisation. ... © Shellbook Publishing
  Systems"`) — real publisher credits, not story/context content, and nothing
  the French "Intro" (just the book's title) meaningfully corresponds to.
  The two are still folded into the same `kade-ch0-context` unit today since
  role-matching doesn't know this section is different in kind, not just
  content — worth revisiting if chapter 0 keeps needing special-casing.
- **Bug found and fixed: copyright-boilerplate leak.** That same colophon's
  `"Utilisée avec autorisation."` sentence leaked into the Mooré corpus
  (found by inspecting `data/omegat/sida-facilitateur_mo.txt` directly).
  `_COPYRIGHT_RE` only matched a literal `©`; this sentence has none of its
  own; `segment_mo` splits the colophon into three sentences and only the
  first/third contain `©`. Fixed by extending `_COPYRIGHT_RE` to also match
  `"[Uu]tilisée?s? avec (l')?autorisation"` — a standard SIL/Shellbook
  permission-notice phrase likely to recur in other books from the same
  publisher family (the SIDA bilingual book's front matter credits the same
  Shellbook Publishing Systems origin).
- **Two more correctness bugs found via code-review + verified against the
  real Kadé PDFs, both fixed:**
  - `segment=False` paths in `_facilitateur_flatten_fr/mo_parts` filtered
    `_facilitateur_keep()` on the raw string *before* `_PAGE_REF_RE` stripped
    it, so a line that's entirely a page reference (e.g. `"(voir p. 12)"`)
    passed the filter but became an empty string after cleaning, appending
    empty entries. Didn't reproduce against the real book (no body line
    there is purely a page ref) but is real and reachable via `moore-web
    flatten -s kade --no-segment`. Fixed by filtering on the cleaned text.
  - The `kade-ch{N}-title` unit was guarded by the raw, uncleaned
    `chapter.title.strip()` check instead of re-checking `_facilitateur_keep()`
    on the cleaned title the way every section/subsection title append
    already does — a title that's entirely a page reference would produce an
    empty-string title unit. Fixed to match the existing pattern.
- **One review finding refuted by testing against the real PDFs**: a claim
  that Mooré chapter headings use word-numerals (not digits) that
  `CHAPTER_RE`'s `(\d+)` can't match, desyncing chapter-number pairing. Ran
  `flatten_facilitateur_pair_per_unit` end-to-end against both real PDFs:
  chapters matched 0–5 correctly on both sides, no `ValueError`. The
  finding's cited supporting test doesn't exist in the repo. **Method note:**
  always re-verify a review finding against real data before acting on it,
  especially ones citing "existing tests" as evidence — this one didn't.
- **`scripts/export_review_units.py --source kade` (alias `facilitateur`)**
  added, taking `--fr-input`/`--mo-input`. This is also what actually fixes
  the proper-name bug found in `data/omegat/sida-facilitateur_fr.txt`
  (`Kadé`/`Kaluu`/`Katiu`/... never replaced with the SIDA book's standard
  names): that file was a scratch export that skipped
  `replace_facilitateur_names_fr`; the new script goes through
  `flatten_facilitateur_pair_per_unit`, which calls it correctly via the
  shared `_facilitateur_flatten_fr_parts` helper. Verified: 0 unreplaced
  names in the new export vs. 148 in the old one.

## 2026-09-16

Session started from a request to check segmentation quality; escalated into
a full audit once the first duplicate was confirmed against the raw PDF text.
Six commits on `fix/sida-facilitateur-segmentation` (PR #41).

- **Missing intro-subsection title duplicated a paragraph.**
  `FRENCH_INTRO_SUBSECTION_TITLES` listed only 5 of the "Comment utiliser ce
  manuel" intro's 6 subsections — missing `"Prier et agir"`. Its `"6. Prier
  et agir"` line fell through to generic numbered-item parsing, so the
  paragraph appeared twice (once in the previous subsection's body, once as
  a stray item). Mooré's equivalent list already had all 6.
- **Unanchored heading regexes matched mid-paragraph.** Both the intro-
  subsection patterns (`cli.py::_parse_kade_file`) and the main
  `SECTION_TITLES`/`MOORE_SECTION_TITLES` patterns matched a title as an
  unanchored *prefix* of any line. Two distinct failure modes found: (1) a
  wrapped body line starting with lowercase `"l'histoire"` (French, mid-
  sentence) got matched as the `"L'histoire"` heading, truncating one
  section's body and creating a bogus duplicate section; (2) Mooré `"Reem"`
  (a real word, the sketch/story section title) matched inside ordinary
  words like `"reemd"` ("plays"), falsely splitting sections mid-paragraph
  (confirmed: chapter 1's Mooré section list had `"Reem"` three times, one
  before the Questions section where no sketch heading belongs). Fixed with
  two different techniques depending on real-world line shapes found in the
  text: intro-subsections got a hard `\s*$` end-anchor (headings always sit
  alone on their own line there); main section titles got a softer
  "not-followed-by-a-lowercase-letter" guard instead, because at least one
  real heading (`"Ce que dit la Bible – quelques passages"`) has trailing
  text on the same line and a full-line anchor would have broken it.
- **Missing Mooré spelling variants silently merged whole sections.**
  `MOORE_SECTION_TITLES` was missing spellings actually used in the source:
  `"Wẽnnaam sebra sẽn yet bũmb ninga"` (no tilde on the final syllable, vs.
  the one listed variant `"...ningã"`), a circumflex variant
  `"...bûmb ninga"` unique to chapter 3, `"Pʋʋsgo la tʋʋma"` (a third
  spelling of "Prier et agir", chapter 4), and chapter 5's completely
  different wording for "Choses à apprendre", `"Bũmb d sẽn tõe n zãmse"`
  ("things we can learn" vs. the usual "things we should learn" phrasing —
  confirmed same topic by comparing to the French section's opening
  question). Each missing variant meant that section's content silently
  absorbed into whichever section came before it. This was the single
  biggest quality issue in the corpus — chapters 2–5 were each missing at
  least one real section (Bible study and/or Prier et agir) before these
  were added.
- **A heading split across a PDF block boundary was unmatchable.**
  `extract_pdf_blocks` joins each PyMuPDF text block with `"\n\n"`; when a
  heading was itself split into two blocks (`"Wẽnnaam Sebra sẽn yet"` /
  `"bũmb ninga"`), it became two lines with a blank line between them, which
  no line-based regex could ever match as one heading. Fixed with a bounded
  lookahead (`_match_heading_line` / `_lookahead_heading`) that skips blank
  lines and grows the accumulated text one non-blank line at a time, but
  only while it stays a strict prefix of some canonical title — bails out
  immediately otherwise, so it can't accidentally swallow unrelated body
  text.
- **Page-number footers leaked mid-sentence.** `extract_multicolumn_blocks`
  (used for the two-column SIDA book) already drops a block that's purely
  digits; `extract_pdf_blocks` (used here) didn't have the same guard, so a
  standalone page-number block landing between two lines of one paragraph
  got joined straight into the sentence (`"...plusieurs 5 membres du
  groupe..."`). Same fix ported over.
- **Body-building only excluded an item's first line, not its wrapped
  continuation.** Both `_build_section` and `_split_into_subsections`
  excluded a line from `body` only if it itself matched
  `NUMBERED_ITEM_RE`/`BULLET_ITEM_RE` (i.e. started a new item) — every
  *continuation* line of a multi-line PDF-wrapped item stayed in `body` too,
  duplicating it into both `items[].text` and `body`. This alone accounted
  for a large fraction of the corpus (e.g. chapter 3's "Questions à
  discuter" body went from ~13 sentences of leftover fragments to empty).
  Fixed by mirroring `collect_numbered_items`/`collect_bullet_items`'s own
  line-consumption into a shared `_lines_consumed_by_items` /
  `_body_from_lines`, so body excludes exactly what ended up in an item.
- **Numbered items and bullets swallowed each other's lines when
  interleaved.** `collect_numbered_items` and `collect_bullet_items` each
  walked the same lines independently, neither aware of the other's marker.
  A numbered item's continuation didn't stop at a following `"•"`, so it
  absorbed the bullets into its own text *while* `collect_bullet_items`
  captured those same bullets again separately. Worse in the reverse
  direction: `collect_bullet_items`'s continuation never stopped at a
  numbered-item marker either, so one runaway bullet swallowed the *entire
  rest of a section* (every later restarting numbered list) into its own
  text a second time — found while chasing why chapter 5's "Choses à
  apprendre" sentence count was 122 instead of the expected ~70. Fixed by
  replacing both independent passes with one unified `_classify_items_lines`
  that tracks a single mode (item/bullet) at a time, so a marker for the
  other kind always ends the current one instead of being absorbed by it.
  Also affected chapter 1's "sketch" (138/129 → 88/85) and chapter 4's
  "choses" (120/121 → 71/72), not just chapter 5.
- **Mooré scripture-reference citations over-fragment vs. French.** Chapter
  5's Bible appendix groups references under `"Zãmsog a N soaba"` (Lesson N)
  labels (`"Zãmsog a yembr (1) soaba"`, etc.) — not recognized as markers at
  all, so each label leaked into the *previous* lesson's last item as
  trailing text (same bug class as above, different content). Separately:
  each citation entry uses a period between topic and reference
  (`"Wẽnnaam sõngda nãong rãmbã. Yɩɩl Sõamyã 22:24; ..."`), while French
  uses a colon (`"Dieu aide les nécessiteux : Psaumes 22.24 ; ..."`) —
  `segment_mo` correctly splits on that period by its own rules, but it
  isn't a real sentence boundary here, so each Mooré citation fragmented
  into 2–3 pieces where French's punctuation choice kept the same content as
  one unit. User confirmed via `data/omegat/sida-facilitateur_fr.txt` that
  French already produces "one line per citation" naturally (no fix needed
  there). Fixed with `LESSON_GROUP_RE` (ends the current item cleanly
  instead of leaking into it) plus a new `NumberedItem.atomic` flag set for
  items following a lesson-group marker; `flatten_facilitateur_pair` checks
  each flattened Mooré part against `atomic_item_texts()` and keeps atomic
  entries as one line instead of running `segment_mo` on them. Scoped to
  this citation-list content only — ordinary prose items (Questions à
  discuter, etc.) still get normal sentence-splitting.
- **Page-range refs and letterless lines leaked into the corpus.**
  `_PAGE_REF_RE` matched `"(p. 3)"` but not a range like `"(p. 42 - 43)"`,
  so ranges survived as their own junk lines; extended the pattern to accept
  an optional `"- N"` suffix. Also added a "has at least one letter" check
  to `_keep()` — a lone `"."` (left over from splitting a parenthetical
  aside) wasn't caught by the existing digits-only/URL/copyright filters.
- **Stray control byte from a symbol font.** Page 49 of the French PDF
  renders a decorative sub-bullet glyph as a raw `\x01` (SOH) control byte
  instead of real Unicode — same class of font-encoding gap
  `_expand_ligatures` already handles for fi/ff/ffl ligatures, just landing
  on a control character instead of a private-use codepoint (13 occurrences,
  French only). Fixed by stripping all C0 control bytes except `\n`/`\t` in
  `_expand_ligatures`. Output counts unchanged — it was already being
  silently absorbed into whitespace, not producing its own junk lines; this
  was a text-cleanliness fix, not a segmentation-count fix.
- **Checked and confirmed NOT bugs** (verified against raw PDF occurrence
  counts before ruling either way — this was the deciding technique
  throughout the session):
  - French's `_merge_open_quotes` keeping a multi-sentence quoted dialogue
    passage as one block (chapter 3 "story" slot showing a -8 to -10 diff).
    Same known, intentional design tradeoff documented in the SIDA
    bilingual-book logbook — Mooré doesn't reliably pair quote marks, so
    `segment_mo` skips the merge and legitimately produces more sentences
    from the same content.
  - Chapters 2/3's extra `"L'histoire de Kadé"` section and chapter 5's
    doubled `"Ce que dit la Bible"`: real content, not duplication — a
    genuine `"(suite)"` recap heading mid-chapter, and a genuine
    end-of-book supplementary-passages appendix respectively.
- **Method note for next time:** an early diagnostic script joined a
  section's items/bullets/body with a single space before re-segmenting,
  which silently collapsed short unpunctuated list entries and produced a
  misleading "bible: FR=17 vs MO=69" reading (the real, per-item-segmented
  gap was 44 vs 70, later closed to 32=32 by the atomic-item fix above).
  **Always segment each flattened part independently, matching what the
  production pipeline (`flatten_book_to_list` + per-item `segment_fr`/
  `segment_mo`) actually does — never join parts before segmenting when
  diagnosing count mismatches.**
- **Section-count mismatches make positional diffing misleading.** French
  and Mooré don't always produce the same *number* of sections per chapter
  (e.g. one language has an extra recap section, or the intro chapter's
  structure differs entirely). Comparing section `i` in French against
  section `i` in Mooré routinely produced 3–6x-looking "divergences" that
  were purely a pairing artifact. Matching by canonical semantic slot
  (`story`/`questions`/`choses`/`sketch`/`bible`/`prier`, chapter 0 handled
  separately) instead of raw list position was necessary to get a trustworthy
  per-chapter diff.
- **Impact.** Total parsed sentence count (`parse-flat -s kade`, unaligned):
  French 1440 → 911, Mooré 1423 → 917 (~36%/~36% reduction) across all fixes
  above. Every drop was verified as duplicate content removed (confirmed
  against raw PDF occurrence counts), not data loss.
- **Open, not investigated further:** chapter 0's Mooré intro has only 2
  recognized sections (`MOORE_INTRO_SECTION_TITLES`) vs. French's 4 — not
  yet determined whether this is a genuine structural simplification in the
  Mooré intro or a missing-title bug like the ones found above. Also
  unconfirmed: a `"choses"`-slot duplicate-looking entry ("Un groupe de 3 à
  6 personnes...") that looks like it could be legitimate repeated content
  (a microfinance/tontine definition reused across chapters) rather than a
  bug — flagged but not checked against raw-text occurrence counts.
- **Added `data/omegat/sida-facilitateur_fr.txt` / `sida-facilitateur_mo.txt`**
  (OmegaT-ready exports, one sentence per line), matching the format of the
  existing `sida_fr.txt`/`sida_mo.txt` for the other SIDA book. Since this
  book has no simple page structure, markers are `==chapN-slot==` (canonical
  slot names) instead of `==page-N==`. Generated by a scratch script, not
  committed to the repo — same as the sibling sida_fr/mo files, which also
  have no in-repo generator.
