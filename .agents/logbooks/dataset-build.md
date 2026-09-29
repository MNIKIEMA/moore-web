# Dataset build (`build_fr_mos_dataset.py`, `fr_mos_sources.toml`, `reviewed_export.py`)

Assembles the French–Mooré training dataset from every source: automatic
outputs in `final_data_hf/`, expert translations, and accepted review-app
units. Cross-source plumbing, so it has no single parser/source.

## 2026-09-29 (Unicode fixes, frozen splits)

- **Look-alike letters and decomposed accents in the Mooré text**, found while
  fixing NLLB <unk>s in mt-training. `scripts/check_orthography.py` on v1.0.0:
  3,677 Mooré rows not NFC (mafand 2,923, news 424, digital 207, conseils
  123), Greek ι for ɩ in 606 mafand rows, script ɡ for g in 481 du-moore rows,
  ʊ for ʋ in 36 udhr rows, ɭ for capital ɩ (Ɩ) in 21 all-caps conseils
  headings, Greek ε for ɛ in 6, a Greek perispomeni used as tilde in 23 mafand
  rows. A model sees each as a different token (it learned ι as a spelling of ɩ).
- **`moore_web.orthography`, applied to every row at build time**:
  `normalize_moore` (look-alikes -> Mooré letters, then NFC),
  `normalize_french` (same minus the Greek letters, which may be real Greek
  in French; then NFC). Not opt-in like punctuation: these are errors for any
  use. Ambiguous letters (ə, œ, ª, ƴ: a few Mooré rows each) are left for
  manual review; the check script lists them. The build fails if a known
  look-alike survives. Typography (’ « ») is kept; the NLLB mapping lives in
  mt-training.
- **More substitutions, confirmed with the user**: the first scanner accepted
  any Latin letter with an ASCII base, so it missed nasal vowels typed with a
  macron: ē 165 rows (expert 155, kade 10), ā 18, ū 18, ī 8, ō 6 ("tēedame" =
  tẽedame). The user confirmed the replacements: ā ē ī ō ū -> ã ẽ ĩ õ ũ, ª ->
  ẽ (conseils font: "sªn" = sẽn), ƴ/Ƴ -> y/Y ("ƴʋʋm"), ǎ ă -> ã, ṅ -> n. Mooré
  side only: French keeps macrons, carons and hooked letters (foreign names).
  The scanner now flags any mark other than tilde and French accents
  (`ALLOWED_MARKS`). Order matters: NFC first (a + U+0304 -> ā), then the
  table, then NFC again; the build's look-alike check caught 2 rows with
  decomposed macrons before this.
- **Row corrections: `corrections/moore.tsv`** (`moore_web.corrections`,
  `corrections = …` in `fr_mos_sources.toml`): one line per fix (`id`,
  `column`, `wrong`, `right`, `note`), applied after the Unicode
  normalization to every split and to mafand; `wrong` carries a little
  context and is matched in that row only; the build fails if a correction
  does not apply (unknown id or text not found). First 10, readings confirmed
  by the user: the 6 ə (la, bãag, sõmblem, tɩ, and two stray), udhr "pœga" ->
  pʋga, and š (lʋɩša -> lʋɩɩsã, vššm -> vɩɩm, Lše -> Lɩse: š stands for ɩ or
  ɩs, so no rule).
- **Left for manual review** (`check_orthography.py` lists them, ~17 Mooré
  rows after the corrections): ə 6, š 3, and one-offs (ɗ "beɗã", ɖ "ɖɩ", ø, Greek ͻ "nͻͻg",
  Armenian ղ "sɩղg", ſ "dirɛktɛſ", udhr "pœga" = pʋga). ɓ (Laaɓal) and œ in
  French words (Sacré-Cœur) are correct.
- **Ids come from the raw text**, computed before any normalization
  (previously the punctuation step ran first; harmless for unit/line ids,
  wrong for text-hash ids such as mafand's). 4,246 rows have corrected text
  and the same id; 15 train rows disappear because they become identical to
  another pair after the fix (11 lexicon_entries, 2 news, 2 du-moore).
- **A seeded split is not stable**: dropping those 15 rows moved 330 rows
  between train/dev/test. Implemented the freeze of `docs/dataset-splits.md`:
  `[splits] frozen_from/frozen_revision` in `fr_mos_sources.toml`; known ids
  keep their split, new ids go to train, removed ids are reported. With
  v1.0.0 pinned: 0 rows change split, dev 2,491 and test 2,573 unchanged,
  train 39,038 -> 39,023. Rebuilt output passes `check_orthography.py
  --strict`.
- **Release**: this is a patch-level content fix of v1.0.0 (same dev/test,
  corrected text), to publish as v1.1.0 together with any other fixes.

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
