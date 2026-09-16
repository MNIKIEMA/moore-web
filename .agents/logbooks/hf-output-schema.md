# HF output schema (`flatten.py::AlignedCorpus`/`flat_rows_to_long`, `annotate.py`, `cli.py::_finalize_aligned`)

The corpus-output layer shared by every source (sida, kade, raamde-news,
conseils, niggli-dictionary-mos-fra-eng, digital-postal-glossary*) once the
pipeline reaches `AlignedCorpus`. Cross-cutting pipeline module, not a single
parser -- findings here apply regardless of which source produced the rows.

## 2026-09-16

- **Switched from flat `french`/`moore`/`english` columns to a long-format
  schema: one row per `(src_lang, tgt_lang)` pair.** Row shape: `{id,
  src_lang, tgt_lang, source_text, target_text, is_source_orig, doc_id,
  source, laser_score?}`. Driving problem: a pairwise score (LASER cosine
  similarity) belongs to exactly *one* language pair, but the old schema put
  a French+Mooré+English dictionary triplet in one row with a single
  `laser_score` -- no way to say which pair that score was for (and in
  practice it wasn't a real pairwise score at all for that source, just a
  constant `1.0` "exact match" flag). Now a triplet becomes two rows (a
  mos-fra row and a mos-eng row) sharing one `id`, each with its own
  unambiguous score slot. `flat_rows_to_long` (flatten.py) is the single
  conversion function; both `AlignedCorpus.to_jsonl_rows()` (plain output)
  and `cli.py::_finalize_aligned` (annotated/HF output) go through it, so the
  two paths can't drift apart. The lexicon-cleaning postprocess step
  (`--split-synonyms`/`--strip-proverb-notes`) still runs on the old flat
  shape it expects (hardcodes `entry["french"]`/`entry["moore"]`); conversion
  to long-format happens only after that.
- **`is_source_orig` tracks translation direction within this corpus, not
  ultimate authorship** -- via `ORIGINAL_LANGUAGE: dict[str, str]` mapping
  source name to which language is `src_lang` (the non-translated side).
  Resolved per source by checking actual evidence, not assumption (this took
  real digging and shouldn't need re-doing):
  - **sida-bilingual-book, kade**: Mooré was translated FROM the French
    text (`src_lang="fra"`) -- but neither is the *true* original. Both
    books' own credit pages say "Traduit en français par..." *and*
    "Traduit en mooré par... GANSAONRE Guillaume" -- the real original is
    the underlying Shellbook Publishing Systems story ("Histoires de
    Poko/Kande"), in a language this corpus doesn't include (almost
    certainly English). `is_source_orig` records the practical translation
    workflow (French was the bridge/pivot text Mooré was translated from),
    not absolute authorship, which doesn't matter for this field's purpose.
  - **conseils**: French original. Evidence: the government site
    (sig.gov.bf) tags every language's PDF filename except French
    (`MOORE_-_CM...`, `FULFULDE_-_CM...` vs. untagged
    `CONSEIL_DES_MINISTRES_...`), consistent with French being Burkina
    Faso's official administrative language for government proceedings.
  - **raamde-news**: French original. Evidence: 205/429 scraped articles
    carry an explicit Mooré attribution line naming a French-language
    source -- `"Kibarã yii Faso.net"`, `"Kibarã yii Burkĩna 24"`,
    `"Kibarã yii na-zakẽ kiba-kɩtbã nengẽ"` ("news from the Presidency's
    press office"), etc. -- the same government-communication pattern as
    conseils.
  - **niggli-dictionary-mos-fra-eng, digital-postal-glossary(-term)(-definition)**:
    Mooré original. These are lexical (a Mooré headword/term with
    French/English glosses), not translated running prose -- the "original
    vs. translation" framing doesn't map onto dictionary compilation the
    same way, but Mooré is the documented/subject language.
  - `is_source_orig` is `None` for any source not in `ORIGINAL_LANGUAGE`
    (direction not yet determined) rather than a guessed default.
- **`id` carries the source document when one is known, not just a flat
  position.** Several sources already align *per unit* (FastDTW per
  page/enum for sida, per article for raamde-news, per session date for
  conseils) and concatenate the results -- the per-unit key was being
  discarded at concatenation time even though it was known a moment earlier.
  `AlignedCorpus` gained `doc_ids` (optional, same convention as `english`:
  empty or one entry per pair), wired through the three per-unit-aligned
  sources' concatenation loops in `cli.py` *and* through `_dedup_aligned`
  (which rebuilds an `AlignedCorpus` from COMET-deduplicated pairs and was
  silently dropping any per-pair metadata not explicitly carried through).
  `id` becomes `"{source}-{nth distinct doc_id}-{nth row within that doc}"`
  (e.g. `"conseils-000042-003"`) -- an ordinal, not the raw doc_id text
  (which varies wildly in shape: a date, a full URL, `"page-7"`). The raw
  value is kept separately as its own `doc_id` field for exact
  grouping/lookup. Rows without a `doc_id` (kade -- single whole-book
  alignment pass, no per-unit loop; niggli-dictionary/digital-postal-glossary
  -- no per-unit key wired up yet, though `entry.lemma`/`fr_term` are
  available and could be) keep the old flat `"{source}-{i:06d}"` scheme.
- **LASER scoring needed to become per-row-language-aware, COMET-QE
  didn't.** `score_laser.py::score_dataset` loaded one encoder pair for the
  *whole* dataset, inferred from the column name via `_FIELD_TO_LANG`
  (`"french"` -> `"fra"`, etc.) -- silently wrong for a long-format dataset
  where `source_text`/`target_text` share column names across rows with
  *different* language pairs (a trilingual entry's mos-fra row next to its
  mos-eng row). Fixed: when neither `src_lang` nor `tgt_lang` is passed
  explicitly and the dataset has `src_lang`/`tgt_lang` columns, `score_dataset`
  now groups rows by distinct pair, scores each group with its own encoder
  pair (caching loaded encoders by language), and reassembles in original row
  order. `run_laser`'s own `src_lang`/`tgt_lang` defaults had to change from
  `"fra"`/`"mos"` to `None` too -- otherwise "unspecified" never actually
  reached `score_dataset` as `None`, masked by a concrete default at the
  wrapper layer. COMET-QE (`score_comet_qe.py`) needed no change: it's one
  multilingual model with no per-language encoders, so mixed-language rows
  were never a problem for it.
- **Output splits into one file/config per language pair when more than one
  is present**, instead of one file/config a consumer has to filter first
  (the convention e.g. `opus100` uses: separate `en-fr`, `en-de`, ... HF
  configs). `AlignedCorpus.write_jsonl` and `annotate.save_data` both do
  this now -- locally as `"{stem}.{src}-{tgt}{suffix}"` files, on the Hub as
  separate `push_to_hub(..., config_name=f"{src}-{tgt}")` configs in the same
  repo. A single-pair corpus (everything except niggli-dictionary-mos-fra-eng
  when English is present) is unaffected, written as one file/the repo's
  `"default"` config. `annotate.load_data` gained a matching `config_name`
  param (`moore-web annotate -i hf://owner/repo --config mos-fra ...`) to
  read a specific pair back -- it didn't exist before and `load_dataset`
  errors on an ambiguous multi-config repo with no config specified.
- **Source names renamed to be self-describing** (source field / HF config
  basis only -- the short CLI flags `-s sida`/`-s news`/`-s simple`/`-s
  digital` are unchanged): `sida` -> `sida-bilingual-book`, `news` ->
  `raamde-news`, `simple` -> `niggli-dictionary-mos-fra-eng` (credits the
  actual compiler, Niggli), `digital` -> `digital-postal-glossary` (+
  `-term`/`-term-definition` variants). `kade` and `conseils` untouched --
  already name a specific book/body, not a category. Motivation: `"sida"`
  was actively ambiguous with `kade`, which is informally called
  `"sida-facilitateur"` elsewhere in the repo (`final_data/`,
  `data/omegat/`, `docs/kade-facilitator-manual.md`'s own text) -- two
  different pipeline sources about the same disease topic, only one of which
  owned the `"sida"` tag.
- **Open, not done**: `doc_id` for niggli-dictionary-mos-fra-eng
  (`entry.lemma` -- not unique, homonyms exist, e.g. two different dictionary
  entries both lemma `"a"` in the real PDF; would need lemma + a
  disambiguating index) and digital-postal-glossary (`fr_term` -- unique by
  construction, since it's `align_glossaries`' own join key, so safe to wire
  up as-is) aren't wired up yet. Neither is kade's (no per-unit alignment
  loop today -- would need `flatten_facilitateur_pair` restructured to
  return per-chapter units the way `flatten_sida_book_per_unit` already
  does for sida, a bigger change than the wiring done for the three sources
  above).
