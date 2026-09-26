# Dataset split strategy (proposal)

Status: **proposal, not implemented.** `build_fr_mos_dataset.py` still draws
dev/test at random per source (`_stratified_split`). This note records the
intended practice for the French–Mooré dataset; the split itself is still to
be decided.

## Why the current split misleads

The build shuffles each source's rows and takes a proportional share for dev
and test. Two problems follow:

- **Same document on both sides.** A Council of Ministers session has about
  140 pairs; a random split puts ~135 in train and a few in dev/test. Test
  sentences then share names, topics, boilerplate and neighbouring sentences
  with training data, so test scores overestimate how well a model translates
  a new document.
- **One source dominates.** Shares follow source size: after the conseils
  re-alignment, conseils was 310 of the 500 dev and test rows (62 %), so
  scores mostly measure government-communiqué translation.

## Practice

1. **Split by document, never by row.** Every pair from one document
   (`doc_id`: session date, article URL, tale, review unit, page) goes to a
   single split. Sources without documents (dictionaries, glossaries) stay
   train-only.
2. **Dev/test from human-quality data.** A wrong reference penalises correct
   output. Candidates: reviewed review-app units (du-moore, kade, sida, udhr,
   tales), expert translations, and possibly spot-checked high-score conseils
   sessions. Automatically aligned sources (raamde, most of conseils) stay in
   train.
3. **Balance domains and report per domain.** Cap each source's share of
   dev/test, and report scores per domain (administration, news, stories,
   health, education) as well as overall.
4. **Add external benchmarks and keep them out of training.** FLORES+
   (`mos_Latn`), Bouquet `fra-mos` (no overlap with the expert batch; see
   `.agents/logbooks/expert-translations.md`) and the MAFAND test split make
   results comparable with other work. Check that no dev/test sentence occurs
   in train, exactly or after normalising case, punctuation and whitespace.
5. **Freeze dev/test, then grow train.** Choose dev/test once and store their
   `id`s in a pinned file, like the reviewed export in `fr_mos_sources.toml`.
   New data only goes to train; dev/test change deliberately, with a new
   version, so scores stay comparable across dataset versions.
6. **Keep the translation direction.** Record `original_lang` and evaluate
   fra→mos and mos→fra separately; where possible test on text originally
   written in the source language, since translated source text
   ("translationese") gives optimistic scores.

Sizes: the current random split defaults to 1 000 dev and 1 000 test rows,
but that doesn't carry over: human-quality data is only about 3 100 rows (see
below), so the local sets will be smaller and the external benchmarks carry
most of the evaluation. Once dev/test are frozen, their size is fixed when the
id list is made, not by `--dev-size`/`--test-size` on every build.

## Decisions (2026-09-26)

- **Small local test, Bouquet as the main external benchmark.** Local dev
  roughly 300–500 and test roughly 500–800 rows, for an in-domain view;
  Bouquet `fra-mos` (1 358 sentences: dev 504, test 854; human-translated,
  independent of our data) carries the main comparison, with FLORES+ as a
  second external set. External sets are never used for training.
- **Unreviewed sources stay in train, not in dev/test.** conseils, raamde,
  the lexicon and the digital glossary make up about 35 000 of 38 400 rows
  and train fine with their per-source filters, but an automatic reference
  can be a wrong translation (64 % of the old raamde pairs were), which makes
  scores meaningless.
- **Cover their domains by reviewing whole documents.** To get
  administration and news into dev/test, review a few whole documents in the
  review app: 3–5 conseils sessions from different periods (high-score ones
  review fastest) and 10–20 raamde articles (already imported; expect to
  reject lines, the Mooré is summary-style). Once accepted they are reviewed
  data and can enter dev/test by document. Dictionaries and glossaries stay
  train-only.

| Source | Train | Local dev/test |
| --- | --- | --- |
| Reviewed (du-moore, kade, tales, sida, udhr, …) | yes | yes |
| Expert translations | yes | yes (one PDF: goes to a single side) |
| conseils, raamde (not reviewed) | yes, filtered | only documents reviewed in the app |
| Lexicon, digital glossary | yes | no |
| Bouquet, FLORES+ | never | external test |

Human-quality rows available today (reviewed export + expert): du-moore 936,
kade 797, tales 748, expert 347, sida 140, udhr 66, messages-nouvel-an 51,
abcburkina-contes 47, about 3 130 in total.

## Dev and test stay fixed across releases

Both are frozen by `id` and change only deliberately, as a major version:

- **A seed doesn't freeze a split.** The build shuffles each source's rows and
  cuts; adding data changes the shuffled list, so the same seed (42) picks
  different rows. Re-seeding per release means a new dev every release.
- **Comparability.** Dev picks checkpoints and settings; if it changes
  between releases, a score change can't be told apart from a data change.
- **Leakage.** Rows in one release's train could land in the next release's
  dev, so older models would be scored on sentences they trained on.

| Event | dev/test | Version |
| --- | --- | --- |
| New data (sessions, reviewed units) | unchanged; new data goes to train | minor |
| A dev/test pair found wrong, removed or fixed | edited in place, noted in the changelog | patch |
| Enlarging or redesigning dev/test (e.g. adding newly reviewed conseils and raamde documents) | new frozen id list | major |

Newly reviewed documents can be set aside as candidates for the next major
dev/test instead of going straight into train, so the next version has
material no earlier model trained on. Dev gets used often during development
and slowly overfits, which is another reason to renew it at major versions;
test is only read for final results, and Bouquet and FLORES+ never change.

## Applied to this dataset

1. Row metadata: `id`, `original_lang`, `doc_id`, `reviewed` (done, see
   below).
2. Review a few conseils sessions and raamde articles so those domains have
   human-checked documents.
3. Build dev/test once from reviewed and expert data, by document, with a
   per-source cap; store the ids in a pinned file.
4. Train on everything else, minus any sentence that occurs in dev/test.
5. Evaluate on the frozen local test, Bouquet fra–mos and FLORES+ fra–mos,
   reported per domain.

Open questions: exact local sizes; the per-source cap; which conseils sessions
and raamde articles to review; how the frozen id file is versioned.

## Row metadata

Each dataset row carries:

| Column | Meaning |
| --- | --- |
| `id` | Stable row id: the upstream id when there is one (`conseils-000042-003`), `{source}-{unit}-{line}` for review-app rows, otherwise `{source}-` + a hash of the French and Mooré text. Unchanged across rebuilds. |
| `original_lang` | Language the pair was translated *from* (`fra` or `mos`), `null` when unknown (e.g. `udhr`: the Mooré may come from the French or the English text). MAFAND is `fra`: news is written in French, no media publish original articles in Mooré. |
| `doc_id` | Document the pair comes from (session date, article URL, review unit, PDF); `null` for dictionaries. The unit a document-level split works on. |
| `reviewed` | `true` when a human produced or checked the pair (review-app units, expert translations, MAFAND), `false` for automatic alignments. |
