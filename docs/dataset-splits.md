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

Sizes: dev and test target 1 000 local rows each (the build default since
2026-09-26, was 500), spread over many documents; external sets add about
1 000 more (FLORES+ devtest has 1 012).

## Applied to this dataset

1. Row metadata: `id`, `original_lang`, `doc_id`, `reviewed` (done, see
   below).
2. Build dev/test once from reviewed and expert data, by document, with a
   per-source cap; store the ids in a pinned file.
3. Train on everything else, minus any sentence that occurs in dev/test.
4. Evaluate on the frozen test, FLORES+ fra–mos and Bouquet fra–mos,
   reported per domain.

Open questions: which sources may enter dev/test; the per-source cap; whether
high-score conseils sessions qualify; how the frozen id file is versioned.

## Row metadata

Each dataset row carries:

| Column | Meaning |
| --- | --- |
| `id` | Stable row id: the upstream id when there is one (`conseils-000042-003`), `{source}-{unit}-{line}` for review-app rows, otherwise `{source}-` + a hash of the French and Mooré text. Unchanged across rebuilds. |
| `original_lang` | Language the pair was translated *from* (`fra` or `mos`), `null` when unknown (e.g. `udhr`: the Mooré may come from the French or the English text). MAFAND is `fra`: news is written in French, no media publish original articles in Mooré. |
| `doc_id` | Document the pair comes from (session date, article URL, review unit, PDF); `null` for dictionaries. The unit a document-level split works on. |
| `reviewed` | `true` when a human produced or checked the pair (review-app units, expert translations, MAFAND), `false` for automatic alignments. |
