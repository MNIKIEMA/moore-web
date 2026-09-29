# HPLT Mooré monolingual text (`hplt_mono.py`, `mono_publish.py`)

`madoss/mos-latn-hplt`: 1,892 web documents from HPLT v3 (`mos_Latn`,
Common Crawl), downloaded with `download_hplt.sh` and pushed with
`create_hf_dataset_hplt.py`. Mooré only, no French. Cleaned here into
sentences for backtranslation (Mooré → French) in `mt-training`, and
published as `madoss/moore-web-mono`.

## Current design (keep up to date; dated entries below are history)

**Run:**

```bash
uv run moore-web hplt-mono                      # → data/mono/hplt_mos_wikipedia.jsonl
uv run moore-web publish-mono data/mono/*.jsonl --push -m "…"   # private by default
```

**Pipeline, in order** (`clean_sentences`; defaults are CLI options):

| # | Step | Default | Why |
| --- | --- | --- | --- |
| 1 | Keep documents by host | `wikipedia.org`, `incubator.wikimedia.org`, `incubator.m.wikimedia.org` | Only source with clear reuse terms (CC BY-SA 4.0) and general domain; jw.org (68% of the crawl) forbids reuse |
| 1b | Drop documents with leaked NLLB language tags | `xxx_Latn`, `xxx_Arab`, … anywhere in the page | "… mos_Latnmos_Latn be be be": the page was machine-translated with NLLB; training NLLB on its own Mooré teaches it nothing (H2 in mt-training) |
| 2 | Strip citation markers per line, then split with `segment_mo` | `[1]`, `[ 3]`, `[a]`, `[note 2]`, `[DM 1]`, and `3]` with a lost `[` | Stripping first: syntok does not split `ye.[1] A…` after the full stop |
| 3 | GlotLID says `mos_Latn` | – | Drops reference lists, English captions, citations |
| 4 | GlotLID probability | ≥ 0.8 | Conservative: kept Mooré scores 1.0 at the 10th percentile, so this only drops uncertain lines |
| 5 | No other script, no IPA | any non-Latin letter, or IPA/modifier letters except ɛ ɔ ɩ ʋ ə | Glosses like "(Korean: 연등회)", "[kəsˈteʎ]"; dropped whole, stripping leaves broken parentheses |
| 5b | No wiki markup | `[[`, `]]`, `{{`, `}}`, `\|thumb`, `File:` | Raw markup and English image captions left by the extraction |
| 6 | Fewer than 2 lowercase tone-accented words | ó ò ú ù í ì á ǎ ě ǐ ǒ ǔ ā ē ī ō ū û | Other spelling systems or languages; lowercase only, so foreign names (Martínez) don't count |
| 7 | Length | 4+ words, ≤ 500 chars | Drops fragments and run-ons; 500 Mooré chars ≈ 165–200 NLLB tokens (0.33 tokens/char on these sentences, 0.40 on the parallel data), under the 256-token training limit |
| 8 | Not a generation loop | 8+ word sentences need ≥ 55% distinct words (`--min-distinct-ratio`) | Machine-translation loops ("b sẽn yaa b sẽn yaa …") |
| 9 | Deduplicate | normalized text (NFKC, lowercase, punctuation removed, spaces collapsed) | Same normalization as the id, so ids are unique |
| 10 | Not in excluded texts | FLORES+ `mos_Latn` dev/devtest, Bouquet fra-mos targets (all levels/splits), Mooré side of `moore-web-parallel` v1.0.0 | Keeps eval sets clean; **exact (normalized) matches only**, so near-duplicates can remain |

**Row fields:** `id` (`hplt-` + 16 hex chars of the SHA-1 of the normalized
text: content-based, stable across re-runs and re-translations), `text`,
`source` (`wikipedia`), `license` (`CC-BY-SA-4.0`), `doc_id` (HPLT
document), `url` (attribution), `line` (paragraph index in the page),
`lang_prob`, `words`. Rows are in page, line, sentence order, so `doc_id` +
`line` rebuild paragraphs.

**Publishing (`madoss/moore-web-mono`):** one parquet folder per source
(`data/<source>/train.parquet`); configs `default` (every source) and one per
source, because sources carry different licenses; `source` and `license` on
every row so terms travel after mixing. To add a source: produce its JSONL
with the same core fields, add it to `mono_publish.SOURCES` (card text,
license), and run `publish-mono` with all JSONL files; unknown sources,
license mismatches and duplicate ids are rejected. Tag each release
(`v1.0.0` = Wikipedia, 9,614 sentences; `v1.1.0` = NLLB-translated pages and
wiki markup removed, 8,817) and pin the tag downstream.

**Known limits:** exact-match decontamination only; machine-translated
articles are only caught when they leaked NLLB tags or produce loop
sentences; spelling conventions vary across volunteer-written articles.

## 2026-09-29 (NLLB-translated pages, v1.1.0)

- **Found while testing backtranslation in mt-training**: 25 sentences on 10
  incubator pages ended with leaked NLLB language tags ("Karen-biisã …
  wakat fãa. mos_Latnmos_Latnmos_Latn be be be"). Those articles were
  machine-translated with NLLB. Decided with the user: drop tagged pages
  whole (not only the tagged sentences), since the whole page is MT output.
  At page level, **12 pages** carry tags (the 10 were those whose tagged
  sentences survived the other filters). Loop-heavy pages without tags are
  kept (option not chosen).
- **Wiki markup filter** added: one sentence carried
  `[[File:…|thumb|right|<English caption>]]`; cheap rule, useful for future
  sources.
- **Counts**: 186 documents (12 dropped) → 13,214 sentences → … → 9,648 no
  markup → 9,079 loops → **8,817** after dedup and exclusion.
- **`madoss/moore-web-mono` v1.1.0** (private), commit `a5d76ea`: 8,817
  sentences. v1.0.0 (9,614) stays available; backtranslation should use
  v1.1.0.

## 2026-09-29 (quality check, v1.0.0)

- **Sample reviewed by the user** (`data/mono/review_sample_50.csv`: 30
  random sentences, `random.seed(0)`, + the 20 others with the lowest share of
  words found in the `mos_Latn` list of `madoss/mos-eng-fra-wordlists`, ties
  broken by highest French/English share; tokens = letter runs incl. `-`/`'`).
  Built with a one-off script, not in the repo. Almost no French/English mixed in (2
  sentences ≥ 30% fr/en words); median 61% of words in the 10k-word Mooré
  list (low share usually means names or rare words, not bad Mooré).
  Confirmed problems, each now a filter:
  - **Other scripts and IPA** (231 sentences): glosses like "(Korean:
    연등회)", "[kəsˈteʎ]". Dropped whole: stripping them leaves broken
    parentheses. `ə` is allowed (Azerbaijani names such as Rövşən).
  - **Tone-accented spelling or another language** (24): "À galóùnting (vil)
    kãsngã lã Wùgdgù", "A goùnbga noûg … sì lœtin nē …". Rule: 2+ lowercase
    words with ó ù ī ǔ … Counting all accented words would also drop good
    Mooré full of Spanish names (Martínez, Román, Díaz), hence lowercase only.
  - **Citation leftovers** (49 → 1): `[ 3]`, `[DM 1]`, and `3] …` where the
    crawl lost the `[`. Stripped per line before splitting.
  - **Machine-translation loops** (551): "b sẽn yaa b sẽn yaa b to wã …",
    "n tõog n tõog n tall n tall". Rule: 8+ word sentences with < 55%
    distinct words (user's choice; < 0.5 caught 248, the 0.50–0.55 band was
    still mostly loops). A repeated word 3-gram was too broad (21.6%:
    Mooré repeats "b sẽn da", "tẽnga taoor soab"). Some incubator articles
    look machine-translated as a whole (Kashan_rug: 629 sentences, a third
    repeating phrases); page-level filtering was not chosen.
- **Final: 9,614 sentences** (1.14 M chars before the loop filter), all ids
  unique. Steps: 14,379 split → 10,864 GlotLID mos → 10,767 prob ≥ 0.8 →
  10,536 no foreign script/IPA → 10,512 tone rule → 10,462 length → 9,911
  loops → 9,614 dedup → 9,614 not in eval refs / parallel Mooré.
- **`madoss/moore-web-mono` v1.0.0** (private), commit `385b8f3`, tag
  `v1.0.0`. Pin it for backtranslation.

## 2026-09-29

- **What the crawl is.** 11.6 M chars, all documents `filter=keep`, doc
  language `mos_Latn` with probability 1.0; 91% of HPLT's segments are Mooré.
  By host: jw.org 1,258 docs / 7.94 M chars (68%), Wikipedia (incubator +
  mos) 210 / 1.45 M, other sites 82 / 1.33 M, raamde-bf.com 181 / 0.41 M
  (articles we already have in parallel), Bible sites 110 / 0.28 M, Islamic
  sites 51 / 0.22 M. ~10% of lines are exact duplicates.
- **Only Wikipedia is used** (decided with the user). jw.org's terms of use
  forbid this reuse (the reason JW300 was withdrawn); Bible pages may hold
  the hidden references of `burkimbia/mt-benchmark-public`'s religious
  domain; the rest is small and mixed. Wikipedia is general-domain (what
  FLORES+ needs) and CC BY-SA 4.0: keep the URL for attribution, and
  share-alike applies to anything published from it. Note that
  `madoss/mos-latn-hplt` itself is public and redistributes the jw.org text.
- **Pipeline** (`moore-web hplt-mono`, defaults): hosts `wikipedia.org`,
  `incubator.wikimedia.org`, `incubator.m.wikimedia.org` (meta, wikimania
  and a fan wiki excluded) → strip citation markers (`[1]`, `[a]`) from each
  line → `segment_mo` → GlotLID `mos_Latn` with prob ≥ 0.8 → 4+ words,
  ≤ 500 chars → dedup (normalized) → drop sentences found in FLORES+
  `mos_Latn` dev/devtest, Bouquet fra-mos targets, or the Mooré side of
  `moore-web-parallel` v1.0.0.

  | Step | Sentences |
  | --- | ---: |
  | split (198 documents) | 14,410 |
  | GlotLID `mos_Latn` | 10,856 |
  | prob ≥ 0.8 | 10,759 |
  | 4+ words, ≤ 500 chars | 10,708 |
  | deduplicated | 10,404 |
  | not in eval refs / parallel Mooré | 10,404 |

  Output: `data/mono/hplt_mos_wikipedia.jsonl` (`id`, `doc_id`, `url`,
  `line`, `text`, `lang_prob`, `words`): 10,404 sentences, 197 pages,
  1.17 M chars, median 21 words.
- **`id` is content-based**: `hplt-` + the first 16 hex chars of the SHA-1 of
  the normalized text. It survives re-runs, re-splitting of other lines and
  re-translation with another model (a position-based id would shift: the
  citation-marker fix alone moved ~270 sentences), and it is unique because
  dedup uses the same normalization. Backtranslated pairs should keep it, so
  French from different models can be compared on the same sentences.
- **Strip citation markers before splitting**: 12% of sentences had `[n]`
  markers, and `ye.[1] A…` is not split after the full stop by syntok.
  Stripping per line first raised the count from 10,136 to 10,404.
- **What gets dropped** by GlotLID is mostly reference lists, English
  captions and citations ("Retrieved March 28, 2017"); GlotLID is confident
  on the kept Mooré (prob quantiles 1.0).
- **Published as `madoss/moore-web-mono` (private)**, commit `34aebba`, with
  `moore-web publish-mono <jsonl…> --push` (`mono_publish.py`). Layout: one
  parquet folder per source (`data/<source>/train.parquet`); card configs
  `default` (every source) and one per source (`wikipedia`), because sources
  will carry different licenses. Every row has `source` and `license`
  (`wikipedia`, `CC-BY-SA-4.0`), so terms travel with rows after mixing.
  Adding a source: its JSONL (same core fields), an entry in
  `mono_publish.SOURCES` (card text, license), then `publish-mono` with all
  the JSONL files; `group_by_source` rejects unknown sources, license
  mismatches and duplicate ids. (Tagged `v1.0.0` after the quality check,
  see above.)
- **Not checked yet:** quality of the incubator Mooré (written by
  volunteers; some articles may be machine-translated), and near-duplicates.
  Sample some sentences before scaling up. (Sample checked the same day:
  see the quality-check entry above.)
