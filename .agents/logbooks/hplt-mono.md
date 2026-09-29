# HPLT Mooré monolingual text (`hplt_mono.py`, `moore-web hplt-mono`)

`madoss/mos-latn-hplt`: 1,892 web documents from HPLT v3 (`mos_Latn`,
Common Crawl), downloaded with `download_hplt.sh` and pushed with
`create_hf_dataset_hplt.py`. Mooré only, no French. Cleaned here into
sentences for backtranslation (Mooré → French) in `mt-training`.

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
  mismatches and duplicate ids. Not tagged yet.
- **Not checked yet:** quality of the incubator Mooré (written by
  volunteers; some articles may be machine-translated), and near-duplicates.
  Sample some sentences before scaling up.
