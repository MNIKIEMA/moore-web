# Mooré Web

A bilingual French/Mooré corpus pipeline: parse → flatten → align → annotate.

## Installation

### Install `uv`

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Alternative
pip install uv
```

### Install `just`

```bash
# Linux
apt install just

# Windows
winget install --id Casey.Just --exact
```

See [Just](https://github.com/casey/just) for more details.

### Install dependencies

```bash
just install
```

## Usage

The CLI is available as `moore-web` after installation.

```bash
moore-web --help
```

### Commands

| Command | Description |
| ------- | ----------- |
| `parse` | Parse source document(s) to structured JSON |
| `flatten` | Flatten parsed JSON to a sentence list |
| `parse-flat` | Parse and flatten in one step |
| `align` | Align a sentence list using LASER + FastDTW |
| `annotate` | Enrich an aligned dataset with quality signals |
| `e2e` | Full pipeline: parse → flatten → align (with optional annotation) |
| `clean-lexicon` | Clean a lexicon JSONL file (synonym splitting, proverb stripping) |
| `parse-expert-translations` | Extract existing French–Mooré PDF table pairs with QA notes |
| `parse-moore-proverbs` | Pair archived Mooré proverbs with French renderings |
| `parse-moore-tales` | Pair archived Mooré tales (volume 5) with their French translations |
| `segment-abc-coepouses` | Segment the French/Mooré abcBurkina tale into paired story units |

### Sources

| Source | Description |
| ------ | ----------- |
| `sida` | Bilingual SIDA book (single PDF, columns interleaved) |
| `kade` | Kadé facilitator manuals (two separate PDF/TXT files) |
| `news` | Raamde news corpus (JSON with `text_units` lists) |
| `simple` | Simple bilingual dictionary PDF |
| `conseils` | Conseil-des-ministres bilingual corpus (JSON) |
| `udhr` | Universal Declaration of Human Rights (French and Mooré TXT, paired by article) |

### Presidential New Year messages

Prepare the curated French–Mooré texts named by the collection manifest, then
use the existing alignment command:

```bash
moore-web prepare-new-year-message \
  --collection-dir ../faso-web-docs/messages-nouvel-an \
  --output messages-nouvel-an.parallel.json

moore-web align messages-nouvel-an.parallel.json \
  --output messages-nouvel-an.aligned.jsonl
```

Blank lines in each UTF-8 text file are preserved as manually curated segment
boundaries. PDF extraction for the other languages is handled in `faso-web`.

### Universal Declaration of Human Rights

The French and Mooré texts share the same structure, so `e2e -s udhr` pairs
them article by article and paragraph by paragraph instead of using LASER:

```bash
moore-web e2e -s udhr \
  --fr-input ../faso-web-docs/universal-declaration-human-rights/udhr-fra.txt \
  --mo-input ../faso-web-docs/universal-declaration-human-rights/udhr-mos.txt \
  --output udhr.aligned.jsonl
```

Segmentation (on by default; `--no-segment` keeps whole paragraphs) splits a
paragraph pair into sentence pairs only when both sides split into the same
number of sentences. Sections missing on one side are
skipped and reported: the Mooré translation has no article 12 (the source text
only holds an `&1` placeholder) and no closing proclamation of the preamble.

### Expert translation batch

The expert translation PDF already pairs each French source with its Mooré
translation in one table row. Extract the rows and preserve their review notes:

```bash
moore-web parse-expert-translations \
  -i '../faso-web-docs/expert-translations/mos_Mossi_seed_batch2 (3).pdf' \
  -o expert_translations.jsonl
```

See [the source format and output fields](docs/expert-translations.md).

### Mooré proverbs

The archived Proverbs Volume 1 app has a Mooré saying, a French rendering,
and a repeated Mooré reading on each page. Extract one pair per page:

```bash
moore-web parse-moore-proverbs \
  --input-dir ../faso-web-docs/mooreburkina-priority/apps/mos-proverbes-volume-1 \
  --output moore_proverbs.jsonl
```

See [the source format and pairing checks](docs/moore-proverbs.md).

### Mooré tales, volume 5

The archived Contes volume 5 app has 30 tales, each a Mooré page followed by
its French translation. Extract one record per tale, or align sentences
within each tale:

```bash
moore-web parse-moore-tales \
  --input-dir ../faso-web-docs/mooreburkina-priority/apps/mos-contes-volume-5 \
  --output moore_tales.jsonl

moore-web e2e -s moore-tales \
  -i ../faso-web-docs/mooreburkina-priority/apps/mos-contes-volume-5 \
  -o moore_tales_aligned.jsonl
```

See [the source format and alignment notes](docs/moore-tales.md).

### abcBurkina coépouses tale

The French and Mooré editions have different paragraph breaks. Segment them
into hand-anchored parallel story units, with separate sentence lists for later
alignment:

```bash
moore-web segment-abc-coepouses \
  --fr-input ../faso-web-docs/abcburkina-contes/text/266-les-couses.txt \
  --mo-input ../faso-web-docs/abcburkina-contes/text/267-les-co-epouses-moore.txt \
  --output abc_coepouses_units.jsonl
```

See [the source format and anchor list](docs/abc-coepouses.md).

### Review bilingual units in Shiny

Export source files with `scripts/export_review_units.py`, then run:

```bash
uv run shiny run apps/review_app.py --host <host> --port <port>
```

The app imports **all** `data/review/*_units.jsonl` files into
`data/review/reviews.sqlite3` and shows units in pages of 25, 50, or 100.
Filter the list by review status or source, expand or collapse visible units,
enter a reviewer name, and use **Edit / review** to change sentence boundaries
or text. **Save draft** stores work under that reviewer name; **Restore source
text** loads the original sentences into the editor; **Mark reviewed** requires
equal, nonzero French and Mooré line counts.
If another reviewer accepted a newer version, the app retains the draft and
asks for an explicit comparison before it can be accepted.

**Download aligned pairs JSONL** exports every unit with equal sentence counts,
including untouched units, matching the Marimo notebook's export behavior.
**Download reviewed only** exports accepted units. Both downloads use
`{"french": ..., "moore": ..., "source": ..., "unit": ...}` rows.

You can set `REVIEW_INPUT_DIR` and `REVIEW_DB_PATH` to use other locations.
The database imports new units on startup and preserves existing edits. Keep
the SQLite database on the app server's local disk. Reviewer names identify
drafts but do not provide authentication; use authenticated hosting before
making the app available to an untrusted audience.

### Examples

**End-to-end pipeline:**

```bash
# SIDA book
moore-web e2e -s sida -i book.pdf -o aligned.json

# Kadé manuals
moore-web e2e -s kade --fr-input fr.pdf --mo-input mo.pdf -o aligned.json

# News corpus
moore-web e2e -s news -i corpus.json -o aligned.json

# Push directly to HuggingFace with all annotations
moore-web e2e -s sida -i book.pdf -o hf://owner/repo --annotate
```

**Step by step:**

```bash
# 1. Parse
moore-web parse -s sida -i book.pdf -o parsed.json

# 2. Flatten
moore-web flatten -s sida -i parsed.json -o parallel.json

# 3. Align
moore-web align parallel.json -o aligned.json --min-laser-score 0.6
```

**Clean a lexicon JSONL file:**

```bash
# Split comma/semicolon synonym lists into one entry per FR/MOS pair
moore-web clean-lexicon -i final_data_hf/lexicon_entries.jsonl --split-synonyms

# Strip proverb annotations from french/english fields (in-place)
moore-web clean-lexicon -i final_data_hf/lexicon.jsonl --strip-proverb-notes

# Both at once
moore-web clean-lexicon -i lexicon.jsonl --split-synonyms --strip-proverb-notes
```

The `--split-synonyms` flag is also available in `e2e --source simple` to apply
synonym splitting during the pipeline:

```bash
moore-web e2e -s simple -i dict.pdf -o out.jsonl --split-synonyms --strip-proverb-notes
```

`one-column-dict` is an alias for `simple` that makes the dictionary layout explicit:

```bash
moore-web e2e -s one-column-dict -i dict.pdf -o out.jsonl
```

**Annotate an existing dataset:**

```bash
# Add specific annotations
moore-web annotate -i data.jsonl -o out.jsonl --consistency --quality-warn

# Add all annotations
moore-web annotate -i data.jsonl -o out.jsonl --all

# From/to HuggingFace
moore-web annotate -i hf://owner/src -o hf://owner/dst --all

# Custom field names with explicit LASER language codes
moore-web annotate -i data.jsonl -o out.jsonl --src en --tgt mo --laser-score
moore-web annotate -i data.jsonl -o out.jsonl --src my_col --tgt other_col --laser-score --src-lang fra_Latn --tgt-lang mos_Latn
```

**Available annotation flags:**

| Flag | Description |
| ---- | ----------- |
| `--lang-id` | GlotLID language-ID scores |
| `--consistency` | Identification consistency score |
| `--quality-warn` | Quality warnings list |
| `--laser-score` | LASER cosine similarity |
| `--comet-qe` | COMET-QE translation quality score |
| `--all` | Enable all of the above |

**LASER language codes** (`--laser-score` only):

| Flag | Description |
| ---- | ----------- |
| `--src-lang` | LASER language code for the source encoder (e.g. `fra`, `eng`, `fra_Latn`). Inferred from `--src` for known fields. |
| `--tgt-lang` | LASER language code for the target encoder (e.g. `mos`, `mos_Latn`). Inferred from `--tgt` for known fields. |

Known fields resolved automatically: `french`/`fr`/`fra` → `fra`, `english`/`en`/`eng` → `eng`, `moore`/`mo`/`mos` → `mos`. Pass `--src-lang`/`--tgt-lang` explicitly for any other field.

## Dataset builder

`build_fr_mos_dataset.py` assembles a combined French–Mooré parallel corpus from
the local moore-web files and the [`madoss/mafand-fr-mos`](https://huggingface.co/datasets/madoss/mafand-fr-mos) HuggingFace dataset.

### Local sources

| File | Source tag | Rows | Eval-eligible |
| ---- | ---------- | ----: | ------------- |
| `lexicon.jsonl` | `lexicon` | 4 249 | yes |
| `lexicon_entries.jsonl` | `lexicon_entries` | 19 793 | no (dict entries) |
| `conseils_ministres_aligned.jsonl` | `conseils` | 7 596 | yes |
| `raamde_aligned.jsonl` | `news` | 3 915 | yes |
| `sida_aligned.jsonl` | `sida` | 216 | yes |
| `sida-facilitateur_aligned.jsonl` | `kade` | 674 | yes |

Dev/test are built by stratified sampling over eval-eligible sources.
`lexicon_entries` (raw dictionary entries) stays train-only by default.
Duplicate `(french, moore)` pairs are removed globally across all files.

### Output splits

| Split | Local | mafand | Total |
| ----- | ----: | -----: | ----: |
| train | ~32 600 | 2 493 | ~35 100 |
| dev | 500 | 1 492 | ~2 000 |
| test | 500 | 1 574 | ~2 100 |

Output schema: `french | moore | source`

### Dataset builder usage

```bash
# Write train/dev/test JSONL to ./fr_mos_combined/
python build_fr_mos_dataset.py

# Local data only (no HuggingFace download)
python build_fr_mos_dataset.py --no-mafand --output-dir out/

# Larger eval sets
python build_fr_mos_dataset.py --dev-size 1000 --test-size 1000

# Push to HuggingFace Hub
python build_fr_mos_dataset.py --push-to-hub owner/fr-mos-combined

# Keep lexicon_entries in eval too
python build_fr_mos_dataset.py --train-only-sources ""
```

## TODO

- [ ] Add Dioula data and clean it
- [ ] Add Fulfulde data and clean it
- [ ] Add Gulimancema data and clean it
- [ ] Add Bissa data and clean it
