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

### Sources

| Source | Description |
| ------ | ----------- |
| `sida` | Bilingual SIDA book (single PDF, columns interleaved) |
| `kade` | Kadé facilitator manuals (two separate PDF/TXT files) |
| `news` | Raamde news corpus (JSON with `text_units` lists) |
| `simple` | Simple bilingual dictionary PDF |
| `conseils` | Conseil-des-ministres bilingual corpus (JSON) |

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

## TODO

- [ ] Add Dioula data and clean it
- [ ] Add Fulfulde data and clean it
- [ ] Add Gulimancema data and clean it
- [ ] Add Bissa data and clean it
