# Du Moore au Français literacy series

Parser: `src/moore_web/parse_du_moore.py`.

Sources:

```text
Du_Moore_au_Francais_1_Noir_et_Blanc_pp_01-30_Lecons_1-16.pdf
Du_Moore_au_Francais_2_Noir_et_Blanc_pp_31-60_Lecons_17-31.pdf
Du_Moore_au_Francais_3_Noir_et_Blanc_pp.61-94_Lecons_32-48.pdf
```

Output: `du_moore_parallel.jsonl`, one JSON object per French–Mooré pair.

## Page pairing

Each lesson consists of two consecutive pages with the same `Kaoreng … soaba` heading:

```text
French lesson page: Kaoreng <n> soaba
Mooré lesson page:  Kaoreng <n> soaba
```

The parser reads every page, detects this heading, and pairs adjacent pages with the same printed lesson number `(<n>)`. Matching the whole header text dropped lessons 3 and 21, whose two pages carry header typos (`kaoreng`/`Kaoreng`, `pisi a la ye`/`pisi la a ye`). Lessons 1–2 are alphabet drills with no parallel text. It does not use hard-coded page indices. The first page in a matched pair is French and the second is Mooré.

## Output record

```json
{
  "fr": "Fabéré pilera le mil.",
  "mos": "a Fabeere na n too ki wã.",
  "source": "Du_Moore_1",
  "lesson": 4,
  "section": "key"
}
```

`lesson` is the printed lesson number (1–48 across the three books). `item` is omitted for the lesson key. `section` is one of `key`, `vocab` or `sentences`. Passages are not written to this file: they are free translations and even equal sentence counts misalign (lesson 48), so they are only paired through the review app.

## Layout 1: sectioned lessons

Books 1–2 and early Book 3 use section markers:

```text
Kaoreng … soaba
<subtitle>                         # skip
<key sentence>

① 1 – vocabulary item ...
  2 – vocabulary item ...

② sentence 1
  sentence 2

D gom fãrende / Kʋmbɡo
<conversation or drill lines>
```

Extract the sections in this order:

1. **Key** — first sentence-like line after the heading and subtitle.
2. **Vocabulary** — numbered `N - value` or `N – value` items between `①` and `②`; pair matching numbers only.
3. **Sentences** — lines after `②` until a conversation or free-expression heading.
4. **Conversation** — text under `D gom fãrende` or `Kʋmbɡo`, stopping at writing/free-expression headings.

## Layout 2: prose lessons

Book 3 lessons 40–48 omit `①` and `②`:

```text
Kaoreng … soaba
<subtitle>                         # skip
<key sentence>

1 - vocabulary item
2 - vocabulary item

<reading passage sentence, possibly wrapped>
Questions de compréhension
```

Extract numbered vocabulary first. The first long non-numbered line afterwards starts the reading passage. Stop at `Questions de compréhension`, `Ecriture`, `Copie`, and typographic variants. Passage lines are rebuilt into sentences like section ② and exported for review only.

## Regex and extraction safeguards

- Lesson heading: match `Kaoreng`/`kaoreng` followed later by `soaba`.
- Subtitles: skip lines matching `kaorengo`, `kaorenɡo`, or `karem`. If no explicit subtitle match is found, skip the first non-short line after the heading positionally.
- Numbered vocabulary: split using a captured separator (`(\d+)\s*[–-]\s*`), not repeated `finditer`; this prevents a two-digit number such as `17` from being partly consumed.
- Ignore an empty vocabulary slot: `N –` without a non-numeric value creates no record.
- Group words into lines by chaining bottom edges (≤ 5 px step between neighbours), not by snapping `top` to a fixed grid: drop caps, bold names and item numbers sit a few px off the text they belong to, and a grid split those lines in two.
- A drop cap is a capital more than 1.15× taller than the following non-capital word, within 8 px. pdfplumber leaves it alone (`C écile`), glues it to the previous token (`deC éline`, `–C écile`) or glues it on both sides (`voisineCaroline`); all become `Cécile` / `de Céline` / `voisine Caroline`. This is geometric, so Mooré's one-letter words (`A`, `B`) are never glued.
- Section ② lines are rebuilt into sentences before pairing: a line without terminal punctuation continues on the next, and a line holding several sentences is split at `. X`. French and Mooré wrap at different points, so pairing raw lines shifts every later pair.
- If a lesson's French and Mooré ② sentence counts still differ, skip that lesson's sentences (with a warning) instead of zipping. The four remaining cases are real text differences (A/B dialogue lines, one sentence rendered as two, an extra or unpunctuated sentence).
- The subtitle is skipped positionally only when it does not end with sentence punctuation; lesson 47's Mooré page has no subtitle and its key sentence used to be skipped.

## Reproducibility

Run from the repository root:

```bash
uv run python src/moore_web/parse_du_moore.py --output du_moore_parallel.jsonl
```

The current three PDFs produce **834** pairs (key 46, vocab 574, sentences 214):

| Book | Pairs |
| --- | ---: |
| 1 | 264 |
| 2 | 299 |
| 3 | 271 |
| Total | 834 |

## Review units

Everything, including what the JSONL leaves out, is exported for the review app, one unit per lesson section:

```bash
uv run python scripts/export_review_units.py --source du-moore \
    --input ../faso-web-docs/du-moore-literacy-series \
    -o data/review/du-moore_units.jsonl
```

Unit ids are `du-moore-<lesson>-<section>` (`du-moore-30-vocab`, `du-moore-45-passage`), built from the printed lesson number so they stay stable across re-exports. Sections are separate units so a mismatch in one (e.g. a missing vocab item) doesn't shift the rows of another. Vocab rows matched by item number come first; items found on one side only are appended at the end of that side. The three PDFs give 138 units (46 key, 46 vocab, 37 sentences, 9 passage), 12 of them with uneven sides.
