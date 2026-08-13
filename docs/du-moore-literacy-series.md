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

The parser reads every page, detects this heading, and pairs only adjacent pages whose normalized headers match exactly. It does not use hard-coded page indices. The first page in a matched pair is French and the second is Mooré.

## Output record

```json
{
  "fr": "Fabéré pilera le mil.",
  "mos": "a Fabeere na n too ki wã.",
  "source": "Du_Moore_1",
  "lesson": 1,
  "section": "key",
  "item": 1
}
```

`item` is omitted for the lesson key. `section` is one of `key`, `vocab`, `sentences`, `conversation`, or `passage`.

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

Extract numbered vocabulary first. The first long non-numbered line afterwards starts the reading passage. Stop at `Questions de compréhension`, `Ecriture`, `Copie`, and typographic variants. Pair passage sentences by order.

## Regex and extraction safeguards

- Lesson heading: match `Kaoreng`/`kaoreng` followed later by `soaba`.
- Subtitles: skip lines matching `kaorengo`, `kaorenɡo`, or `karem`. If no explicit subtitle match is found, skip the first non-short line after the heading positionally.
- Numbered vocabulary: split using a captured separator (`(\d+)\s*[–-]\s*`), not repeated `finditer`; this prevents a two-digit number such as `17` from being partly consumed.
- Ignore an empty vocabulary slot: `N –` without a non-numeric value creates no record.
- A one-to-four-letter line in a nearby y-bucket is a drop cap. Reattach it to the following sentence; for French, repair the artificial space (`L e bébé` → `Le bébé`).
- In prose passages, append a following line to the preceding record if that record has no `.`, `!`, or `?` terminator. This restores PDF-wrapped sentences.

## Reproducibility

Run from the repository root:

```bash
uv run python src/moore_web/parse_du_moore.py --output du_moore_parallel.jsonl
```

The current three PDFs produce **941** pairs:

| Book | Pairs |
| --- | ---: |
| 1 | 249 |
| 2 | 341 |
| 3 | 351 |
| Total | 941 |

The checked-in `du_moore_parallel.jsonl` was verified to match a fresh run byte-for-byte.
