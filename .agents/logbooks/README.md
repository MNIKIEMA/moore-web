# Logbooks

These track how the pipeline evolved: decisions made, what real data revealed,
fixes applied, and limitations left open. The goal is that the next session
(human or agent) doesn't have to re-derive a quirk we already found.

## Convention: one logbook per parser/source

A logbook maps 1:1 to a source document family or pipeline module. Findings
and fixes are driven by *document format*, not by language -- a fix in
`parse_du_moore.py` typically applies to all three volumes simultaneously.

- [`du-moore-livre-de-lecture.md`](du-moore-livre-de-lecture.md) -- `parse_du_moore.py` (three-volume bilingual reading textbook).
- [`sida-bilingual-book.md`](sida-bilingual-book.md) -- `book_parser.py` / `flatten.py::flatten_sida_book*` (two-column HIV/AIDS awareness book).
- [`kade-facilitateur-book.md`](kade-facilitateur-book.md) -- `book_parser_facilitateur.py` / `flatten.py::flatten_facilitateur_pair` (two monolingual HIV/AIDS facilitator manuals, aka "sida-facilitateur").
- [`udhr.md`](udhr.md) -- `udhr.py` / `cli.py::e2e -s udhr` (Universal Declaration of Human Rights, paired by article).
- [`messages-nouvel-an.md`](messages-nouvel-an.md) -- `new_year_message.py` (presidential New Year address, curated text files).
- [`moore-tales.md`](moore-tales.md) -- `moore_tales_parser.py` / `cli.py::e2e -s moore-tales` (mooreburkina.com Contes volume 5, 30 tales, Mooré page + French page).
- [`raamde-news.md`](raamde-news.md) -- `segment_news_data.py` / `flatten.py::flatten_news_per_entry` / `scripts/compare_raamde_splitters.py` / `scripts/align_raamde_sat.py` (raamde-bf.net bilingual news, summary-style Mooré).
- [`expert-translations.md`](expert-translations.md) -- `expert_translation_parser.py` / `moore-web parse-expert-translations` (expert-translated seed batch PDF, already paired).
- [`hf-output-schema.md`](hf-output-schema.md) -- `flatten.py::AlignedCorpus`/`flat_rows_to_long`, `annotate.py`, `cli.py::_finalize_aligned` (the shared corpus-output layer: long-format row schema, `is_source_orig`/`ORIGINAL_LANGUAGE`, `doc_id`, per-language-pair file/config splitting).

Add a new logbook when a new source or parser module is added, not a new language.

## Language coverage index

| Language | ISO | Logbooks |
| --- | --- | --- |
| Mooré | `mos` | du-moore-livre-de-lecture, moore-tales, sida-bilingual-book, kade-facilitateur-book, udhr, messages-nouvel-an, raamde-news, expert-translations, hf-output-schema |
| French | `fra` | du-moore-livre-de-lecture, moore-tales, sida-bilingual-book, kade-facilitateur-book, udhr, messages-nouvel-an, raamde-news, expert-translations, hf-output-schema |

## Entry format

Dated, terse, newest entry on top. Prefer "what we learned / decided" over
"what the diff was" -- `git log` already has the diff.

```markdown
## 2026-09-14

- Finding or decision, one or two sentences.
- Why it matters / what to check next.
```
