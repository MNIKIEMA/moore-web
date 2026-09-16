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

Add a new logbook when a new source or parser module is added, not a new language.

## Language coverage index

| Language | ISO | Logbooks |
| --- | --- | --- |
| Mooré | `mos` | du-moore-livre-de-lecture, sida-bilingual-book, kade-facilitateur-book |
| French | `fra` | du-moore-livre-de-lecture, sida-bilingual-book, kade-facilitateur-book |

## Entry format

Dated, terse, newest entry on top. Prefer "what we learned / decided" over
"what the diff was" -- `git log` already has the diff.

```markdown
## 2026-09-14

- Finding or decision, one or two sentences.
- Why it matters / what to check next.
```
