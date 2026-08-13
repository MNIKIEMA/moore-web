# Parsed document formats

These notes describe the *input structure* used by the parsers. They are intentionally concrete: headings, separators, ordering, page boundaries, and exceptions are recorded so a regex-based parser can be rebuilt without reverse-engineering the PDFs again.

Historical fixes and their PDF evidence: [resolved issues](resolved-issues.md).

| Format | Parser/source | Notes |
| --- | --- | --- |
| [One-column dictionary](one-column-dictionary.md) | `simple` | Block-ordered dictionary entries |
| [Two-column dictionary](two-column-dictionary.md) | legacy `bicolumns_parser.py` | Read the left column before the right column |
| [SIDA bilingual book](sida-bilingual-book.md) | `sida` | One PDF; Mooré left, French right |
| [Kadé facilitator manual](kade-facilitator-manual.md) | `kade` | Separate French and Mooré manuals |
| [Raamde news](raamde-news.md) | `news` | JSON articles with an interleaved language boundary |
| [Council of Ministers](council-of-ministers.md) | `conseils` | Same-date PDF/JSON documents in several languages |
| [Digital glossaries](digital-glossaries.md) | `digital` | Two table-based PDFs joined by normalized French terms |

## Common extraction contract

For PDF sources, preserve text-block order, retain newlines until structural splitting is complete, normalize Unicode to NFC, and keep an explicit page number while parsing. Do not globally collapse whitespace before matching headings or entries: line starts and blank lines are significant in several formats.
