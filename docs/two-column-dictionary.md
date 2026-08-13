# Two-column dictionary

Source: dictionary PDFs handled by `src/moore_web/bicolumns_parser.py` (legacy parser and useful format reference).

## Physical layout

Each page has two vertical columns. Read all blocks in the left half from top to bottom, then all blocks in the right half from top to bottom. Ignore page numbers and date/header/footer blocks before entry parsing.

```text
PAGE
  LEFT COLUMN:  entry A ... entry B (possibly unfinished)
  RIGHT COLUMN: completion of B ... entry C
```

Never read across rows (`left line 1`, `right line 1`, ...): that interleaves unrelated entries. A later parser should split entries per column before reconciling a cross-column continuation.

## Entry grammar

```text
headword[homonym] [tone] POS.
French gloss; English gloss.
Mooré example.
French translation.
English translation.
```

The POS period is the practical entry/sense delimiter. Known POS tokens include `Verbe.`, `Pronom.`, `Nom.`, `n.pl.`, `n.propre.`, `v.inaccompli.`, `v.`, `expression.`, `interj.`, `particule grammaticale.`, `préfixe.`, `Adverbe.`, `auxiliaire.`, `Adjectif.`, `conjonction.`, `indéfinie.`, `démonstratif.`, `interrogatif.`, `Déterminant.`, and `postposition.`

## Sense and example rules

- A sense is generally `French; English`.
- Multiple senses can be numbered: `1 - ...`, `2 - ...`.
- Examples are ordered Mooré → French → English, but the English line may be absent.
- Proverb explanations (`Proverbe:` / `proverbe indiquant:`) are annotations, not translations.
- Scientific names can be a semicolon-separated sequence and must stay in the same field.

## Boundary hazards

- The final visible text in a column can be the start or end of an entry in the other column.
- A bare headword plus optional `[tone]` after final sentence punctuation may be the next entry, rather than definition text.
- Question-mark-only examples are valid; do not rely on the number of `.` characters.
