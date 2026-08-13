# Digital and postal glossaries (`digital`)

Sources: the French `Lexique_de_l_economie_numerique_et_des_postes.pdf` and Mooré `Glossaire_des_termes_usuels_du_numerique_et_de_la_poste_en_Moore__valide.pdf`, parsed by `src/moore_web/glossary_parser.py`.

## Physical layout

These are table PDFs, not prose. Extract rows with ruled-line table detection; do not parse a page as a simple text stream.

```text
Mooré table (normally 4 columns)
N° | TERMES (French key) | GOM-BI-TIGSI (Mooré term) | B VÕOR WILGRI (Mooré definition)

French table (3 columns)
N° | Mots clés (French key) | Définitions
```

The Mooré table sometimes has only three columns (`N° | Mooré term | Mooré definition`), in which case the French key is absent and the row cannot be matched by the normal key join.

## Page ranges

- Mooré pages 1–3: prose introduction; segment separately. Page 4: section header; skip. Pages 5–47: tables; page 45 is an additional section header and has no usable table.
- French pages 1–4: prose introduction; segment separately. Page 5: section header; skip. Pages 6–86: tables. Pages 87–94: appendices/end matter; skip.

## Alignment key

Join a Mooré row to a French row on the French term/key after lowercasing and removing diacritics. The aligned record is:

```text
French term ↔ Mooré term
French definition ↔ Mooré definition
```

Normalize typographic quotes, dashes, ellipses, and non-breaking spaces before matching. In Mooré cells, collapse spurious whitespace immediately around hyphens in compound words.
