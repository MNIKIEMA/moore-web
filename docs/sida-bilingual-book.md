# SIDA bilingual book (`sida`)

Source: `data/2 SIDA mooré - français.pdf` and `src/moore_web/book_parser.py`.

## Physical layout

One PDF contains both languages on every content page:

```text
| Mooré column (left) | French column (right) |
```

Find the central vertical rule when present; otherwise use the page midpoint. Assign every text block by its left x-coordinate, then join each language column independently. Ignore isolated numeric page numbers. Content pages currently end at PDF page 47.

## Hierarchy

```text
Book
└── Chapter
    ├── chapter number and bilingual title
    └── ordered pages
        ├── page_number
        ├── french_text
        └── moore_text
```

Chapter starts are anchored by known chapter pages and bilingual title patterns. Chapter 5 additionally contains six numbered questions; use the number and both language headings as anchors, because each item can span several pages.

## Text cleanup before regex splitting

- Normalize Unicode quotation marks and apostrophes.
- Join discretionary hyphens at a line break. In French, preserve genuine clitic hyphens such as `a-t-il`.
- Preserve paragraph breaks after cleanup.
- Keep a page-level exception mechanism: a small number of pages have blocks assigned to the wrong side by the PDF extractor.

## Reference headings

French chapter headings follow `Chapitre <number> ...`; Mooré follows `Sak a <number> soaba ...`. Section headings include the French `Questions à discuter`, `Choses à apprendre`, `Sketch et chant`, `Ce que dit la Bible`, `Prier et agir` and their Mooré equivalents. The detailed French/Mooré outline is retained in the repository root as `book_struct_fr.md` and `book_struct_mos.md`.
