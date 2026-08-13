# Kadé facilitator manual (`kade`)

Source: separate French and Mooré facilitator PDFs, parsed by `src/moore_web/book_parser_facilitateur.py`.

## Physical layout

There are two monolingual manuals, not two columns in one PDF. Extract their text independently (current content ranges: French pages 3–57; Mooré pages 3–55), parse each into the same hierarchy, then align equivalent sections by order.

```text
Book
└── Chapter: `Chapitre <n>` or `Sak a <n> soaba`
    └── Section
        └── optional Subsection
            ├── numbered items: `<n>. <text>`
            ├── bullets: `• <text>`
            └── body text
```

## Stable section headings

French: `L'histoire de Kadé`, `Questions à discuter`, `Choses à apprendre`, `Sketch et chant`, `Sketch`, `Ce que dit la Bible`, `Prier et agir`.

Mooré: `Karem-y kibarã`, `Sõaseg sokdse`, `D sẽn segd n zãms bũmb niisi`, `Reem`, `Reem la yɩɩla`, `Wẽnnaam sebra sẽn yet bũmb ningã`, `Pʋʋsg la tʋʋmde`.

Introductory sections occur before Chapter 1 and have their own titles/subheadings. Stop at the training-materials end matter (`Matériels de formation` in French; `Tʋʋm teedo` in Mooré).

## Regex rules

- Chapter: `^(chapitre|sak a)\s+(\d+)(\s+soaba)?` (case-insensitive).
- Numbered item: optional parenthetical prefix, then `^<number>. <text>`; continuation lines belong to that item.
- A question-ending line and `Lisez ...` commonly introduce a subsection.
- Match heading text after whitespace normalization, but retain original line boundaries while collecting items.

The French facilitator names are normalized to the SIDA-book character names after parsing; this is a content-normalization step, not a structural boundary.
