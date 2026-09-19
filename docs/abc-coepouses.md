# abcBurkina: Les coépouses

The archived French and Mooré text files are translations of the same Samo
tale, but their paragraph breaks differ. The parser pairs them at 25 hand-anchored
story-beat boundaries, including the three songs. It keeps the title,
subtitle, moral, and attribution as separate units.

```bash
moore-web segment-abc-coepouses \
  --fr-input ../faso-web-docs/abcburkina-contes/text/266-les-couses.txt \
  --mo-input ../faso-web-docs/abcburkina-contes/text/267-les-co-epouses-moore.txt \
  --output abc_coepouses_units.jsonl
```

Each row has `source_text` (`fra`), `target_text` (`mos`), a unit type,
source URLs, and character spans in whitespace-normalized article bodies.
`source_sentences` and `target_sentences` are provided for later alignment;
they are **not** paired by list position. In particular, the songs have
different sentence counts in the two languages.

The source-specific anchors are checked for uniqueness and order. If either
archived text changes, the parser fails rather than silently shifting a
boundary or dropping text. The output covers both article bodies in full.
