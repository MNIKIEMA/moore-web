# Mooré proverbs, volume 1

The archived app at
`../faso-web-docs/mooreburkina-priority/apps/mos-proverbes-volume-1/`
has 40 HTML proverb pages and a `segments.jsonl` file. Each page contains
three ordered blocks: the Mooré proverb, its French rendering or explanation,
and a repeated Mooré reading. Segment labels in HTML link each block to the
clean text and audio metadata in JSONL.

```bash
moore-web parse-moore-proverbs \
  --input-dir ../faso-web-docs/mooreburkina-priority/apps/mos-proverbes-volume-1 \
  --output moore_proverbs.jsonl
```

The parser emits one Mooré (`mos`) to French (`fra`) pair per proverb. It
removes the displayed number from the Mooré text, retains all French text
including explanatory lines, and excludes the repeated Mooré reading from
the pair. Each row has the page name and URL, proverb number, audio URL, and
segment labels for all three blocks.

It checks that the HTML and segment pages match, that every segment belongs
to one block, and that the repeated Mooré text agrees with the first reading.
The current archive yields 40 pairs, numbered 1 through 40.
