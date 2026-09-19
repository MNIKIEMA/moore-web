# Expert translation batch

`moore-web parse-expert-translations` reads the archived Google Sheets PDF at
`../faso-web-docs/expert-translations/mos_Mossi_seed_batch2 (3).pdf`.
Its eight columns are `ID`, `DOMAIN`, `CONTEXT`, `SOURCE`, `TRANSLATION`,
`COMMENTS`, `qa.check`, and `qa.comments`. Each physical table row is an
already paired French (`fra`) to Mooré (`mos`) translation. The parser does
not split sentences or run statistical alignment.

```bash
moore-web parse-expert-translations \
  --input '../faso-web-docs/expert-translations/mos_Mossi_seed_batch2 (3).pdf' \
  --output expert_translations.jsonl
```

Each JSONL row retains the source `id`, `source_text`, `target_text`, language
codes, `domain`, `context`, both comment fields, `qa_check`, and the 1-based
PDF `page`. `doc_id` defaults to the input filename stem and can be overridden
with `--document-id`. A `flag` or `reviewed` QA status is retained as-is; no
row is silently filtered by review state.

The current 17-page archive yields 352 unique rows: 324 `ok`, 23 `reviewed`,
and 5 `flag`. The parser rejects missing source text, translation, IDs,
duplicate IDs, and pages without exactly one eight-column table.
