# Council of Ministers (`conseils`)

Source: `../faso-web-docs/conseils-ministres/<date>/` (the `faso-web-docs` archive, synced to the `madoss/faso-web-docs` HF bucket). Each date directory has `info.json` plus PDFs for French, Mooré (`mossi`), and sometimes other languages.

## Manifest structure

```json
{
  "french": "https://.../CONSEIL_DES_MINISTRES_....pdf",
  "mossi": "https://.../MOORE_....pdf",
  "goulma": "https://...",
  "dioula": "https://...",
  "peulh": "https://..."
}
```

For French–Mooré parsing, pair only the `french` and `mossi` documents from the same date. Never align text across dates: each council session is an independent alignment unit.

## Document structure

The paired PDFs are sequential, mostly single-column government communiqués: document masthead/title, dated introductory paragraphs, then ordered agenda/ministry sections and closing credits. Extract each PDF in reading order, discard repeated headers/footers and page numbers, sentence-segment the remaining body, then align within that date.

## Regex anchors

- Use the date directory (`YYYY-MM-DD`, e.g. `2024-02-21`) as the primary session key; the parsed corpus's `date` field uses the same format. Directory dates can be wrong: `2026-06-21` holds only a Mooré PDF byte-identical to session N°021 in `2026-06-25`, so check the `PP-G N°` session number on page 1 when a date has only one language.
- Treat uppercase title/masthead lines and repeated page furniture as non-body text.
- Preserve numbered section headings and ministry names as segmentation anchors; a section can contain several paragraphs and sentences.

The input is a collection of PDFs plus manifests, not one JSON document; the final flattened corpus groups text by date before alignment.
