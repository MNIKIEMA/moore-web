# Council of Ministers (`conseils`)

Source: `data/conseils_ministres/<date>/`. Each date directory has `info.json` plus PDFs for French, Mooré (`mossi`), and sometimes other languages.

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

- Use the date directory (`DD-MM-YY`) as the primary session key.
- Treat uppercase title/masthead lines and repeated page furniture as non-body text.
- Preserve numbered section headings and ministry names as segmentation anchors; a section can contain several paragraphs and sentences.

The input is a collection of PDFs plus manifests, not one JSON document; the final flattened corpus groups text by date before alignment.
