# Mooré tales, volume 5

The archived app at
`../faso-web-docs/mooreburkina-priority/apps/mos-contes-volume-5/` has 60
HTML pages, one Mooré page followed by its French translation for each of 30
tales. It is the only bilingual Mooré tale collection in the archive; volumes
1 and 2 are Mooré only.

Each page's `#content` holds `div.m` blocks: the first is the numbered title
(`3 Katre ne wãamba` / `3 Le singe et l’hyène`; French sometimes writes
`12. Chat et souris`), the rest are paragraphs. Mooré paragraphs are made of
`div.txs` audio segments whose labels match `segments.jsonl`; French pages
are not narrated and have no segments.

```bash
moore-web parse-moore-tales \
  --input-dir ../faso-web-docs/mooreburkina-priority/apps/mos-contes-volume-5 \
  --output moore_tales.jsonl
```

One record per tale (`mos-contes-volume-5-01` … `-30`), Mooré as source
(these are oral tales told in Mooré; the French glosses Mooré words). Each
record has both titles, the paragraph texts, sentence lists for later
alignment, page URLs and the audio URL.

## Pairing checks

- Pages alternate Mooré/French; the two pages of a tale must carry the same
  title number, and numbers must run 1…30 in page order.
- Audio segments must exist for the Mooré page only, and the HTML segment
  labels must equal those in `segments.jsonl`.
- Segments are joined with a space (adjacent `div.txs` sometimes have no
  whitespace between them: `ye.</div><div>Yaa`); spans inside a segment are
  joined without one (tale 21's title is `2` + `1 Kɩɩba`). The extracted
  words match the archived `text/*.txt` files exactly, apart from the
  stripped title numbers.

## Alignment

The tale is the only reliable anchor. Paragraph counts agree for just 3 of
30 tales, and the French is a free translation: 970 Mooré sentences for 785
French. `e2e` aligns sentences inside each tale with LASER + FastDTW, never
across tales:

```bash
moore-web e2e -s moore-tales \
  -i ../faso-web-docs/mooreburkina-priority/apps/mos-contes-volume-5 \
  -o moore_tales_aligned.jsonl
```

This gives 987 pairs: 71 % score ≥ 0.6, 18 % below 0.5. About a third of
the rows repeat a French sentence because French merges two or more Mooré
sentences (1 FR : n MOS); those pairs are partial until merged or reviewed.
`--no-segment` writes the 30 tales as whole pairs.

## Review app

```bash
uv run python scripts/export_review_units.py --source moore-tales \
  --input ../faso-web-docs/mooreburkina-priority/apps/mos-contes-volume-5 \
  -o data/review/mos-contes-volume-5_units.jsonl
```

One unit per tale, title on the first row, sentence-segmented. Every unit
has uneven sides (up to 86 Mooré rows), so aligning in the app is real work.
