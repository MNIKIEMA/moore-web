# Raamde bilingual news (`news`)

Source: `data/raamde/raamde_corpus.json`, scraped by `src/moore_web/news_scaper.py` and segmented by `segment_news_data.py`.

## Input record

```json
{
  "url": "https://...",
  "title": "Mooré article title",
  "text_units": [
    "Mooré title/body paragraph",
    "Mooré body paragraph",
    "Kibarã yii <source>",
    "French title",
    "French body paragraph",
    "French source"
  ]
}
```

`text_units` preserves webpage paragraph order. The article is bilingual but has no explicit language field per unit.

## Language boundary

The structure is normally a contiguous Mooré run followed by a contiguous French run. `Kibarã yii ...` is a common Mooré source-credit/boundary marker, but it is not universal. A parser should classify each unit by language and split on the first stable language transition, rather than assuming an equal number of French and Mooré paragraphs.

After classification, segment paragraphs into sentences; retain article URL/title metadata until alignment is complete.

## Hazards

- Some scraped units contain multiple sentences with no separating whitespace.
- Credits and titles are structurally useful but often do not have a translation.
- Unicode diacritics and non-breaking spaces occur in Mooré text; normalize before language matching, but do not remove tone marks.
