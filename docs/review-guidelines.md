# Alignment review guidelines

How to review French–Mooré review units in the annotation app. Each unit is a
tale, a lesson section, an article or a paragraph: a stretch of text whose two
sides are known to match as a whole. Inside the unit, the text is split into
sentences and pre-aligned automatically. Your job is to fix that alignment.

## Principle: sentences first, paragraph as a fallback

Work at sentence level by default. Always read the whole unit, not just the
pair in front of you: you need the context to catch pronoun, ellipsis and
reordering errors, and to see when a "missing" sentence was merged into a
neighbour.

Fall back only as far as you have to:

1. **1:1 link.** One French sentence matches one Mooré sentence.
2. **n:m link.** When sentences don't line up, merge only the mismatched
   sentences (2:1, 1:2, 3:2…). The rest of the unit stays at sentence level.
3. **Paragraph link.** When content is reordered across non-adjacent
   sentences, link the whole paragraph.
4. **Free translation.** When one side paraphrases or summarises the other and
   there is no sentence-level correspondence, flag the unit
   `free_translation`. It is kept at paragraph level only.

| Situation | Action |
| --- | --- |
| Content matches 1:1 | Keep the sentence pair |
| One side splits or merges sentences | Merge the span into an n:m link |
| Content reordered across sentences | Link the paragraph |
| Paraphrase or summary, no sentence correspondence | Flag the unit `free_translation` |
| Content on one side has no counterpart | Reject that sentence, keep the rest |

## How to merge

Favour **the smallest link whose two sides mean the same thing**.

1. **Fix segmentation before merging.** Many apparent n:m cases are splitter
   errors: abbreviations, dialogue quotes, ellipses. If a sentence was cut in
   the wrong place, re-split or re-join it on that side first, then align.
   Tag the link `segmentation`.
2. **Keep links minimal.** Merge only as many adjacent sentences as needed.
   2:1 is better than 3:2, and 3:2 is better than a paragraph. If a merged link
   can be split back into two clean links, split it.
3. **Merging beats dropping.** If the content corresponds, merge. Don't reject
   a sentence just because it doesn't line up 1:1.
4. **Don't merge to hide missing or extra content.** If one side adds or
   leaves out a whole clause or sentence, leave that part unaligned or
   rejected; don't absorb it into a neighbouring link. Small additions are
   fine: connectives and discourse markers (*alors*, *donc*), repeated
   subjects, the usual Mooré storytelling formulas.
5. **Keep adjacency and order.** A link joins only neighbouring sentences, and
   links never cross each other. Reordering across non-adjacent sentences
   means a paragraph link.
6. **Tie-break by meaning overlap.** If a sentence could go with the link
   before or after it, attach it where most of its content lands. If it is
   really split between both, merge all three.
7. **Never edit the text to make it fit.** Align what is there. Translation
   corrections go in the separate correction field, so the raw pair stays
   faithful to the source.

### Quick check

> Could a translator produce side B from side A alone, without adding or
> losing information?

- Yes: the link is right.
- It needs a neighbouring sentence: merge.
- It needs content that exists nowhere on the other side: reject that part.

## Reported speech

A quote is a natural unit: one speaker, one turn, one pair of guillemets.
Sentence splitters are least reliable inside quotes, and translators often
restructure dialogue. So the quote is the largest link you should need for
reported speech. Inside it:

- **Split the quote into 1:1 links only if every sentence of the quote matches
  one to one** (see the two examples below).
- **Otherwise, keep the whole quote as one link**, tagged `translator_merge`,
  rather than a mix of 1:1 and n:m links inside the quote.
- **The introduction goes with the quote.** "Le caméléon dit à l'homme :" is
  attached to the first link of the quote, or to the whole quote if it stays
  one link. It is never a link of its own.
- **A quote interrupted by the speech verb is one sentence.**
  « Viens, dit-il, rentrons. » is not split.
- **Direct speech on one side and indirect speech on the other** ("il lui dit
  de rentrer" vs « Rentrons ») is still one link if the content matches. It is
  a translation choice, not a mismatch.
- **If the introduction is placed differently** (« Rentrons », dit l'homme.
  vs a Mooré introduction before the quote), keep the introduction and the
  quote together in one link.

Splitting a quote loses nothing: links keep their position in the unit, so
exports can rebuild the full quote or the full dialogue.

### Example: dialogue with a two-sentence quote

A unit from `mos-contes-volume-5`:

> **fr:** Le caméléon dit à l'homme : « Je te remercie beaucoup pour m'avoir
> sauvé dans cette situation difficile. Dès les premières pluies, quand tu
> passes par ici pour aller semer, je saurai te remercier pour ce que tu m'as
> tait ».
>
> **mos:** Gomtɩʋʋg yeela ninsaala : « M pʋʋsd-f-la bark wʋsg f sẽn yiis-m
> yel-to-kãngã pʋgẽ wã. Sigr saag sã n Iʋɩ, f sã n wa loogd ka n na n tɩ
> bʋde, m na bãng n pʋʋs-f bark f sẽn maan-mã yĩnga».

Both sides have the same structure (introduction, then a quote of two
sentences), so this becomes **two 1:1 links**, not one 2:2 link:

| # | fr | mos |
| --- | --- | --- |
| 1 | Le caméléon dit à l'homme : « Je te remercie beaucoup pour m'avoir sauvé dans cette situation difficile. | Gomtɩʋʋg yeela ninsaala : « M pʋʋsd-f-la bark wʋsg f sẽn yiis-m yel-to-kãngã pʋgẽ wã. |
| 2 | Dès les premières pluies, quand tu passes par ici pour aller semer, je saurai te remercier pour ce que tu m'as tait ». | Sigr saag sã n Iʋɩ, f sã n wa loogd ka n na n tɩ bʋde, m na bãng n pʋʋs-f bark f sẽn maan-mã yĩnga». |

In link 2, the clauses match one to one and in the same order: *sigr saag sã n
lʋɩ* = "dès les premières pluies"; *f sã n wa loogd … n tɩ bʋde* = "quand tu
passes par ici pour aller semer"; *m na bãng n pʋʋs-f bark* = "je saurai te
remercier"; *f sẽn maan-mã yĩnga* = "pour ce que tu m'as fait". Nothing is added
or missing on either side, so merging would break rule 2.

Points to note:

- **The introduction stays with the first sentence of the quote.** "Le
  caméléon dit à l'homme :" is not a link of its own. If the splitter cut at
  the colon, re-join it (rule 1).
- **Leave the unbalanced guillemets as they are.** Link 1 opens « and link 2
  closes ». Don't add or remove quote marks during review (rule 7); the
  punctuation normalization pass handles them, along with spacing differences
  such as a space before » on one side and none on the other.
- **Record OCR errors in the correction field, not in the raw text:** *tait* →
  *fait*, and *Iʋɩ* → *lʋɩ* (capital I read in place of the letter l). The I/l
  confusion is likely to recur across this source.

### Example: short sentences inside a quote

> **fr:** Désespéré, l'homme dit à sa femme : « Repartons chez nous. Tu as
> vu ? Il m'a frappé les yeux. Qu'est-ce que je vais faire à la campagne si
> je ne vois plus ? Viens, rentrons chez nous ».
>
> **mos:** Ne sũ-sãanga, rao wã yeela a pagã : « D lebg n kuili. Fo yãame ?
> A wẽe mam ninã. M na n tɩ, maana bõe weoogẽ wã tɩ m sã n pa le neẽ ? Wa, d
> leb yiri ».

All five sentences of the quote match one to one, so this becomes five 1:1
links:

| # | fr | mos |
| --- | --- | --- |
| 1 | Désespéré, l'homme dit à sa femme : « Repartons chez nous. | Ne sũ-sãanga, rao wã yeela a pagã : « D lebg n kuili. |
| 2 | Tu as vu ? | Fo yãame ? |
| 3 | Il m'a frappé les yeux. | A wẽe mam ninã. |
| 4 | Qu'est-ce que je vais faire à la campagne si je ne vois plus ? | M na n tɩ, maana bõe weoogẽ wã tɩ m sã n pa le neẽ ? |
| 5 | Viens, rentrons chez nous ». | Wa, d leb yiri ». |

Points to note:

- **Don't merge a link because it is short.** "Tu as vu ?" / "Fo yãame ?" is a
  correct, complete pair. Whether short pairs are used for training is decided
  at export time (minimum length, or joining neighbours with a sliding window),
  not during review.
- **Watch for splitter traps.** French puts a space before ? and ! ("vu ?"),
  and some splitters miss that boundary. If links 2–4 arrive glued together,
  re-split them (`segmentation`); don't align them as 3:3. The comma in
  "M na n tɩ, maana bõe …" is inside the sentence and must not be split.

## Reason tags

Give every link that is not a plain 1:1 a reason:

| Tag | Meaning |
| --- | --- |
| `segmentation` | The sentence splitter cut in the wrong place |
| `translator_merge` | The translator merged or split sentences |
| `reorder` | Content moved across sentences (paragraph link) |
| `free` | Free translation (unit flagged `free_translation`) |

## Data model

A paragraph fallback is just a link that covers every sentence of the
paragraph on both sides, so the same schema covers every case:

```json
{
  "doc_id": "mos-contes-volume-5",
  "unit_id": "mos-contes-volume-5-03",
  "position": 7,
  "fr_sent_ids": [12, 13],
  "mo_sent_ids": [11],
  "shape": "2:1",
  "reason": "translator_merge",
  "status": "accepted"
}
```

`doc_id`, `unit_id` and `position` keep the order, so exports can be built
from the same data:

- **Sentence pairs** (MT fine-tuning, evaluation): filter on `shape`, e.g.
  1:1 only, or links up to 2:2.
- **Paragraph or document pairs** (LLM training, document-level evaluation):
  join consecutive links within a unit.
- **Sliding windows** of k consecutive links as a middle ground.

Units longer than about 300 tokens per side should be split for review, at a
link boundary where both sides agree, never at an arbitrary point on one side.

## Monitoring

Track the share of non-1:1 links and `free_translation` units per source. If a
source falls back more than about 30% of the time, automatic alignment is not
working for it: pre-segment that source at paragraph level instead of fixing
it by hand, and check whether `segmentation` tags point to a splitter bug.

## Background

- Läubli et al. (2018), *Has Machine Translation Achieved Human Parity? A Case
  for Document-level Evaluation*: judging sentences without context hides
  errors.
- Post & Junczys-Dowmunt (2023), *Escaping the sentence-level paradigm in
  machine translation*: keep document boundaries and order in training data.
- Thompson & Koehn (2019), *Vecalign*: real translations need n:m alignment,
  not forced 1:1.
- Kreutzer et al. (2022), *Quality at a Glance*: misalignment is one of the main
  faults in low-resource web corpora.
- Thai et al. (2022), *Exploring Document-Level Literary Machine Translation
  with Parallel Paragraphs from World Literature* (Par3): sentence-level
  alignments "are rarely available for literary translations because
  translators merge and combine sentences"; Par3 aligns at paragraph level.
