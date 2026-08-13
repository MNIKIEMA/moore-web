# Simple dictionary parser (`simple`)

Implementation reference for `src/moore_web/one_column_dict_parser.py`. This is the most regex-sensitive source. It parses `Dictionnaire-Moore-français-English-avec-images.pdf` into `DictionaryEntry → Sense → Example` records.

## Extraction and page contract

The PDF is read as one vertical stream per page: obtain PyMuPDF blocks with `sort=True`, join each block's text with `\n`, then clean it. Do **not** treat a line break as an entry boundary by itself; PDF line wrapping can place English prose at the beginning of a line.

An entry can cross a page boundary. On page *N*, text before the first valid entry start is a spillover. Append it to the body of the final entry from page *N − 1*, then re-parse that merged body. If a page image causes the final line to repeat the headword, remove only that exact trailing duplicate.

```text
page N−1                         page N
------------------------------   ------------------------------
lemma [tone] POS Frn... Eng...   continuation of previous body
                                  next-lemma [tone] POS Frn...
```

## Canonical entry language

The normal form is compact: `Frn` and `Eng` are labels attached directly to their text.

```text
lemma [tone] POS FrnFrench definition EngEnglish definition
```

Full entry with senses and chained example:

```text
bãoogo [ã̀-ó] adj
1) adj Frncalme, en sécurité Engcalm, secure
2) adj Frnaligné à la file indienne Englined up in a queue
3) n Frnpaix Engpeace
```

Interpretation:

```text
DictionaryEntry
  lemma: bãoogo
  ipa: ã̀-ó
  pos: adj                         # entry-level POS: first header POS
  senses:
    1: pos=adj, french=calme..., english=calm...
    2: pos=adj, french=aligné..., english=lined up...
    3: pos=n,   french=paix,       english=peace
```

### Entry-start grammar

An entry start requires all of the following, in order:

```text
<lemma> <optional [tone]> <optional parenthetical> <optional "N)"> <POS> <Frn or N)>
```

The parser's POS vocabulary is deliberately finite:

```text
part.gram | expr. | indéf. | num | n.pl | interj | aux | <Not Sure> |
n.propre | postpos | Verbe | adj | n | v | v:Any | dém | Nom | inter. |
Adjectif | verbe.it | v.inacc | conj | pron | adv
```

Use an anchored multiline pattern equivalent to:

```regex
^(?!\d+\)\s)
([^\s\n](?:(?![,\.]\s)[^\n])*?)
\s+(?:\[([^\]]+)\]\s+)?(?:\([^)]+\)\s+)?(?:\d+\)\s+)?
(POS)\s+(?=Frn|\d+\))
```

The `POS` placeholder must be substituted with the full vocabulary above. The leading negative lookahead is mandatory: `2) adj Frn...` is a sub-sense, never a lemma. The lemma pattern must reject `", "` and `". "` inside a candidate: those patterns identify wrapped sentence prose, such as `person, there are many.`, rather than a headword.

When compiling this regex with `re.VERBOSE`, escape the space in `<Not\ Sure>`; otherwise verbose mode silently changes it to `<NotSure>`.

## Senses, fields, and examples

### Numbered senses

Split a body at `\s+N)\s+` only if it is followed by an optional POS and `Frn`. Capture the optional POS; it overrides the entry POS for that sense. Both forms are valid:

```text
2) Frndeuxième définition Engsecond definition
2) adj Frndeuxième définition Engsecond definition
```

Do not split merely because `N)` appears: it can be a reference or a wrapped artifact.

### French/English pairs

Within a sense, extract repeated pairs:

```text
Frn<french>Eng<english>
```

The first pair is the definition. Later pairs normally provide translations for a Mooré example embedded in the preceding English text:

```text
Frndefinition Engdefinition {e.g. Mooré example}
FrnFrench example EngEnglish example
```

`{e.g. ...}` is therefore Mooré example text, not English definition text. There is also a compact form where the French example follows the marker in the first pair:

```text
Frndefinition {e.g. Mooré example} FrnFrench example EngEnglish example
```

An English definition or example may be missing. Preserve an empty/missing value; do not invent a translation.

### Labelled metadata

Extract these case-insensitive, labelled fields from a sense while keeping the surrounding definition intact:

```text
var.:        variant
syn.:        synonym
Nominal.:    nominal form
scient.:     scientific name
Racine.:     root
infinitif.:  infinitive
Empr.:       borrowing
sg.:         singular
Inaccompli.: imperfective
ant.:        antonym
(catégorie: <value>.)
```

Fields can appear on the same line as glosses and can contain commas. Their terminator is the next known field label, a newline, or the end of the sense—not every punctuation mark.

## Required normalization, in order

1. Expand PDF ligatures before matching. In particular, map both standard `ﬁ` and this PDF's erroneous Coptic `ϧ` extraction to `fi`; otherwise lemmas such as `fika`, `film`, and `fisiye` are corrupted.
2. Remove isolated page numbers, dates, `file:///...`, dictionary mastheads, lone dash lines, and blank-line runs.
3. Join a word broken around a hyphen (`word- word` → `word-word`).
4. Normalize `unspec. var.` lines as described below.
5. Keep remaining newlines until entry and page-spillover splitting is complete.

## Variant-only records: `unspec. var.`

These source records have no POS and no `Frn`/`Eng` labels, so they cannot be handled by the normal entry-start regex:

```text
ãbe [ã̀] unspec. var. of wãbe
barkudi unspec. var. of barkudga
kʋɩdga
[ʋ́] unspec. var. of kʋdga
```

Before splitting entries:

- join a newline between lemma and tone/`unspec.`;
- remove nested `(unspec. var. of ...)` text from the target;
- rewrite the record to a synthetic parseable form:

```text
ãbe [ã̀] n Frn (unspec. var. of wãbe) Eng
```

After parsing, recognize that synthetic French field. Produce an entry with no senses, an empty POS (not synthetic `n`), and `variants.variant = [target]`.

## Non-negotiable false-positive guards

| Source shape | Required behavior | Why |
| --- | --- | --- |
| `person, there are many. ... 2) n Frn...` | Do not start a lemma at `person` | Wrapped English sentence can otherwise become a bogus lemma. |
| `2) adj Frn...` | Keep under prior entry as a sense | It is not a new dictionary entry. |
| `barkudga [tone] (unspec. var. barkudi) n Frn...` | Keep `barkudga` as lemma and `[tone]` as IPA | Parenthetical must not force regex backtracking into the lemma. |
| `bala <Not Sure> Frn...` | Parse `<Not Sure>` exactly as POS | A verbose regex loses its embedded space unless escaped. |
| `gẽ[é] v Frn...` | Recover `lemma=gẽ`, `ipa=é` | PDF extraction may attach tone brackets to the lemma. |
| `poorẽ1)` | Strip only trailing sub-entry index | This is a known residual artifact, not a different lemma. |
| `Frnnous Engwe` or a candidate containing `?` | Drop as an artifact | Definition text or sentence fragment was misidentified as a lemma. |

## What cannot safely be inferred

- A missing French or English translation may be genuinely absent from the PDF.
- Image pages and broken blocks can repeat a headword or split body text in unusual places.
- New POS labels are not automatically safe: add them to the POS vocabulary and test that they do not create new false entry starts.

## Minimum regression fixtures for a new parser

The parser must pass these structural cases before being run on the full PDF:

```text
# no bare-number lemma; three senses with POS adj, adj, n
bãoogo [ã̀-ó] 1) adj Frncalme Engcalm
2) adj Frnaligné Englined up
3) n Frnpaix Engpeace

# variant-only entry splits out of its predecessor
ãase [ã́-é] v Frncasser Engbreak
ãbe [ã̀] unspec. var. of wãbe
ãbga [ã́-à] n Frnpuce Engflea

# `<Not Sure>` is a valid POS
bala <Not Sure> Frnseulement Engonly

# wrapped English continuation must not start a lemma
foo n Frnx EngIt's not only one
person, there are many. syn: wʋsgo. 2) n Frnpluralité Engplural
```

The corresponding executable regression tests are in `tests/test_simple_parser.py`. The underlying issue history is in `ISSUES.md` and `docs/resolved-issues.md`; retain it as the evidence log, while this file is the parser-building contract.
