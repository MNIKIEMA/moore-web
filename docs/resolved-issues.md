# Data Quality Issues


## Issue #4 — simple_parser: corrupted lemmas (~106 entries)

**Source:** `simple_parser.py` / `output.jsonl`
**Status:** S1 fixed, S2 fixed, S3 fixed, S4 fixed, S5 open, S6 fixed

### S1 — False entry match from PDF line wrap · **~78 entries** · 🔴 High

**What happens:** A `.` or `,` appears in the `lemma` field (e.g. `"person, there are many. syn: wʋsgo ."`).

**Root cause:** PDF line wrapping splits an English sentence mid-word, placing the continuation (e.g. `person, there are many.`) at the start of a new line. The `entry_start_pattern` in `split_dictionary_entries` matches it as a new lemma because `([^\s\n][^\n]*?)` accepts any character including `,·` and `.·` (where `·` is a space).

**PDF evidence:**

```text
...Eng It's not only one
person, there are many. syn: wʋsgo .  2) n Frn pluralité, plurielEng
```

`person` starts a new line → regex fires → fake lemma `person, there are many. syn: wʋsgo .`, grammar `n`, body `pluralité / plural`.

**Proposed fix:** Restrict the token group in `entry_start_pattern` to disallow sentence-style punctuation (`,·` or `.·`):

```python
# Before
r"^([^\s\n][^\n]*?)\s+..."
# After — token must not contain ", " or ". " (sentence punctuation)
r"^([^\s\n](?:(?![,\.]\s)[^\n])*?)\s+..."
```

---

### S2 — `(unspec. var. …)` annotation absorbed into lemma · **~28 entries** · 🟡 Medium

**What happens:** The lemma contains the full parenthetical, e.g. `"barkudga [à-ú-à] (unspec. var. barkudi)"` instead of just `barkudga`.

**Root cause:** The entry pattern has no group for the `(unspec. var. X)` parenthetical that appears between the tone bracket and the grammar tag. The regex backtracks and absorbs everything (including the tone) into the token group.

**PDF evidence:**

```text
barkudga [à-ú-à] (unspec. var. barkudi) n Frn arbre (espèce)...
```
Regex tries: token=`barkudga`, tone=`[à-ú-à]`, but `(unspec. var. barkudi)` blocks grammar match → backtracks → token=`barkudga [à-ú-à] (unspec. var. barkudi)`, tone skipped, grammar=`n` ✓.

**Proposed fix:** Add an optional parenthetical group between tone and grammar in `entry_start_pattern`:

```python
r"^([^\s\n][^\n]*?)\s+(?:\[([^\]]+)\]\s+)?(?:\([^)]+\)\s+)?(?:\d+\)\s+)?({GRAMMAR_PATTERN})\s+(?=Frn|\d+\))"
#                                                              ^^^^^^^^^^^^^^^^ new optional group
```

Then strip `(unspec. var. X)` from the token group as a post-processing step, and add the variant to the `variants` dict.

---

---

### S3 — `unspec. var.` entries not split off · **15 entries** · 🟡 Medium

**What happens:** Entries of the form `lemma [tone] unspec. var. of target` have no `Frn`/`Eng`
marker and no grammar tag, so `split_dictionary_entries` never starts a new entry there.
They bleed into the previous entry's body — e.g. `ãbe [ã̀] unspec. var. of wãbe` appears
inside `ãase`'s English field.

**PDF evidence (page 12):**

```text
ãase [ã́-é] v
Frncasser sur l'arbre Engbreak on a tree
ãbe [ã̀] unspec. var. of wãbe          ← no Frn/Eng, no grammar tag
ãbga [ã́-à] n
Frnpuce, tique Engflea, tick ...
```

**Categories (15 total, from `unspec_var_analysis.txt`):**

| Cat | Count | Format                                      |
|-----|------:|---------------------------------------------|
| A   |     7 | `lemma [tone] unspec. var. of X` (one line) |
| B   |     5 | `lemma unspec. var. of X` (no tone)         |
| C   |     1 | split across PDF blocks (newline inside)    |
| D   |     2 | nested: `(unspec. var. …) unspec. var.`     |

See `unspec_var_analysis.txt` for full context and page references.

**Proposed fix — two steps:**

1. **Pre-process** the raw page text in `parse_doc` before `parse_page`. Detect lines matching
   `unspec. var.` and normalise them into a parseable stub entry:

   ```python
   UNSPEC_RE = re.compile(
       r'^([^\s\n][^\n]*?)\s+(?:\[([^\]]+)\]\s+)?unspec\.\s+var\.\s+of\s+(\S+)',
       re.MULTILINE,
   )
   ```

   Replace each match with a synthetic `Frn`-bearing line so the splitter can cut there:

   ```text
   lemma [tone] n Frn (unspec. var. of target) Eng
   ```

   (Category C needs the newline collapsed first; D needs the nested parenthetical stripped.)

2. **Post-process** in `_make_entry` or `analyze_body`: detect the `(unspec. var. of X)` French
   field and record `X` as a variant instead of a sense.

**Effort:** ~20 lines. Covers categories A and B cleanly; C and D need minor extra
normalisation (collapse newline, strip nested parens).

---

### S4 — Sub-entry numbers `2)` / `3)` / `4)` parsed as lemmas · **34 entries** · 🔴 High

**What happens:** Entries like `{"lemma": "2)", "pos": "<Not Sure>", ...}` appear in the output.

**Root cause:** PDF line wrapping places a sub-entry number at the start of a new line.
`entry_start_pattern` matches `2)` as the token, the following grammar tag as `pos`,
and the sense as the body — detaching the sub-entry from its parent lemma.

**PDF evidence:**

```text
bãoogo [ã̀-ó] 1) <Not Sure> Frncalme, en sécurité Engcalm
2) <Not Sure> Frnaligné à la file indienne Englined up in a queue
```

`2)` starts a new line → regex matches it as token, `<Not Sure>` as grammar → fake entry.

**Fix:** Add `(?!\d+\)\s)` negative lookahead at the start of both `entry_start_pattern`
and `entry_start_re` to reject lines that open with a bare sub-entry number:

```python
r"^(?!\d+\)\s)([^\s\n](?:(?![,\.]\s)[^\n])*?)\s+..."
```

**Status:** done — 55 tests passing.

---

### S5 — `<Not Sure>` grammar tag broken in `split_first_entry` · fixed · 🟡 Medium

**What happens:** When a page's first entry uses `<Not Sure>` as its grammar tag,
`split_first_entry` fails to find it. The entry is silently dropped into `before`
and lost — its sub-entries may then bleed into the previous page's last entry.

**Root cause:** `entry_start_re` in `split_first_entry` is compiled with `re.VERBOSE`,
which ignores unescaped spaces in the pattern. The space in `<Not Sure>` is swallowed,
making the pattern match `<NotSure>` instead of `<Not Sure>`.

`split_dictionary_entries` uses `re.MULTILINE` only, so `<Not Sure>` works there.

**Fix:** Escape the space inside `<Not Sure>` in `GRAMMAR_PATTERN` for VERBOSE contexts,
or compile `entry_start_re` without `re.VERBOSE`.

---

### S6 — `2) grammar Frn` sub-entries not split into separate senses · open · 🔴 High

**What happens:** When a numbered sub-entry has a grammar tag between the number and `Frn`
(e.g. `2) adj Frnaligné...`), it is not recognised by `split_sub_entries`. The sub-sense
merges into the English field of the preceding sense instead of becoming its own sense.

**Root cause:** `SUB_SPLIT_RE = re.compile(r"\s+\d+\)\s+(?=Frn)")` only matches
`2) Frn` — the lookahead requires `Frn` immediately after the number. When a grammar
tag like `adj` or `n` appears between the number and `Frn`, the match fails.

**PDF evidence:**

```text
bãoogo [ã̀-ó] 1) adj Frncalme, en sécurité Engcalm, secure
2) adj Frnaligné à la file indienne Englined up in a queue
3) n Frnpaix Engpeace
```

After S4 fix, `2) adj Frnaligné...` stays in `bãoogo`'s body — but `analyze_body`
produces only 1 sense because `split_sub_entries` can't cut at `2) adj Frn`.
The `2) adj` text bleeds into the English field of sense 1.

**Fix:** Extend `SUB_SPLIT_RE` to optionally consume a grammar tag before `Frn`:

```python
SUB_SPLIT_RE = re.compile(
    rf"\s+\d+\)\s+(?:(?:{GRAMMAR_PATTERN})\s+)?(?=Frn)"
)
```

---

### S-series fix priority

| Issue | Priority | Effort    | Impact               | Status |
|-------|----------|-----------|----------------------|--------|
| S1    | High     | 2 lines   | -78 bad lemmas       | done   |
| S2    | Medium   | 3 lines   | -28 bad lemmas       | done   |
| S3    | Medium   | ~20 lines | -15 bleeding entries | done   |
| S4    | High     | 1 line    | -34 bad lemmas       | done   |
| S5    | Medium   | 1 line    | -37 bad pos tags     | done   |
| S6    | High     | 1 line    | -34 merged senses    | done   |

- **S1** Restrict token pattern to reject sentence-style punctuation (`,·` / `.·` where `·` is a space)
- **S2** Add optional parenthetical group between tone bracket and grammar tag
- **S3** Detect and normalise `unspec. var.` stub entries (no `Frn`/`Eng` marker)
- **S4** Negative lookahead to reject bare sub-entry numbers (`2)`, `3)`) as lemmas
- **S5** Fix `re.VERBOSE` swallowing the space in `<Not Sure>` grammar tag
- **S6** Extend `SUB_SPLIT_RE` to handle `2) grammar Frn` format

S6 case

```json
{"lemma":"about what?","ipa":"","pos":"<Not Sure>"}
{"lemma":"poorẽ1)","ipa":"","pos":"<Not Sure>"}
```

---

## Issue #2 — sida-facilitateur: FR/MO inputs swapped

The `french` column contains Mooré text (326/351 sentences have Mooré diacritics like ɛ ɔ ã ẽ ɩ),
and the `moore` column contains French text (0/351 sentences have those diacritics).

**Impact:** All 351 pairs in `sida-facilitateur_aligned.json` were unreliable.

**Result after fix:** 692 pairs (vs 351), mean score 0.700 (vs 0.366). File regenerated.

Moore name:
In sida:
-Fr: Poko, Yembi, Yõdi, Aminata, Adama, Séni, Mariam
-Mo: Poko/Pok, Yembi, Yõdɩ, Ãminat, Ãdam, Seeni, Mariyam

Map for facilitateur:

- Kadé: Poko
- Kaluu: Mariam
- Katiu: Yembi
- Kayaga: Aminata
- Kande: Poko
- Atiana: Séni
- Betaro: Yõdi
- Apiu: Adama
- Atega: Abdou

## Steps to fix

1. Fix name mappping in code for facilitateur (Done)
2. Run simple parser and bicol parser to check the most reliable pairs
3. Rerun `conseils_ministres_aligned.json` because I fix the pagination issue
4. Push first vesion to HF dataset

---

## Issue #3 — Dictionary parser failures (bicolumns_parser.py)

**Status:** in progress
**Source:** `parser.log`, run on `data/Dictionnaire-Moore-français-English.pdf`
**Baseline warnings:** 1296 (1256 from `extract_examples`, 40 from `parse_complex_definition`)
**Current warnings:** 103 (93 from `extract_examples`, 10 from `parse_complex_definition`)

---

### C1 — Period-only fragments · **618 occurrences** · 🔴 Easy fix

**What happens:** `extract_examples` receives the string `"."` or `".."`. It finds 1 segment `["."]`. Since `num_segments < 2`, it logs a failure and returns `[]`.

**Root cause:** Two sub-causes:

1. `split_entry` over-splits. When a definition ends with `word. NextLemma`, `split_entry` peels off `NextLemma` but leaves a dangling `.` which is forwarded to `extract_examples`.
2. Scientific names / abbreviations. After `cat_sci_pattern` consumes `Category: Tree. bauhinia reticulata.`, the remainder is `"."`.

**PDF evidence (page 13):**
> `Nom. arbre (espèce); tree sp. Category: Tree. bauhinia reticulata, pilostigma reticulatum.`
After `cat_sci_pattern` strips the category + scientific name the remaining text is `"."` which enters `extract_examples`.

**Fix:** One-liner guard at the top of `extract_examples`:

```python
if re.match(r'^\.+$', text.strip()):
    return []
```

---

### C2 — Orphan `)` artifacts · **15 occurrences** · 🟢 Easy fix

**What happens:** The example text is just `")"`, `") miiga [í]"`, or `") raag biiga [à-í-à]"`. These produce 0 segments (no sentence-ending punctuation), triggering the failure.

**Root cause:** Two-column merge bleeding. An entry in column 1 has a cross-reference like `(Comparez: miiga)`. The closing `)` lands at the top of a column-2 block, immediately followed by the next-entry lemma. After linear column merge, this appears as `") miiga"` inside the previous entry's text.

**PDF evidence (page 15):**

```shell
[Left col]   bale-sãoogo  bãmba
[Right col]  bãmba1   Comparez: -kãnga. ...
[Right col]  bãmba2   [ã̀-á]  Comparez: kãnga (sg.), ada.
             ...Bãmba waa ka zaamẽ.
             Ceux-là sont venus ici hier.
```

The `(sg.)` from a cross-reference in column 1 closes on column 2, appearing as `") bãmba2"` in the merged text.

**Fix:** The code already strips `^\)\s*` from segments. Additionally, filter any segment whose sole content is `)` plus optional word/IPA (no verb/noun content):

```python
segments = [s for s in segments if not re.match(r'^\)\s*[\w\s\[\]áàâéèêíîóôúûãẽĩõũ-]*$', s)]
```

---

### C3 — Fragments starting with `,` or `/` · **117 occurrences** · 🔴 Easy fix

**What happens:** Segments like `", pilostigma reticulatum."`, `"/25 litre can."`, `"/sth."`, `"/ neotis denhami."` are fed to `extract_examples`. They form 1 segment and fail.

**Root cause — comma (92 occurrences):** Scientific name lists use commas between species: `Category: Tree. bauhinia reticulata, pilostigma reticulatum.` After the first scientific name the remainder `", pilostigma reticulatum."` is incorrectly treated as example text.

**PDF evidence (page 13):**
> `Varinat: bãgen-daaga. Nom. arbuste (espèce)... Category: Bush, shrub. piliostigma reticulata, bãgen-yãanse.`
The comma in the scientific name list after `reticulata` is not a sentence boundary.

**Root cause — slash (25 occurrences):** `/` is used as "or" in English definitions (`"to whip sb./sth."`) and as separator in alternative scientific names (`"eupodotis senegalensis / neotis denhami"`). The parser splits on the preceding `.` leaving `"/sth."` or `"/ neotis denhami."` as orphan fragments.

**PDF evidence (page 34):**
> `bidon; can, a 5-gal./25 litre can. Category: Container.`
The `.` in `5-gal.` triggers a split, leaving `"/25 litre can."` as a fragment.

**PDF evidence (page 50):**
> `donner un coup de fouet; hit someone with a whip, to whip sb./sth.`
The `./` causes `"/sth."` to be left as a fragment.

**PDF evidence (page 14):**
> `Varinat: bakargo. Nom. outarde (petite)... Category: Bird. eupodotis senegalensis / neotis denhami.`
The `/` is an alternative scientific name separator.

**Fix:** Filter segments starting with `,` or `/` in `extract_examples`:

```python
segments = [s for s in segments if not re.match(r'^[,/]', s.strip())]
```

---

### C4 — Odd segment count (miscellaneous) · **48 occurrences** · 🟡 Medium

**What happens:** The block passed to `extract_examples` does not produce a segment count divisible by 2 or 3, and it's not a proverb. The full text is logged at line 280 (no segment list).

**Sub-types:**

**C4a — `v.itératif.` label (10 occurrences):** The POS label `v.itératif.` contains a period. `re.findall(r"[^.?!]+[.?!]", text)` splits inside it, yielding extra segments `"v"` and `"itératif"`.

**PDF evidence (page 76):**
> `fẽde Plural: fẽdse. singulier: fẽdge. v.itératif. écraser, écrabouiller; to squash, crush.`
After segmentation: `["fẽde Plural: fẽdse.", "singulier: fẽdge.", "v.", "itératif.", "écraser, écrabouiller; to squash, crush."]` — 5 segments, not divisible by 2 or 3.

**Fix:** Normalize `v.itératif.` → `v-itératif.` before segmentation, or add to the POS-label exclusion list.

**C4b — Encyclopedic inline Moore definition (38 occurrences):** Entries where the example block contains a Moore definition (`Gepeyɛse (GPS) : yaa tʋʋmd...`) + French definition, without an English sentence. These produce an indeterminate number of segments.

---

### C5 — Proverb / saying examples · **61 occurrences** · 🟡 Medium fix

**What happens:** The example block contains a proverb in all three languages plus a parenthetical explanation. The explanation adds extra segments, making the total not divisible by 3.

**PDF evidence (page 2):**

```shell
Weer a to. Une autre fois / moment. Another time / moment.
Lampalga pa yiisd a to bugmẽ ye. Du coton ne sort pas du coton du feu.
(proverbe, par ex. un pauvre ne peut pas sortir un autre d'une difficulté financière).
Cotton can not save other cotton that caught fire.
(proverb, e.g. a poor person can not help another poor person...)
```

The `(proverbe, par ex. ...)` parenthetical contains a period, producing an extra segment.

**PDF evidence (page 32):**

```text
Rũndã bɛnd são beoog kurga. Le cache-sexe d'aujourd'hui vaut mieux que la culotte
(ou le pantalon) de demain. proverbe indiquant: une solution concrète...
```

The proverb label `proverbe indiquant:` is NOT in parentheses here. The current filter
`re.search(r'proverb[e]?\s+indiquant|\(proverb', s)` misses it.

**PDF evidence (page 46):**

```text
Bõn-paall wɩɩda a soaba. La chose neuve fait prendre des airs supérieurs.
(proverbe disant: il ne faut pas montrer une fierté exagérée...)
```

**Fix:** Broaden the proverb filter to catch non-parenthesized labels:

```python
re.search(r'\bproverb[e]?\b', s, re.IGNORECASE)
```

Also filter segments that are entirely a parenthetical expansion `(...)`.

---

### C6 — Embedded next-entry header in example block · **1 occurrence** · 🟢 Edge case

**What happens:** The example block for `a1` contains the POS header of the next entry `-a2`. `split_entry` fails to separate them because `-a2` starts with `-`, which is excluded by the leading-lemma regex `^[^\s(]\S*` (which requires the first char to not be a space or `(`).

**Log entry:**
> `A baaba waame. Son père est venu. His father came. -a2 v. le, lui; it, him, her. A koos-a-la zaamẽ...`

**PDF evidence (page 1):**

```
a1   [à]  ...
2 • son, sa ses; his, her. A baaba waame. Son père est venu. His father came.
-a2
v. le, lui; it, him, her. A koos-a-la zaamẽ. ...
```

**Fix:** Allow `-word` headwords in `split_entry` / `extract_trailing_lemma`:

```python
re.match(r'^-?[^\s(]\S*', after_dot)
```

---

### C7 — Fragment with no sentence-ending punctuation · **15 occurrences** · 🟢 Low priority

**What happens:** `extract_examples` receives text like `"Poanda bãka"`, `"N wa ne vẽenem"`, `"ll, gg, dd, mm"`. These have 0 or no useful segments.

**Root cause:** These are block-split artifacts — partial Moore words or abbreviation lists that ended up as example text. They include:

- Incomplete headword fragments: `"Poanda bãka"` (a compound noun split at column boundary)
- Phonological notes: `"ll, gg, dd, mm"` (a note about double-consonant graphemes)
- Opening of an example sentence cut mid-block: `"N wa ne vẽenem"` (`"Je suis venu avec de la lumière"`)

**Fix:** Already mostly handled by the guard `if not re.search(r'[.?!]', text): return []`. Remaining 15 cases pass the guard because they contain a period from a following fragment. Stricter filtering after segmentation:
```python
segments = [s for s in segments if len(s.strip()) > 3]
```

---

### C8 — Moore-only example: translation absent · **371 occurrences** · 🟡 Partial fix

**What happens:** `extract_examples` receives a Moore sentence with no French/English translation. It produces 1 segment and fails.

**Sub-types:**

| Sub-type | Approx. count | Description |
|----------|-------------:|-------------|
| C8a — `split_entry` cuts off the French translation | ~150 | `split_entry` mistakes the French sentence for the next lemma; `extract_examples` receives only the Moore sentence |
| C8b — English gloss qualifier | ~80 | `(many instances).`, `(fiercely and long).` — parenthetical appended to English gloss |
| C8c — Inline sub-definition | ~50 | `neerlem bonté; goodness, kindness.` — a sub-definition, not an example |
| C8d — Genuinely untranslated | ~91 | PDF has Moore sentence only, no French/English counterpart |

**PDF evidence — C8a (page 15, `bãmba2`):**

The block received by `parse_dictionary_entries` is:

```text
ces, ceux-ci, celles-ci, ceux-là; these ones, that ones. Bãmba waa ka zaamẽ. Ceux-là sont venus ici hier.
```

`split_entry` scans backward for the last `. ` (period + space). It finds `". Ceux-là..."` at position 75. `"Ceux-là sont venus ici hier."` starts with a capital letter and passes all guards, so it is treated as the next lemma. The guard in `parse_dictionary_entries` that should catch this:

```python
if candidate_lemma and re.search(r"[.?!]\s+", candidate_lemma):
```

requires a period **followed by a space**. But `"Ceux-là sont venus ici hier."` ends the string — no trailing space — so `re.search` returns `None`. The French translation is silently stored as `raw_next_lemma` and then discarded. `extract_examples` receives only `"Bãmba waa ka zaamẽ."` → 1 segment → failure.

**Fix for C8a:** Replace `split_entry` with `extract_trailing_lemma` from `PARSER_PLAN.md §Root cause 2`. A real next-lemma is a bare headword (no spaces between words, no verb/noun structure). `"Ceux-là sont venus ici hier."` contains multiple words with spaces — it would never match as a headword.

**PDF evidence — C8b (page 34 `bidu`):**
> `bidon; can, a 5-gal./25 litre can.` — after the slash split (C3), the remaining `(many instances).` is a parenthetical qualifier that should be appended to the English gloss, not treated as an example.

**Fix for C8b/C8c:** In `parse_complex_definition`, before calling `extract_examples`, check if `definition_part` starts with `(` or a lowercase letter and contains no Moore sentence structure (no diacritics typical of Mooré). If so, append to English gloss instead:

```python
if re.match(r'^[(\[a-z]', definition_part) and not re.search(r'[ãẽĩõũɛɔɩ]', definition_part):
    english_gloss = english_gloss.rstrip() + " " + definition_part
```

**C8d** is a PDF limitation — no fix possible.

**C8d** is a PDF limitation — no fix possible.

---

### P1 — `parse_complex_definition`: no semicolon separator · **40 occurrences** · 🔴 Medium fix

**What happens:** The pattern `^(?P<french>[^;]+;)\s*(?P<english>[^.]+?\.)` requires a `;` between French and English and requires English to end with `.`. When either is missing the pattern fails and the English gloss is left empty.

**Unique patterns and root causes:**

| Pattern (truncated) | Count | Root cause |
|--------------------|------:|------------|
| `itératif.` | 10 | Grammatical label only, no translation provided in PDF |
| `ou, ou bien?; or not?... biifu` | 3 | English ends with `?`; next lemma bleeds in |
| `quoi ?; what?... bõe2 [ṍ]` | 3 | Same — next lemma in definition block |
| `pourquoi ?; why? bõeedga` | 3 | English ends with `?`; next lemma follows without separator |
| `boulangerie. Category: Building. synonyme:` | 3 | French-only entry (no English gloss in PDF) |
| `quels?, lesquels?; which ones?... bʋta1` | 3 | English ends with `?`; next lemma bleeds in |
| `tiens !, voilà !; here you are! hũl-hũli` | 3 | English ends with `!`; next lemma follows |
| `quand (quel jour?); when (what day?)... rab-wɛɛlẽ` | 3 | Same |
| `rag ...` | 3 | Incomplete / truncated entry in PDF |
| `22:10).` / `22:6).` / `22:13).` | 6 | Bible verse references leaked into example sentences |

**PDF evidence — `bii` (page 34):**
```
bii  interrogatif. ou, ou bien?; or not?, or else?, isn't it?
     A Zã waame bii? Jean est-il venu ? Did John come (or not)?
biifu Plural: bi. Nom. grain d'ose...

```text
`split_entry` fails to separate `biifu` from the `bii` block, so `biifu` bleeds in. The English sentence ends with `?`, not `.`, so the pattern doesn't match.

**PDF evidence — `boulangerie` (page 55):**

```text
bur maneg roogo   Nom. boulangerie.
Category: Building.
synonyme: bur-doogo.
```

This entry has only a French gloss — no English translation in the PDF.

**PDF evidence — `pourquoi` (page 42):**

```text
bõe yĩnga   interrogatif. pourquoi ?; why?
bõeedga     Nom. pépinière; nursery, garden centre.
```

The interrogative entry `pourquoi?` has no example sentence. The English gloss ends with `?`. The next lemma `bõeedga` bleeds in.

**Fix:**

1. Extend the pattern to accept `[.?!]` as English sentence terminator:

   ```python
   pattern = r"^(?P<french>[^;]+;)\s*(?P<english>[^.?!]+?[.?!])\s*(?P<remaining>.*)"
   ```

2. Before matching, strip trailing next-lemma bleed (bare word + optional `[IPA]` at very end of block).

---

### Recommended fix priority

| Priority | Category | Effort | Impact |
|----------|----------|--------|--------|
| 🔴 High | **C1** Period-only filter | 1 line | −618 warnings |
| 🔴 High | **C3** Comma/slash fragment filter | 2 lines | −117 warnings |
| 🔴 High | **P1** Accept `?` and `!` as English sentence terminator | 5 lines | −40 warnings |
| 🟡 Medium | **C5** Broadened proverb filter | 3 lines | −61 warnings |
| 🟡 Medium | **C4a** Normalize `v.itératif.` before segmentation | 2 lines | −10 warnings |
| 🟡 Medium | **C8b/C8c** Parenthetical qualifiers → English gloss append | 5 lines | ~−130 warnings |
| 🟢 Low | **C2** Stricter orphan-paren filter | 2 lines | −15 warnings |
| 🟢 Low | **C6** Affix lemma (`-word`) in `split_entry` | 3 lines | −1 warning |
| ⚪ None | **C8d** Genuinely untranslated entries | n/a | PDF limitation |
| ⚪ None | **C8a** Column-split translations | large | Requires column-independent parsing (PARSER_PLAN.md §1) |

**Total addressable warnings: ~972 out of 1296 (75%)**
The remaining ~324 require the architectural column-split fix from `PARSER_PLAN.md` or are genuine PDF limitations.
