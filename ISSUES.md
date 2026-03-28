
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

### S7 — `ﬁ` ligature (U+FB01) extracted as `ϧ` (U+03E7) · fixed · 🔴 High

**What happens:** All Moore lemmas beginning with `fi` are stored in output.jsonl with `ϧ`
instead of `fi` (e.g. `ϧka`, `ϧlm`, `ϧrgo`, `ϧsiye` …). This affects ~20 entries around
output.jsonl lines 1254–1274 (PDF pages 192–193).

**Root cause:** The PDF font's `ToUnicode` table maps its `fi`-ligature glyph directly to
`ϧ` (U+03E7, COPTIC SMALL LETTER HAI) instead of `ﬁ` (U+FB01) or the two-character
sequence `fi`. PyMuPDF faithfully emits whatever the font's encoding says, so the Coptic
character comes through verbatim.

**PDF evidence (pages 192–193):**

```
fika  [ĩ-á]  n   Frn éventail  Eng fan          → ϧka   in output
film  n       Frn film  Eng movie                 → ϧlm   in output
firgo n       Frn réfrigérateur  Eng refrigerator → ϧrgo  in output
fisiye n      Frn fichier  Eng file               → ϧsiye in output
```

**Fix:** Add `"\u03E7": "fi"` to `_LIGATURE_MAP` in `pdf_extractor.py`, alongside the
standard U+FB01 entries. `_expand_ligatures()` is called from both `pdf_extractor.py`
return sites and at the top of `simple_parser.clean_text()`.

---

### S6 — `2) grammar Frn` sub-entries not split into separate senses · patched · 🔴 High

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

**Patch applied (temporary workaround — not a full fix):**

The three known artifact entries produced by this bug are handled in `parse_page`
via `_s6_clean_token` / `_s6_is_garbage`:

| Artifact | Action | Reason |
| -------- | ------ | ------ |
| `"poorẽ1)"` | cleaned → `"poorẽ"` | trailing sub-entry index bled into lemma |
| `"Frnnous Engwe"` | dropped | `Frn…Eng` definition text mistaken for a headword |
| `"about what?"` | dropped | punctuation fragment, not a valid Moore lemma |

**What still needs to be solved:**
The regex fix above (`SUB_SPLIT_RE`) was merged but `split_sub_entries` still fails
for sub-entry patterns not covered by `GRAMMAR_PATTERN` or where the grammar tag is
absent. Until every `N) [grammar] Frn` variant is handled at the split level, other
entries of the same shape may produce similar artifacts that the patch won't catch.
The patch must be removed once the root-cause fix is verified to be complete.

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
{"lemma":"Frnnous Engwe","ipa":"","pos":"<Not Sure>"}
```

Other

```json
{"lemma":"gẽ[é]","ipa":"","pos":"v"}
```
