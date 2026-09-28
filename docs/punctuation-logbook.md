# Punctuation logbook

How the reviewed text is normalized after alignment review, source by source.
Each source gets an inventory (measured, not guessed), the rules applied in
order, a list of manual fixes, and a dated log of what was done.

The rules live in `src/moore_web/punctuation.py` and run when the dataset is
built (`build_fr_mos_dataset.py`), for sources with
`normalize_punctuation = true` in `fr_mos_sources.toml`. The review DB and the
reviewed export keep exactly what was reviewed.

Principles, for every source:

- **Same rule on both sides.** A rule applies to French and Mooré alike; the
  normalizer never harmonizes one language to the other (`!` vs `.`,
  `?` vs `.` are translation choices and stay).
- **`original_fra` / `original_mos` are never touched.**
- **Rules for the typography, the review app for the rest.** A wrong line
  pairing or a quote closed in the wrong place needs a judgement; those are
  listed under *Manual fixes* and done in the app.
- **Measure before and after.** Every rule has a count in the inventory; the
  log records the count after the run.

## mos-contes-volume-5

Snapshot 2026-09-28: 30 units, all reviewed, 865 kept pairs. Counts come from
the kept (non-rejected) lines of the review DB. Unit names below drop the
`mos-contes-volume-5-` prefix (`20` = `mos-contes-volume-5-20`); line numbers
are kept-line numbers, as in the editor when nothing is rejected.

### Inventory

| # | Issue | fr | mos | Rule |
| --- | --- | --- | --- | --- |
| 1 | Punctuation after the closing quote: `»?` `»!` `».` | 158 | 185 | P2 |
| 1a | … of which with a space: `» ?` `» !` `» .` | 29 | 18 | P2 |
| 1b | … of which the quote is embedded in a sentence (`s'appelle « patagsde ».`), where `».` is correct | 3 | 3 | P2 keeps |
| 2 | Punctuation on both sides of »: `! ».` `.».` | 2 | 2 | P2 |
| 3 | Quote split across links: opens but doesn't close on the line | 44 | 42 | P3 |
| 4 | … closes but doesn't open on the line | 47 | 46 | P3 |
| 5 | … line inside a quote with no guillemet at all | 68 | 49 | P3 |
| 6 | Units with a stray or unclosed quote in the source | 6 | 4 | P3, then manual |
| 7 | Link ends with `,` `;` `:` (cut made during review, or intro split from its quote) | 5 | 5 | P0 / P4 |
| 8 | Link has no final punctuation (not a title) | 0 | 3 | P4 |
| 9 | Titles (line 1) ending with `.` on one side only | 0 | 4 | P4 |
| 10 | Link starts with a lowercase letter | 1 | 1 | P4 |
| 11 | Spacing: `word?` vs majority `word ?` | 21 | 22 | P1 |
| 12 | Spacing: `word:` vs majority `word :` | 17 | 9 | P1 |
| 13 | Spacing: `«word` / `word»` vs majority `« word` / `word »` | 19 / 57 | 17 / 43 | P1 |
| 14 | Space before `.` or `,` | 10 | 5 | P1 |
| 15 | Straight quotes `"` | 4 lines | 5 lines | not changed |

No non-breaking spaces occur anywhere in the source.

### Rules, in order

**P0 — manual fixes in the review app.** Everything in
[Manual fixes](#manual-fixes) below. Not required before the rules run (P3
also removes stray guillemets), but the rules can't repair a line paired with
the wrong translation.

**P1 — spacing.** Make the majority style universal, with ordinary spaces
(no NBSP: the source has none, and it would be invisible in the app):

- one space before `?` `!` `:` `;` → `word ?`, `word :`
- one space inside guillemets → `« word`, `word »`
- no space before `.` and `,` → `vieux .` → `vieux.`; a space after a comma
  followed by a letter → `est ,vraiment` → `est, vraiment` (not `1,5`)
- collapse repeated spaces; `10:30` keeps its colon.

**P2 — punctuation next to a closing ».** The quotes here are dialogue: a
full sentence spoken by a character. French typography puts that sentence's
own `?` `!` `.` inside the guillemets, and nothing after them.

- `x »?` / `x » ?` → `x ? »` (same for `!`)
- `x ».` → `x. »`, **unless the quote is embedded** in a running sentence
  (it neither starts the line nor follows `:` or a speech-tag comma). Those
  3 + 3 lines keep `».`: *Depuis cette époque, le singe s'appelle toujours
  « patagsde ».*
- Double punctuation: `! ».` → `! »`, `.».` → `. »` (drop the outer mark).
- Straight `"` quotes are left alone (lines 15 in the inventory).

**P3 — drop the guillemets left alone on a line.** Review split long quotes
into several links (see *Reported speech* in the review guidelines): the first
piece keeps only «, the last only ». Each line is checked on its own: a « or »
whose partner is not on the same line is removed, together with its space.
Nothing is added, and no state is carried from line to line.

| Line | Before | After |
| --- | --- | --- |
| first piece | `Gomtɩʋʋg yeela ninsaala : « M pʋʋsd-f-la bark wʋsg …` | `Gomtɩʋʋg yeela ninsaala : M pʋʋsd-f-la bark wʋsg …` |
| first piece | `« Ah, voilà mon vieux qui vient !` | `Ah, voilà mon vieux qui vient !` |
| last piece | `… continuera vivre en bonne santé».` | `… continuera vivre en bonne santé.` |
| whole quote | `« Ah, c'est les œufs de ma femme Poko. … les avoir vus »!` | `« Ah, c'est les œufs de ma femme Poko. … les avoir vus ! »` (P2) |
| embedded | `wãamb yʋʋr lebga «a pa tagsde».` | `wãamb yʋʋr lebga « a pa tagsde ».` (P1, P2 keeps `».`) |

When removing » leaves two marks (`morte ! »!` → `morte ! !`), one is kept,
`?` or `!` before `.`. Middle pieces carry no guillemet and are not changed.

Side effects to know:

- A quote that resumes after a speech tag loses its reopening «:
  *« Hyène ! » dit le chien de brousse, ce sale truc-là…* (`20` line 28). The
  line still reads correctly.
- A source error that closes a quote too early (*« Moi je n'ai pas de
  vitesse ». Mais…*, `08` line 13) comes out well-formed but with the quote
  still ending in the wrong place: fix it in the app (manual list).

**P4 — link boundaries.**

- A link ending with `,` or `;` (a sentence cut during review) ends with `.`
  instead; if the next link starts with a lowercase letter, capitalize it.
  (`25` line 13–14: *…en voulait une deuxième,* / *le dernier n'avait pas…*)
- A link ending with `:` whose quote is **not** on the other side of the
  pair ends with `.` (`22` line 14 mos, `24` line 4 fr, `24` line 23 mos).
  When both sides end with `:` the intro was separated from its quote: that
  is a P0 join, not a punctuation fix.
- A link with no final punctuation gets `.`, placed before a closing »
  (`24` line 10 mos, `27` line 28 mos).
- Titles (line 1 of a unit) carry no final `.` on either side (`16`, `20`,
  `21`, `27` mos).

**Not changed:** straight `"`; `!`/`?`/`.` differences between the two sides;
`…` vs `...`; wording.

### Decisions

- **D1 — rules run at dataset build time, per source.** Decided 2026-09-28.
  The review DB and the reviewed export (pinned on the Hub) stay exactly as
  reviewed, a rule can be fixed and re-run without a backup/restore, and a
  document-level export can rebuild paragraphs from the un-normalized text.
  Row ids come from unit and line, so they don't change.
- **D2 — drop lone guillemets rather than add the missing ones.** Decided
  2026-09-28. Nothing is added to the text, each line is checked on its own
  (no cross-line state, so no wrong « … » when a source quote is broken), and
  a line reads the same with or without the quote marks.
- **D3 — no NBSP, ordinary spaces (P1).** Applied 2026-09-28 (the source
  has no NBSP).
- **D4 — straight quotes stay.** Decided 2026-09-28.

### Manual fixes

Done in the review app; they take effect at the next reviewed export. P3
already removes the stray marks, so these only matter where the quote ends in
the wrong place or a word is on the wrong line.

Stray or unclosed quotes (inventory 6):

- [ ] `03` line 17 fr — quote closed too early: *« Qu'est-ce qu'il y a »?
      Pourquoi … »?* → one quote, `»` only at the end.
- [ ] `04` line 46 fr — *demanda ; «Qui es-tu ? «* → *demanda : « Qui es-tu ? »*
- [ ] `05` line 32 mos — opens with straight `"` but closes with » on line 34:
      use « on line 32 (the only straight quote that is half of a pair).
- [ ] `08` line 13 fr — *« Moi je n'ai pas de vitesse ». Mais j'ai…* : quote
      closed too early; Mooré has one quote for the whole turn.
- [ ] `09` line 22 fr — missing opening «: *Non, dit l'hyène, …*
      (Mooré: *« Ayo », katr sẽn yeele, « … »*).
- [ ] `09` line 53 mos — missing opening «: *La katr bee be bõe yĩnga »?*
- [ ] `13` line 19 mos — extra text after the quote closes: *…m dẽnda »!
      Wakat ninga, zu-loees n be n yɩɩd fo zu-loeesã! »* has no French on
      this line; check whether it belongs to line 20 (alignment, not only
      punctuation).
- [ ] `17` line 24 fr — the quote opened here is never closed in the unit.
- [ ] `19` line 5 mos — stray » at the end: *…la a sẽn gomdã »*.
- [ ] `24` line 5 fr — stray » at the end (no quote opened; line 4 ends with
      `:` for no quote).

Alignment errors found while counting (the quote opens on the wrong line):

- [ ] `21` lines 10–11 — *« Ah !* ends French 10 but its Mooré *« Ha,*
      starts Mooré 11. Move *« Ah !* to the start of French 11.
- [ ] `21` lines 16–17 — same pattern: move *« Ah !* from the end of French 16
      to the start of French 17.

Intro separated from its quote (join the intro with the next line, per the
review guidelines):

- [ ] `03` lines 20–21 — both sides end with `:`.
- [ ] `22` lines 16–17 — fr *il ajoute;* (read `:`), mos *la a yeele :*.
- [ ] `22` lines 18–19 — both sides end with `:`.

OCR:

- [ ] `27` line 1 fr — *Ue femme* → *Une femme*.

### Log

- **2026-09-28** — Inventory measured on the reviewed DB (865 pairs). Rules
  P0–P4 written.
- **2026-09-28** — Implemented in `src/moore_web/punctuation.py` (tests in
  `tests/test_punctuation.py`), enabled for this source with
  `normalize_punctuation = true` and `first_line_title = true`. D1 and D2
  decided as above. Checked on a local export built end to end: 865 rows
  loaded, 864 kept (one duplicate pair already in the source, story 21).
  572 of 1 730 sides change. After the run:

  | Count | fr before → after | mos before → after |
  | --- | --- | --- |
  | punctuation after » | 155 → 3 | 183 → 3 |
  | lines with a lone guillemet | 90 → 0 | 88 → 0 |
  | link ends with `,` `;` `:` | 5 → 0 | 5 → 0 |
  | `word?` / `word:` without space | 40 → 0 | 32 → 0 |
  | `«word` / `word»` | 19 / 57 → 0 / 0 | 17 / 43 → 0 / 0 |
  | space before `.` `,` | 12 → 0 | 7 → 0 |
  | non-title line without final punctuation | 0 → 0 | 3 → 0 |

  The 3 + 3 left after » are the embedded quotes (`03` 33, `19` 8, `30` 25),
  which keep `».` on purpose. Not yet in the published dataset: that needs
  `moore-web export-reviewed --push`, the new sha in `[reviewed].revision`,
  and a rebuild. Manual fixes not done yet.
