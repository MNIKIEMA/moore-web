"""Tests for moore_web.simple_parser — focusing on unspec. var. entry handling (S3)."""

from moore_web.simple_parser import (
    clean_text,
    split_first_entry,
    split_dictionary_entries,
    analyze_body,
    parse_page,
    _make_entry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse(raw: str):
    """clean_text → split_first_entry → split_dictionary_entries."""
    text = clean_text(raw)
    _, from_entry = split_first_entry(text)
    return split_dictionary_entries(from_entry)


# ---------------------------------------------------------------------------
# Unit tests: sub-entry numbers must not become lemmas
# ---------------------------------------------------------------------------


class TestSubEntryNumberNotLemma:
    """Lines starting with '2)', '3)' etc. are sub-entry continuations,
    not new dictionary entries — they must never appear as lemmas."""

    def test_digit_paren_not_split_as_entry(self):
        raw = (
            "bãoogo [ã̀-ó] 1) adj Frncalme, en sécurité Engcalm, secure\n"
            "2) adj Frnaligné à la file indienne Englined up in a queue"
        )
        entries = _parse(raw)
        tokens = [t for t, _, _, _ in entries]
        assert "2)" not in tokens, f"'2)' must not be a lemma, got {tokens}"
        assert "bãoogo" in tokens

    def test_all_sub_senses_under_correct_lemma(self):
        raw = (
            "bãoogo [ã̀-ó] 1) adj Frncalme Engcalm\n"
            "2) adj Frnsécurité Engsecurity\n"
            "3) n Frnpaix Engpeace\n"
            "next-word n FrnAutre Engother"
        )
        entries = _parse(raw)
        tokens = [t for t, _, _, _ in entries]
        assert "2)" not in tokens
        assert "3)" not in tokens
        assert "bãoogo" in tokens


# ---------------------------------------------------------------------------
# Unit tests: numbered sub-entries become separate senses (S6)
# ---------------------------------------------------------------------------


class TestSubEntrySenses:
    """Sub-entries '2) grammar Frn...' must produce distinct senses on the
    parent lemma, not bleed into the English field of sense 1."""

    def _get_senses(self, raw: str, lemma: str):
        dicts, _ = parse_page(clean_text(raw))
        entries = [_make_entry(d) for d in dicts]
        entry = next((e for e in entries if e.lemma == lemma), None)
        assert entry is not None, f"{lemma!r} not found"
        return entry.senses

    def test_two_sub_entries_become_two_senses(self):
        """'1) adj Frn... 2) adj Frn...' must yield 2 senses, not 1."""
        raw = (
            "bãoogo [ã̀-ó] 1) adj Frncalme, en sécurité Engcalm, secure\n"
            "2) adj Frnaligné à la file indienne Englined up in a queue"
        )
        senses = self._get_senses(raw, "bãoogo")
        assert len(senses) == 2, f"expected 2 senses, got {len(senses)}: {senses}"

    def test_three_sub_entries_become_three_senses(self):
        """'1) adj ... 2) adj ... 3) n ...' must yield 3 senses."""
        raw = "bãoogo [ã̀-ó] 1) adj Frncalme Engcalm\n2) adj Frnaligné Englined up\n3) n Frnpaix Engpeace"
        senses = self._get_senses(raw, "bãoogo")
        assert len(senses) == 3, f"expected 3 senses, got {len(senses)}"

    def test_sub_entry_grammar_not_in_english_field(self):
        """The '2) adj' token must not appear inside any sense's English field."""
        raw = "bãoogo [ã̀-ó] 1) adj Frncalme Engcalm\n2) adj Frnaligné Englined up"
        senses = self._get_senses(raw, "bãoogo")
        for s in senses:
            assert "2)" not in (s.english or ""), f"'2)' leaked into english: {s.english!r}"
            assert "2)" not in s.french, f"'2)' leaked into french: {s.french!r}"

    def test_sense_content_correct(self):
        """Each sense must carry the right French and English definitions."""
        raw = "bãoogo [ã̀-ó] 1) adj Frncalme Engcalm\n2) n Frnpaix Engpeace"
        senses = self._get_senses(raw, "bãoogo")
        assert len(senses) == 2
        assert "calme" in senses[0].french
        assert "calm" in (senses[0].english or "")
        assert "paix" in senses[1].french
        assert "peace" in (senses[1].english or "")

    def test_plain_two_frn_format_still_works(self):
        """'2) Frn...' format (no grammar tag) must still split correctly."""
        raw = "foo n 1) Frndéfinition un Engdefinition one\n2) Frndéfinition deux Engdefinition two"
        senses = self._get_senses(raw, "foo")
        assert len(senses) == 2

    def test_each_sense_has_correct_pos(self):
        """Each sense must carry its own POS when sub-entries have different grammar tags."""
        raw = "bãoogo [ã̀-ó] 1) adj Frncalme Engcalm\n2) adj Frnaligné Englined up\n3) n Frnpaix Engpeace"
        senses = self._get_senses(raw, "bãoogo")
        assert len(senses) == 3
        assert senses[0].pos == "adj"
        assert senses[1].pos == "adj"
        assert senses[2].pos == "n"

    def test_sense_pos_same_grammar_no_tag_change(self):
        """When all sub-entries share the same POS, each sense.pos equals it."""
        raw = "foo adj 1) adj Frnpremier Engfirst\n2) adj Frndeuxième Engsecond"
        senses = self._get_senses(raw, "foo")
        assert all(s.pos == "adj" for s in senses)

    def test_entry_pos_unchanged(self):
        """DictionaryEntry.pos must remain the grammar of the first sub-entry."""
        raw = "bãoogo [ã̀-ó] 1) adj Frncalme Engcalm\n2) n Frnpaix Engpeace"
        dicts, _ = parse_page(clean_text(raw))
        entries = [_make_entry(d) for d in dicts]
        entry = next(e for e in entries if e.lemma == "bãoogo")
        assert entry.pos == "adj"  # first sub-entry wins at entry level


# ---------------------------------------------------------------------------
# Unit tests: clean_text normalises unspec. var. lines
# ---------------------------------------------------------------------------


class TestCleanTextUnspecVar:
    """clean_text should rewrite 'lemma [tone] unspec. var. of X' into a
    synthetic 'lemma [tone] n Frn (unspec. var. of X) Eng' so that the
    entry splitter can cut there."""

    def test_cat_a_with_tone(self):
        """Cat A: lemma [tone] unspec. var. of X on one line."""
        raw = "ãbe [ã̀] unspec. var. of wãbe"
        out = clean_text(raw)
        assert "Frn (unspec. var. of wãbe) Eng" in out
        assert "ãbe" in out

    def test_cat_a_multiple(self):
        """Two consecutive Cat A entries on the same page."""
        raw = "gʋɩɩna [ʋ́] unspec. var. of gʋɩɩla\ngʋɩɩnde [ʋ́] unspec. var. of gʋɩɩla"
        out = clean_text(raw)
        assert out.count("Frn (unspec. var. of gʋɩɩla)") == 2

    def test_cat_b_no_tone(self):
        """Cat B: lemma unspec. var. of X (no tone bracket)."""
        raw = "barkudi unspec. var. of barkudga"
        out = clean_text(raw)
        assert "barkudi" in out
        assert "Frn (unspec. var. of barkudga) Eng" in out

    def test_cat_b_no_tone_long_lemma(self):
        """Cat B: multi-part lemma, no tone."""
        raw = "na-mao-ne-bɩto unspec. var. of bɩd-maoore"
        out = clean_text(raw)
        assert "Frn (unspec. var. of bɩd-maoore) Eng" in out

    def test_cat_c_split_across_blocks(self):
        """Cat C: lemma and tone are on separate PDF lines."""
        raw = "kʋɩdga\n[ʋ́] unspec. var. of kʋdga"
        out = clean_text(raw)
        assert "kʋɩdga" in out
        assert "Frn (unspec. var. of kʋdga) Eng" in out
        # The newline between lemma and tone must be gone
        assert "kʋɩdga\n" not in out

    def test_cat_d_newline_before_unspec(self):
        """Cat D: newline between lemma and 'unspec. var.'."""
        raw = "lalle\nunspec. var. of lanego (unspec. var. of lalga)"
        out = clean_text(raw)
        assert "lalle" in out
        assert "Frn (unspec. var. of lanego) Eng" in out

    def test_cat_d_nested_parens_stripped(self):
        """Cat D: nested '(unspec. var. of X)' inside target is stripped."""
        raw = "lalle\nunspec. var. of lanego (unspec. var. of lalga)"
        out = clean_text(raw)
        # The nested parenthetical should not survive into the output
        assert "(unspec. var. of lalga)" not in out

    def test_normal_text_unchanged(self):
        """Lines without 'unspec. var.' must not be altered."""
        raw = "wãbe [ã̀] n\nFrnfeuilles vertes Enggreen leaves"
        out = clean_text(raw)
        assert "Frn (unspec. var." not in out


# ---------------------------------------------------------------------------
# Integration tests: entry splitter cuts at unspec. var. boundaries
# ---------------------------------------------------------------------------


class TestSplitUnspecVar:
    """After clean_text, split_dictionary_entries must recognise the synthetic
    stub as a real entry and cut there — preventing bleed into adjacent entries."""

    def test_unspec_entry_is_split_off(self):
        """ãbe must appear as a separate entry, not inside ãase's body."""
        raw = (
            "ãase [ã́-é] v\nFrncasser sur l'arbre Engbreak on a tree\n"
            "ãbe [ã̀] unspec. var. of wãbe\n"
            "ãbga [ã́-à] n\nFrnpuce Engflea"
        )
        entries = _parse(raw)
        tokens = [t for t, _, _, _ in entries]
        assert "ãase" in tokens
        assert "ãbe" in tokens
        assert "ãbga" in tokens

    def test_previous_entry_body_clean(self):
        """ãase's body must not contain 'unspec. var.'."""
        raw = (
            "ãase [ã́-é] v\nFrncasser sur l'arbre Engbreak on a tree\n"
            "ãbe [ã̀] unspec. var. of wãbe\n"
            "ãbga [ã́-à] n\nFrnpuce Engflea"
        )
        entries = _parse(raw)
        ãase_body = next(body for token, _, _, body in entries if token == "ãase")
        assert "unspec. var." not in ãase_body

    def test_unspec_entry_correct_token_and_tone(self):
        """The unspec. var. stub entry must have the right lemma and tone."""
        raw = "ãase [ã́-é] v\nFrncasser sur l'arbre Engbreak on a tree\nãbe [ã̀] unspec. var. of wãbe\n"
        entries = _parse(raw)
        ãbe = next((t, tone, gram, body) for t, tone, gram, body in entries if t == "ãbe")
        assert ãbe[1] == "ã̀"  # tone preserved
        assert "unspec. var. of wãbe" in ãbe[3]  # target in body

    def test_two_consecutive_unspec_both_split(self):
        """Two consecutive unspec. var. entries must both be split off."""
        raw = (
            "gʋɩɩla [ʋ́-à] n\nFrnarbuste Engshrub\n"
            "gʋɩɩna [ʋ́] unspec. var. of gʋɩɩla\n"
            "gʋɩɩnde [ʋ́] unspec. var. of gʋɩɩla\n"
            "gʋɩɩnga [ʋ́-à] n\nFrnarbuste (espèce) Engshrub sp"
        )
        entries = _parse(raw)
        tokens = [t for t, _, _, _ in entries]
        assert "gʋɩɩna" in tokens
        assert "gʋɩɩnde" in tokens
        gʋɩɩla_body = next(body for t, _, _, body in entries if t == "gʋɩɩla")
        assert "gʋɩɩna" not in gʋɩɩla_body

    def test_cat_b_no_tone_split(self):
        """Cat B (no tone) entry must also be split off correctly."""
        raw = (
            "barkudga [à-ú-à] n\nFrnArbre Engtree\n"
            "barkudi unspec. var. of barkudga\n"
            "barsda n\nFrnmarchandeur Engbargainer"
        )
        entries = _parse(raw)
        tokens = [t for t, _, _, _ in entries]
        assert "barkudi" in tokens
        barkudga_body = next(body for t, _, _, body in entries if t == "barkudga")
        assert "barkudi" not in barkudga_body


# ---------------------------------------------------------------------------
# Sense-level tests: unspec. var. stubs produce variant, not a real sense
# ---------------------------------------------------------------------------


class TestUnspecVarSense:
    """analyze_body on a unspec. var. stub should yield no meaningful senses
    (the variant pointer is handled upstream in _make_entry)."""

    def test_unspec_stub_french_field(self):
        """The French field of the stub should be the '(unspec. var. of X)' marker."""
        stub_body = "Frn (unspec. var. of wãbe) Eng"
        senses = analyze_body(stub_body)
        # There should be one block with one sense
        assert len(senses) == 1
        sense = senses[0][0]
        assert "unspec. var. of wãbe" in sense.get("fr_entry", "")


# ---------------------------------------------------------------------------
# End-to-end: parse_page produces correct DictionaryEntry for unspec. var.
# ---------------------------------------------------------------------------


class TestParsePageUnspecVar:
    """parse_page must turn unspec. var. stubs into entries with no senses
    and the target recorded as a variant."""

    def _get_entry(self, raw: str, lemma: str):
        dicts, _ = parse_page(clean_text(raw))
        entries = [_make_entry(d) for d in dicts]
        return next((e for e in entries if e.lemma == lemma), None)

    def test_unspec_entry_has_no_senses(self):
        """An unspec. var. entry must produce zero senses."""
        raw = "ãase [ã́-é] v\nFrncasser sur l'arbre Engbreak on a tree\nãbe [ã̀] unspec. var. of wãbe\n"
        entry = self._get_entry(raw, "ãbe")
        assert entry is not None, "ãbe entry not found"
        assert entry.senses == [], f"expected no senses, got {entry.senses}"

    def test_unspec_entry_target_in_variants(self):
        """The target lemma must appear in the variants dict."""
        raw = "ãase [ã́-é] v\nFrncasser sur l'arbre Engbreak on a tree\nãbe [ã̀] unspec. var. of wãbe\n"
        entry = self._get_entry(raw, "ãbe")
        assert entry is not None
        assert entry.variants is not None, "variants should not be None"
        all_variants = " ".join(str(v) for v in entry.variants.values())
        assert "wãbe" in all_variants

    def test_unspec_pos_not_synthetic_n(self):
        """The pos must not be 'n' from the synthetic stub (should be empty or omitted)."""
        raw = "ãase [ã́-é] v\nFrncasser sur l'arbre Engbreak on a tree\nãbe [ã̀] unspec. var. of wãbe\n"
        entry = self._get_entry(raw, "ãbe")
        assert entry is not None
        assert entry.pos != "n", "pos should not be the synthetic 'n' injected by the pre-processor"

    def test_previous_entry_senses_clean(self):
        """The preceding entry must have its own senses intact and no unspec bleed."""
        raw = "ãase [ã́-é] v\nFrncasser sur l'arbre Engbreak on a tree\nãbe [ã̀] unspec. var. of wãbe\n"
        entry = self._get_entry(raw, "ãase")
        assert entry is not None
        assert len(entry.senses) == 1
        assert "unspec. var." not in entry.senses[0].french
        assert "unspec. var." not in (entry.senses[0].english or "")


# ---------------------------------------------------------------------------
# S5 — <Not Sure> grammar tag with re.VERBOSE
# ---------------------------------------------------------------------------


class TestNotSureGrammar:
    """S5: <Not Sure> grammar tag must be recognised even inside VERBOSE-mode regexes."""

    def _get_entry(self, raw: str, lemma: str):
        dicts, _ = parse_page(clean_text(raw))
        entries = [_make_entry(d) for d in dicts]
        return next((e for e in entries if e.lemma == lemma), None)

    def test_not_sure_entry_is_parsed(self):
        raw = "bala <Not Sure> Frnseulement Engonly"
        entry = self._get_entry(raw, "bala")
        assert entry is not None, "<Not Sure> entry not parsed at all"

    def test_not_sure_pos_preserved(self):
        raw = "bala <Not Sure> Frnseulement Engonly"
        entry = self._get_entry(raw, "bala")
        assert entry is not None
        assert entry.pos == "<Not Sure>"

    def test_not_sure_senses_intact(self):
        raw = "bala <Not Sure> Frnseulement Engonly"
        entry = self._get_entry(raw, "bala")
        assert entry is not None
        assert len(entry.senses) == 1
        assert entry.senses[0].french == "seulement"
        assert entry.senses[0].english == "only"
