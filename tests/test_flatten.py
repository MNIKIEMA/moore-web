"""Tests for moore_web.flatten — focusing on _join_lines and the missing-space fix."""

from moore_web.flatten import _join_lines, normalize_fr, normalize_mo, segment_fr


class TestJoinLines:
    def test_collapses_single_newline(self):
        assert _join_lines("hello\nworld") == "hello world"

    def test_collapses_multiple_newlines(self):
        assert _join_lines("hello\n\n\nworld") == "hello world"

    def test_collapses_multiple_spaces(self):
        assert _join_lines("hello   world") == "hello world"

    def test_strips_leading_trailing(self):
        assert _join_lines("  hello  ") == "hello"

    def test_empty_string(self):
        assert _join_lines("") == ""

    # --- missing-space fix ---

    def test_inserts_space_after_period_before_capital(self):
        assert _join_lines("smartphones.Les causes ne sont pas connues.") == (
            "smartphones. Les causes ne sont pas connues."
        )

    def test_inserts_space_after_exclamation_before_capital(self):
        assert _join_lines("Incroyable!Elle est arrivée.") == "Incroyable! Elle est arrivée."

    def test_inserts_space_after_question_mark_before_capital(self):
        assert _join_lines("Qui est là?Personne ne répond.") == "Qui est là? Personne ne répond."

    def test_multiple_missing_spaces(self):
        assert _join_lines("phrase un.Phrase deux.Phrase trois.") == ("phrase un. Phrase deux. Phrase trois.")

    def test_already_spaced_unchanged(self):
        assert _join_lines("Il est parti. Elle est restée.") == "Il est parti. Elle est restée."

    # --- acronyms must not be broken ---

    def test_acronym_faso_unchanged(self):
        assert _join_lines("le F.A.S.O. est souverain.") == "le F.A.S.O. est souverain."

    def test_all_caps_sequence_unchanged(self):
        assert _join_lines("BURKINA.F.A.S.O.") == "BURKINA.F.A.S.O."

    def test_section_label_unchanged(self):
        text = (
            "E. AU TITRE DU MINISTERE DES AFFAIRES ETRANGERES, "
            "DE LA COOPERATION REGIONALE ET DES BURKINABE DE L'EXTERIEUR"
        )
        assert _join_lines(text) == text

    def test_section_label_no_space_variant_unchanged(self):
        # E.AU — A followed by U (uppercase-uppercase) → must not gain a space
        assert _join_lines("E.AU TITRE DU MINISTERE") == "E.AU TITRE DU MINISTERE"

    # --- quotation patterns ---

    def test_missing_space_inside_guillemets(self):
        assert _join_lines("«phrase un.Phrase deux»") == "«phrase un. Phrase deux»"

    def test_missing_space_inside_typographic_quotes(self):
        assert _join_lines("\u201cphrase un.Phrase deux\u201d") == "\u201cphrase un. Phrase deux\u201d"

    def test_closing_guillemet_then_capital_unchanged(self):
        # »Le — » is not in [.!?], so no space inserted
        assert _join_lines("«phrase un.»Le lendemain") == "«phrase un.»Le lendemain"

    def test_dot_before_opening_guillemet_unchanged(self):
        # dot before «, not before [A-Z]
        assert _join_lines("Il dit.«Bonjour»") == "Il dit.«Bonjour»"

    # --- honorifics ---

    def test_honorific_m_gains_space(self):
        # M.Traoré → M. Traoré is correct French typography
        assert _join_lines("M.Traoré a dit cela.") == "M. Traoré a dit cela."

    def test_honorific_dr_gains_space(self):
        assert _join_lines("Dr.Kaboré a été nommé.") == "Dr. Kaboré a été nommé."

    # --- Mooré text ---

    def test_moore_missing_space(self):
        assert _join_lines("A yibeogo.Yaa ne taaba.") == "A yibeogo. Yaa ne taaba."

    def test_moore_all_caps_unchanged(self):
        # uppercase followed by uppercase — no space inserted
        assert _join_lines("A yibeogo.YAA ne taaba.") == "A yibeogo.YAA ne taaba."

    # --- French accented capitals ---

    def test_accented_capital_gains_space(self):
        assert _join_lines("fin.Écoles fermées.") == "fin. Écoles fermées."


# ---------------------------------------------------------------------------
# normalize_fr
# ---------------------------------------------------------------------------


class TestNormalizeFr:
    def test_curly_double_quotes_converted(self):
        assert normalize_fr("\u201cBonjour\u201d") == '"Bonjour"'

    def test_space_before_punctuation_removed(self):
        assert normalize_fr("Bonjour !") == "Bonjour!"

    def test_space_inside_guillemets_removed(self):
        assert normalize_fr("« Bonjour »") == "«Bonjour»"

    def test_multiple_spaces_collapsed(self):
        assert normalize_fr("hello   world") == "hello world"

    def test_strips(self):
        assert normalize_fr("  hello  ") == "hello"


# ---------------------------------------------------------------------------
# normalize_mo
# ---------------------------------------------------------------------------


class TestNormalizeMo:
    def test_curly_double_quotes_converted(self):
        assert normalize_mo("\u201cBonjour\u201d") == '"Bonjour"'

    def test_multiple_spaces_collapsed(self):
        assert normalize_mo("a   b") == "a b"

    def test_strips(self):
        assert normalize_mo("  yaa  ") == "yaa"


# ---------------------------------------------------------------------------
# segment_fr: basic smoke tests (syntok integration)
# ---------------------------------------------------------------------------


class TestSegmentFr:
    def test_single_sentence(self):
        result = segment_fr("Bonjour le monde.")
        assert len(result) == 1
        assert "Bonjour" in result[0]

    def test_two_sentences(self):
        result = segment_fr("Il est parti. Elle est restée.")
        assert len(result) == 2

    def test_empty_string(self):
        result = segment_fr("")
        assert result == []

    def test_glued_sentences_split(self):
        # _join_lines is called inside segment_fr, so the fix applies
        result = segment_fr("smartphones.Les causes ne sont pas connues.")
        assert len(result) == 2
