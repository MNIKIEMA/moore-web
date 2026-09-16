"""Tests for moore_web.flatten — focusing on _join_lines and the missing-space fix."""

import pytest

from moore_web.flatten import AlignedCorpus, _join_lines, flat_rows_to_long, normalize_fr, normalize_mo, segment_fr


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


class TestFlatRowsToLong:
    def test_fra_source_puts_french_as_source_text(self):
        rows = flat_rows_to_long([{"french": "Bonjour.", "moore": "Ne y sõma.", "laser_score": 0.8}], "kade")
        assert rows == [
            {
                "id": "kade-000000",
                "src_lang": "fra",
                "tgt_lang": "mos",
                "source_text": "Bonjour.",
                "target_text": "Ne y sõma.",
                "is_source_orig": True,
                "doc_id": None,
                "source": "kade",
                "laser_score": 0.8,
            }
        ]

    def test_mos_source_puts_moore_as_source_text(self):
        rows = flat_rows_to_long([{"french": "chat", "moore": "bagre", "laser_score": 1.0}], "moore-fr-eng-dictionary")
        row = rows[0]
        assert row["src_lang"] == "mos"
        assert row["tgt_lang"] == "fra"
        assert row["source_text"] == "bagre"
        assert row["target_text"] == "chat"
        assert row["is_source_orig"] is True

    def test_unknown_source_has_null_is_source_orig(self):
        rows = flat_rows_to_long([{"french": "a", "moore": "b", "laser_score": None}], "mystery-source")
        assert rows[0]["is_source_orig"] is None

    def test_english_triplet_becomes_two_rows_sharing_id(self):
        rows = flat_rows_to_long(
            [{"french": "chat", "moore": "bagre", "english": "cat", "laser_score": 1.0}], "moore-fr-eng-dictionary"
        )
        assert len(rows) == 2
        assert rows[0]["id"] == rows[1]["id"] == "moore-fr-eng-dictionary-000000"
        assert (rows[0]["tgt_lang"], rows[1]["tgt_lang"]) == ("fra", "eng")
        assert rows[1]["source_text"] == "bagre"
        assert rows[1]["target_text"] == "cat"

    def test_no_english_is_single_row(self):
        rows = flat_rows_to_long([{"french": "a", "moore": "b", "laser_score": 1.0}], "moore-fr-eng-dictionary")
        assert len(rows) == 1

    def test_all_none_scores_omit_laser_score_field(self):
        rows = flat_rows_to_long(
            [{"french": "a", "moore": "b", "laser_score": None}, {"french": "c", "moore": "d", "laser_score": None}],
            "kade",
        )
        assert all("laser_score" not in r for r in rows)

    def test_ids_increment_per_row(self):
        rows = flat_rows_to_long(
            [{"french": "a", "moore": "b", "laser_score": 1.0}, {"french": "c", "moore": "d", "laser_score": 1.0}],
            "kade",
        )
        assert [r["id"] for r in rows] == ["kade-000000", "kade-000001"]

    def test_doc_id_exposed_as_field(self):
        rows = flat_rows_to_long(
            [{"french": "a", "moore": "b", "laser_score": 1.0, "doc_id": "2024-07-24"}], "conseils"
        )
        assert rows[0]["doc_id"] == "2024-07-24"

    def test_id_carries_doc_ordinal_and_local_index(self):
        rows = flat_rows_to_long(
            [
                {"french": "a1", "moore": "b1", "laser_score": 1.0, "doc_id": "2024-07-24"},
                {"french": "a2", "moore": "b2", "laser_score": 1.0, "doc_id": "2024-07-24"},
                {"french": "a3", "moore": "b3", "laser_score": 1.0, "doc_id": "2024-07-31"},
            ],
            "conseils",
        )
        assert [r["id"] for r in rows] == [
            "conseils-000000-000",
            "conseils-000000-001",
            "conseils-000001-000",
        ]

    def test_doc_id_local_index_resets_across_interleaved_docs(self):
        # Same doc_id appearing again after a different one in between still
        # continues that doc's own running count, not the global row index.
        rows = flat_rows_to_long(
            [
                {"french": "a1", "moore": "b1", "laser_score": 1.0, "doc_id": "url-a"},
                {"french": "a2", "moore": "b2", "laser_score": 1.0, "doc_id": "url-b"},
                {"french": "a3", "moore": "b3", "laser_score": 1.0, "doc_id": "url-a"},
            ],
            "raamde-news",
        )
        assert rows[0]["id"] == "raamde-news-000000-000"
        assert rows[1]["id"] == "raamde-news-000001-000"
        assert rows[2]["id"] == "raamde-news-000000-001"

    def test_no_doc_id_falls_back_to_flat_scheme(self):
        rows = flat_rows_to_long([{"french": "a", "moore": "b", "laser_score": 1.0}], "kade")
        assert rows[0]["id"] == "kade-000000"
        assert rows[0]["doc_id"] is None

    def test_english_row_shares_doc_id(self):
        rows = flat_rows_to_long(
            [{"french": "chat", "moore": "bagre", "english": "cat", "laser_score": 1.0, "doc_id": "entry-42"}],
            "moore-fr-eng-dictionary",
        )
        assert rows[0]["doc_id"] == rows[1]["doc_id"] == "entry-42"
        assert rows[0]["id"] == rows[1]["id"]


class TestAlignedCorpusToJsonlRows:
    def test_matches_flat_rows_to_long(self):
        aligned = AlignedCorpus(french=["Bonjour."], moore=["Ne y sõma."], scores=[0.8], source="kade")
        assert aligned.to_jsonl_rows() == flat_rows_to_long(
            [{"french": "Bonjour.", "moore": "Ne y sõma.", "laser_score": 0.8}], "kade"
        )

    def test_includes_english_when_present(self):
        aligned = AlignedCorpus(
            french=["chat"], moore=["bagre"], english=["cat"], scores=[1.0], source="moore-fr-eng-dictionary"
        )
        rows = aligned.to_jsonl_rows()
        assert len(rows) == 2
        assert rows[1]["target_text"] == "cat"

    def test_includes_doc_ids_when_present(self):
        aligned = AlignedCorpus(
            french=["a", "c"],
            moore=["b", "d"],
            scores=[1.0, 1.0],
            doc_ids=["2024-07-24", "2024-07-31"],
            source="conseils",
        )
        rows = aligned.to_jsonl_rows()
        assert [r["doc_id"] for r in rows] == ["2024-07-24", "2024-07-31"]
        assert [r["id"] for r in rows] == ["conseils-000000-000", "conseils-000001-000"]

    def test_mismatched_doc_ids_length_raises(self):
        with pytest.raises(ValueError, match="doc_ids"):
            AlignedCorpus(french=["a", "c"], moore=["b", "d"], scores=[1.0, 1.0], doc_ids=["only-one"], source="x")


class TestAlignedCorpusWriteJsonl:
    def test_single_pair_writes_one_file(self, tmp_path):
        aligned = AlignedCorpus(french=["Bonjour."], moore=["Ne y sõma."], scores=[0.8], source="kade")
        out = tmp_path / "kade_aligned.jsonl"
        written = aligned.write_jsonl(str(out))
        assert written == [str(out)]
        assert out.exists()

    def test_mixed_lang_pairs_split_into_separate_files(self, tmp_path):
        aligned = AlignedCorpus(
            french=["chat", "eau"], moore=["bagre", "koom"], english=["cat", ""], scores=[1.0, 1.0], source="moore-fr-eng-dictionary"
        )
        out = tmp_path / "simple_aligned.jsonl"
        written = aligned.write_jsonl(str(out))

        assert not out.exists()
        assert sorted(written) == sorted(
            [str(tmp_path / "simple_aligned.mos-fra.jsonl"), str(tmp_path / "simple_aligned.mos-eng.jsonl")]
        )

        fra_lines = (tmp_path / "simple_aligned.mos-fra.jsonl").read_text(encoding="utf-8").splitlines()
        eng_lines = (tmp_path / "simple_aligned.mos-eng.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(fra_lines) == 2
        assert len(eng_lines) == 1
