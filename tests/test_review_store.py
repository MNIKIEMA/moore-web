import json

import pytest

from moore_web import review_store


def test_reviewers_keep_separate_drafts_and_conflicts_do_not_overwrite(tmp_path):
    source = tmp_path / "sample_units.jsonl"
    source.write_text(
        json.dumps({"page-3": {"fra": ["Bonjour.", "Bonsoir."], "mos": ["Ney yibeogo."]}}) + "\n",
        encoding="utf-8",
    )
    db = tmp_path / "review.sqlite3"
    review_store.initialize(db)
    assert review_store.import_units(db, [source]) == 1
    assert review_store.import_units(db, [source]) == 0
    unit = review_store.list_units(db)[0]
    assert unit["reviewed"] == 0

    review_store.save_draft(db, unit["id"], "Ada", "Bonjour.\nBonsoir.", "Ney yibeogo.\nWend na yis.", 0)
    review_store.save_draft(db, unit["id"], "Benoît", "Bonjour.\nBonsoir.", "Ney yibeogo.", 0)
    assert review_store.get_draft(db, unit["id"], "Ada")["mos_text"].endswith("Wend na yis.")
    assert review_store.get_draft(db, unit["id"], "Benoît")["mos_text"] == "Ney yibeogo."

    with pytest.raises(ValueError):
        review_store.accept_review(db, unit["id"], "Benoît", "Bonjour.\nBonsoir.", "Ney yibeogo.", 0)

    review_store.accept_review(db, unit["id"], "Ada", "Bonjour.\nBonsoir.", "Ney yibeogo.\nWend na yis.", 0)
    with pytest.raises(review_store.ReviewConflict):
        review_store.accept_review(db, unit["id"], "Benoît", "Bonjour.", "Ney yibeogo.", 0)
    assert review_store.get_draft(db, unit["id"], "Benoît") is not None
    assert review_store.export_pairs(db) == [
        {"french": "Bonjour.", "moore": "Ney yibeogo.", "source": "sample", "unit": "page-3"},
        {"french": "Bonsoir.", "moore": "Wend na yis.", "source": "sample", "unit": "page-3"},
    ]


def test_summary_filters_and_pagination(tmp_path):
    source_a = tmp_path / "alpha_units.jsonl"
    source_a.write_text(
        "\n".join(
            [
                json.dumps({"one": {"fra": ["Un."], "mos": ["A."]}}),
                json.dumps({"two": {"fra": ["Deux.", "Trois."], "mos": ["B."]}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    source_b = tmp_path / "beta_units.jsonl"
    source_b.write_text(
        json.dumps({"three": {"fra": ["Quatre."], "mos": ["C."]}}) + "\n",
        encoding="utf-8",
    )
    db = tmp_path / "review.sqlite3"
    review_store.initialize(db)
    review_store.import_units(db, [source_a, source_b])
    first = review_store.list_units(db, limit=1)[0]
    review_store.accept_review(db, first["id"], "Ada", "Un.", "A.", 0)

    assert review_store.review_summary(db) == {"total": 3, "reviewed": 1, "mismatched": 1}
    assert review_store.list_sources(db) == ["alpha", "beta"]
    assert review_store.count_units(db, status="pending") == 2
    assert review_store.count_units(db, status="mismatched") == 1
    assert review_store.count_units(db, source="beta") == 1
    assert [unit["unit_uid"] for unit in review_store.list_units(db, limit=1, offset=1)] == ["two"]
    assert [unit["unit_uid"] for unit in review_store.list_units(db, status="reviewed")] == ["one"]
    assert list(review_store.iter_pairs(db, reviewed_only=True)) == [
        {"french": "Un.", "moore": "A.", "source": "alpha", "unit": "one"}
    ]


def _one_unit_db(tmp_path, fra, mos):
    source = tmp_path / "sample_units.jsonl"
    source.write_text(json.dumps({"u1": {"fra": fra, "mos": mos}}) + "\n", encoding="utf-8")
    db = tmp_path / "review.sqlite3"
    review_store.initialize(db)
    review_store.import_units(db, [source])
    return db, review_store.list_units(db)[0]["id"]


def test_rejected_lines_are_left_out_of_counts_and_export(tmp_path):
    db, unit_id = _one_unit_db(tmp_path, ["Titre", "Un.", "Deux."], ["A.", "Hors sujet.", "B."])
    fra, mos = "Titre\nUn.\nDeux.", "A.\nHors sujet.\nB."

    with pytest.raises(ValueError, match=r"\(2 / 3\)"):
        review_store.accept_review(db, unit_id, "Ada", fra, mos, 0, rejected_fra=[0])
    review_store.accept_review(db, unit_id, "Ada", fra, mos, 0, rejected_fra=[0], rejected_mos=[1])

    unit = review_store.get_unit(db, unit_id)
    assert unit["fra"] == ["Titre", "Un.", "Deux."]
    assert (unit["rejected_fra"], unit["rejected_mos"]) == ([0], [1])
    assert [(p["french"], p["moore"]) for p in review_store.iter_pairs(db)] == [
        ("Un.", "A."),
        ("Deux.", "B."),
    ]


def test_rejecting_every_line_cannot_be_accepted(tmp_path):
    db, unit_id = _one_unit_db(tmp_path, ["Un."], ["A."])
    with pytest.raises(ValueError, match=r"\(0 / 0\)"):
        review_store.accept_review(db, unit_id, "Ada", "Un.", "A.", 0, rejected_fra=[0], rejected_mos=[0])


def test_blank_lines_do_not_shift_rejected_indices():
    # Index 3 is "Hors sujet." in the editor; after dropping the blank line it is 2.
    assert review_store.clean_lines("Un.\n\nDeux.\nHors sujet.\n", [3]) == (
        ["Un.", "Deux.", "Hors sujet."],
        [2],
    )
    assert review_store.clean_lines("Un.\r\nDeux.", [1]) == (["Un.", "Deux."], [1])


def test_drafts_keep_rejected_indices(tmp_path):
    db, unit_id = _one_unit_db(tmp_path, ["Un."], ["A."])
    review_store.save_draft(db, unit_id, "Ada", "Titre\nUn.", "A.", 0, rejected_fra=[0])
    draft = review_store.get_draft(db, unit_id, "Ada")
    assert (draft["rejected_fra"], draft["rejected_mos"]) == ([0], [])
    review_store.save_draft(db, unit_id, "Ada", "Un.", "A.", 0)
    assert review_store.get_draft(db, unit_id, "Ada")["rejected_fra"] == []


def test_mismatch_filter_counts_kept_lines(tmp_path):
    db, unit_id = _one_unit_db(tmp_path, ["Titre", "Un."], ["A."])
    assert review_store.count_units(db, status="mismatched") == 1
    # Reviewer saves the rejection in a draft but hasn't accepted: still pending and mismatched.
    review_store.save_draft(db, unit_id, "Ada", "Titre\nUn.", "A.", 0, rejected_fra=[0])
    assert review_store.review_summary(db)["mismatched"] == 1
    review_store.accept_review(db, unit_id, "Ada", "Titre\nUn.", "A.", 0, rejected_fra=[0])
    assert review_store.review_summary(db) == {"total": 1, "reviewed": 1, "mismatched": 0}


def test_initialize_adds_rejection_columns_to_an_old_database(tmp_path):
    import sqlite3

    db = tmp_path / "old.sqlite3"
    with sqlite3.connect(db) as conn:
        conn.executescript(
            """
            CREATE TABLE units (id INTEGER PRIMARY KEY, source TEXT NOT NULL, unit_uid TEXT NOT NULL,
                position INTEGER NOT NULL, original_fra TEXT NOT NULL, original_mos TEXT NOT NULL,
                UNIQUE (source, unit_uid));
            CREATE TABLE reviews (unit_id INTEGER PRIMARY KEY REFERENCES units(id), fra TEXT NOT NULL,
                mos TEXT NOT NULL, version INTEGER NOT NULL DEFAULT 0, reviewed INTEGER NOT NULL DEFAULT 0,
                reviewed_by TEXT, updated_at TEXT);
            CREATE TABLE drafts (unit_id INTEGER NOT NULL REFERENCES units(id), reviewer TEXT NOT NULL,
                fra_text TEXT NOT NULL, mos_text TEXT NOT NULL, base_version INTEGER NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY (unit_id, reviewer));
            INSERT INTO units VALUES (1, 'old', 'u1', 0, '["Un."]', '["A."]');
            INSERT INTO reviews VALUES (1, '["Un."]', '["A."]', 1, 1, 'Ada', NULL);
            INSERT INTO drafts (unit_id, reviewer, fra_text, mos_text, base_version) VALUES (1, 'Ben', 'Un.', 'A.', 1);
            """
        )

    review_store.initialize(db)
    review_store.initialize(db)  # idempotent

    unit = review_store.get_unit(db, 1)
    assert (unit["reviewed_by"], unit["rejected_fra"], unit["rejected_mos"]) == ("Ada", [], [])
    assert review_store.get_draft(db, 1, "Ben")["rejected_mos"] == []
    assert [p["french"] for p in review_store.iter_pairs(db)] == ["Un."]
