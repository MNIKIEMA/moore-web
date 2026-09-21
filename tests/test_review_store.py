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
