import json

from moore_web import review_store
from moore_web.reviewed_export import export_reviewed


def _db(tmp_path):
    units = {
        "sida": {"page-1": {"fra": ["Titre.", "Un."], "mos": ["Zu-gʋlsgo.", "A."]}},
        "kade": {"enum-1": {"fra": ["Deux."], "mos": ["B."]}},
        "udhr": {"article-1": {"fra": ["Trois."], "mos": ["C."]}},
    }
    paths = []
    for source, unit in units.items():
        path = tmp_path / f"{source}_units.jsonl"
        path.write_text(json.dumps(unit) + "\n", encoding="utf-8")
        paths.append(path)
    db = tmp_path / "review.sqlite3"
    review_store.initialize(db)
    review_store.import_units(db, paths)
    ids = {u["source"]: u["id"] for u in review_store.list_units(db)}
    review_store.accept_review(
        db, ids["sida"], "Ada", "Titre.\nUn.", "Zu-gʋlsgo.\nA.", 0, rejected_fra=[0], rejected_mos=[0]
    )
    review_store.accept_review(db, ids["kade"], "Ada", "Deux.", "B.", 0)
    return db  # udhr stays unreviewed


def test_export_writes_accepted_units_per_source(tmp_path):
    db = _db(tmp_path)
    out = tmp_path / "export"
    (out / "old-source.jsonl").parent.mkdir()
    (out / "old-source.jsonl").write_text("{}\n", encoding="utf-8")

    summary = export_reviewed(db, out)

    assert sorted(p.name for p in out.iterdir()) == ["_export.json", "kade.jsonl", "sida.jsonl"]
    sida = [json.loads(line) for line in (out / "sida.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(r["french"], r["moore"], r["unit"], r["line"], r["reviewed_by"]) for r in sida] == [
        ("Un.", "A.", "page-1", 0, "Ada")
    ]
    assert summary["sources"]["sida"] == {"rows": 1, "units": 1, "skipped_units": 0}
    assert json.loads((out / "_export.json").read_text(encoding="utf-8")) == summary


def test_export_is_deterministic(tmp_path):
    db = _db(tmp_path)
    export_reviewed(db, tmp_path / "a")
    export_reviewed(db, tmp_path / "b")
    for name in ("_export.json", "kade.jsonl", "sida.jsonl"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
