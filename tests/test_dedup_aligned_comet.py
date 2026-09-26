import json

from moore_web import cli
from moore_web.dedup_aligned_comet import deduplicate_by_score, duplicate_groups
from moore_web.flatten import AlignedCorpus


def test_duplicate_groups_are_transitive_and_skip_singletons():
    pairs = [
        {"fr": "A", "mo": "x"},
        {"fr": "A", "mo": "y"},  # shares fr with 0
        {"fr": "B", "mo": "y"},  # shares mo with 1 -> same group as 0
        {"fr": "C", "mo": "z"},  # alone
    ]
    assert duplicate_groups(pairs) == [[0, 1, 2]]


def test_group_key_keeps_language_pairs_apart():
    rows = [
        {"source_text": "bagre", "target_text": "chat", "pair": "mos-fra"},
        {"source_text": "bagre", "target_text": "cat", "pair": "mos-eng"},
    ]
    assert duplicate_groups(rows, "source_text", "target_text") == [[0, 1]]
    assert duplicate_groups(rows, "source_text", "target_text", group_key=lambda r: r["pair"]) == []


def test_deduplicate_by_score_keeps_best_of_each_group():
    pairs = [
        {"fr": "A", "mo": "x", "comet_qe": 0.4},
        {"fr": "A", "mo": "y", "comet_qe": 0.7},
        {"fr": "C", "mo": "z", "comet_qe": 0.1},
    ]
    assert [p["mo"] for p in deduplicate_by_score(pairs)] == ["y", "z"]


def test_finalize_scores_once_then_drops_duplicates(tmp_path, monkeypatch):
    calls = []

    def fake_comet(dataset, **kwargs):
        calls.append(len(dataset))
        scores = [0.9 if t == "Minisr-dãmbã." else 0.5 for t in dataset["target_text"]]
        return dataset.add_column("comet_qe", scores)

    monkeypatch.setattr("moore_web.annotate.run_comet_qe", fake_comet)
    monkeypatch.setattr(
        "moore_web.dedup_aligned_comet.deduplicate_by_comet",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not score duplicates separately")),
    )
    aligned = AlignedCorpus(
        french=["Le Conseil a adopté.", "Le Conseil a adopté.", "Autre."],
        moore=["Tigsgã.", "Minisr-dãmbã.", "Bõn-a-to."],
        scores=[0.8, 0.8, 0.8],
        source="conseils",
    )
    out = tmp_path / "out.jsonl"
    cli._finalize_aligned(
        aligned, out, True, hf_private=False, add_lang_id=False, add_consistency=False,
        add_quality_warn=False, add_len_ratio=False, add_laser_score=False, add_comet_qe=True,
        drop_duplicate_by_comet_qe=True,
    )  # fmt: skip
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert calls == [3]
    assert [r["target_text"] for r in rows] == ["Minisr-dãmbã.", "Bõn-a-to."]
    assert all("comet_qe" in r for r in rows)
