import importlib.util
import json
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "build_fr_mos_dataset", Path(__file__).resolve().parents[1] / "build_fr_mos_dataset.py"
)
build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build)


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _sources(tmp_path, body):
    path = tmp_path / "sources.toml"
    path.write_text(
        'data_dir = "data"\n[filters]\nlaser_score = ">= 0.5"\ncomet_qe = ">= 0.35"\n' + body,
        encoding="utf-8",
    )
    return build.load_sources(path)


def test_entry_filters_merge_over_defaults(tmp_path):
    config = _sources(
        tmp_path,
        '[[sources]]\ntag = "news"\nfile = "news.jsonl"\nfilters = { laser_score = ">= 0.7" }\n'
        '[[sources]]\ntag = "conseils"\nfile = "conseils.jsonl"\n',
    )
    news, conseils = config["sources"]
    assert news["filters"] == {"laser_score": (">=", 0.7), "comet_qe": (">=", 0.35)}
    assert conseils["filters"]["laser_score"] == (">=", 0.5)
    assert not build._passes_filter({"laser_score": 0.6}, news["filters"])
    assert build._passes_filter({"laser_score": 0.6}, conseils["filters"])
    # Rows without scores (human-reviewed) are never filtered.
    assert build._passes_filter({"laser_score": None}, news["filters"])


def test_source_needs_exactly_one_input(tmp_path):
    with pytest.raises(ValueError):
        _sources(tmp_path, '[[sources]]\ntag = "x"\nfile = "a.jsonl"\nreviewed = "a.jsonl"\n')


def test_load_local_orders_filters_skips_and_dedups(tmp_path):
    reviewed_dir = tmp_path / "reviewed"
    _write_jsonl(
        reviewed_dir / "raamde.jsonl", [{"french": "Un.", "moore": "A.", "source": "raamde", "unit": "u"}]
    )
    _write_jsonl(
        tmp_path / "data" / "news.jsonl",
        [
            {"french": "Un.", "moore": "A.", "laser_score": 0.9},  # duplicate of the reviewed pair
            {"french": "Deux.", "moore": "B.", "laser_score": 0.6},  # below the news threshold
            {"french": "Trois.", "moore": "C.", "laser_score": 0.8},
        ],
    )
    _write_jsonl(
        tmp_path / "data" / "expert.jsonl",
        [
            {"source_text": "Quatre.", "target_text": "D.", "qa_check": "ok"},
            {"source_text": "Cinq.", "target_text": "E.", "qa_check": "flag"},
        ],
    )
    config = _sources(
        tmp_path,
        '[[sources]]\ntag = "news"\nreviewed = "raamde.jsonl"\n'
        '[[sources]]\ntag = "news"\nfile = "news.jsonl"\nfilters = { laser_score = ">= 0.7" }\n'
        '[[sources]]\ntag = "expert"\nfile = "expert.jsonl"\nskip = { qa_check = "flag" }\n',
    )
    rows = build.load_local(config, tmp_path / "data", reviewed_dir)
    assert [(r["french"], r["source"], r["laser_score"]) for r in rows] == [
        ("Un.", "news", None),
        ("Trois.", "news", 0.8),
        ("Quatre.", "expert", None),
    ]

    # Without a pinned export, reviewed entries are skipped, not fatal.
    assert [r["french"] for r in build.load_local(config, tmp_path / "data", None)] == [
        "Un.",
        "Trois.",
        "Quatre.",
    ]


def test_text_pair_follows_language_codes():
    assert build._text_pair({"french": " Un. ", "moore": "A."}) == ("Un.", "A.")
    fra_first = {"source_text": "Un.", "target_text": "A.", "src_lang": "fra", "tgt_lang": "mos"}
    assert build._text_pair(fra_first) == ("Un.", "A.")
    mos_first = {"source_text": "A.", "target_text": "Un.", "src_lang": "mos", "tgt_lang": "fra"}
    assert build._text_pair(mos_first) == ("Un.", "A.")
    mos_eng = {"source_text": "bagre", "target_text": "cat", "src_lang": "mos", "tgt_lang": "eng"}
    assert build._text_pair(mos_eng) == ("", "")


def test_row_metadata(tmp_path):
    path = tmp_path / "rows.jsonl"
    _write_jsonl(
        path,
        [
            # long e2e row: upstream id, doc_id and a recorded direction
            {"id": "conseils-000001-000", "source_text": "Un.", "target_text": "A.", "src_lang": "fra",
             "tgt_lang": "mos", "is_source_orig": True, "doc_id": "2024-02-21"},
            # review export row: id from unit + line, doc_id from unit
            {"french": "Deux.", "moore": "B.", "source": "sida", "unit": "page-3", "line": 4},
            # flat row with nothing: hashed id, no document, direction from the argument
            {"french": "Trois.", "moore": "C."},
        ],
    )  # fmt: skip
    rows = build._load_jsonl(path, source_override="x", original_lang="mos", reviewed=True)
    assert [(r["id"], r["doc_id"], r["original_lang"], r["reviewed"]) for r in rows] == [
        ("conseils-000001-000", "2024-02-21", "fra", True),
        ("sida-page-3-4", "page-3", "mos", True),
        (rows[2]["id"], None, "mos", True),
    ]
    assert rows[2]["id"].startswith("x-")
    # Stable across loads.
    assert build._load_jsonl(path, source_override="x")[2]["id"] == rows[2]["id"]


def test_reviewed_flag_from_entry(tmp_path):
    reviewed_dir = tmp_path / "reviewed"
    _write_jsonl(reviewed_dir / "sida.jsonl", [{"french": "Un.", "moore": "A.", "unit": "u", "line": 0}])
    _write_jsonl(tmp_path / "data" / "expert.jsonl", [{"source_text": "Deux.", "target_text": "B."}])
    _write_jsonl(tmp_path / "data" / "auto.jsonl", [{"french": "Trois.", "moore": "C."}])
    config = _sources(
        tmp_path,
        '[[sources]]\ntag = "sida"\nreviewed = "sida.jsonl"\n'
        '[[sources]]\ntag = "expert"\nfile = "expert.jsonl"\nhuman = true\n'
        '[[sources]]\ntag = "auto"\nfile = "auto.jsonl"\n',
    )
    rows = build.load_local(config, tmp_path / "data", reviewed_dir)
    assert [(r["source"], r["reviewed"]) for r in rows] == [("sida", True), ("expert", True), ("auto", False)]
