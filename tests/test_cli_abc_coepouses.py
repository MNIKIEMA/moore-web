import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from moore_web import abc_coepouses_parser
from moore_web.cli import app


def _unit(index: int, fr: str, mo: str) -> abc_coepouses_parser.TaleUnit:
    return abc_coepouses_parser.TaleUnit(
        id=f"abcburkina-contes-les-coepouses-{index:02d}",
        src_lang="fra",
        tgt_lang="mos",
        source_text=fr,
        target_text=mo,
        source_sentences=[fr],
        target_sentences=[mo],
        source="abcburkina-contes",
        doc_id="les-coepouses",
        unit_index=index,
        unit_type="narrative",
        alignment_method="manual_story_anchors",
        source_url="",
        target_url="",
        source_start=0,
        source_end=len(fr),
        target_start=0,
        target_end=len(mo),
    )


def test_e2e_no_segment_keeps_story_beats_as_pairs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fr_input, mo_input = tmp_path / "fr.txt", tmp_path / "mo.txt"
    fr_input.write_text("fr", encoding="utf-8")
    mo_input.write_text("mo", encoding="utf-8")
    units = [
        _unit(1, "Les coépouses", "Pʋg-taab a yiib solemde"),
        _unit(2, "Il était une fois.", "Rao a ye."),
    ]
    monkeypatch.setattr(abc_coepouses_parser, "parse_abc_coepouses", lambda fr, mo: units)
    output = tmp_path / "out.jsonl"

    result = CliRunner().invoke(
        app,
        ["e2e", "-s", "abc-coepouses", "--fr-input", str(fr_input), "--mo-input", str(mo_input)]
        + ["-o", str(output), "--no-segment"],
    )

    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [(r["source_text"], r["target_text"]) for r in rows] == [
        ("Les coépouses", "Pʋg-taab a yiib solemde"),
        ("Il était une fois.", "Rao a ye."),
    ]
    assert {r["source"] for r in rows} == {"abcburkina-contes"}
    assert [r["doc_id"] for r in rows] == [u.id for u in units]
    assert all(r["laser_score"] is None for r in rows)


def test_e2e_requires_both_inputs(tmp_path: Path) -> None:
    fr_input = tmp_path / "fr.txt"
    fr_input.write_text("fr", encoding="utf-8")
    result = CliRunner().invoke(app, ["e2e", "-s", "abc-coepouses", "--fr-input", str(fr_input)])
    assert result.exit_code == 1
    assert "--mo-input" in result.output
