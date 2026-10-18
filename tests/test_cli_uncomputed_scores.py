import json
from pathlib import Path
from unittest.mock import MagicMock

import pymupdf
import pytest
from typer.testing import CliRunner

from moore_web import annotate, flatten, glossary_parser, one_column_dict_parser
from moore_web.cli import _finalize_aligned, app


@pytest.fixture
def laser_calls(monkeypatch: pytest.MonkeyPatch) -> list[list[dict]]:
    calls = []

    def score(dataset, **kwargs):
        calls.append(dataset.to_list())
        scores = [0.8 if row["tgt_lang"] == "fra" else 0.7 for row in dataset]
        return dataset.remove_columns("laser_score").add_column("laser_score", scores)

    monkeypatch.setattr(annotate, "run_laser", score)
    return calls


@pytest.mark.parametrize("source", ["simple", "one-column-dict"])
@pytest.mark.parametrize("options", [[], ["--entries"], ["--no-examples", "--entries"]])
def test_dictionary_scores_examples_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str, options: list[str], laser_calls
) -> None:
    input_path = tmp_path / "dictionary.pdf"
    input_path.touch()
    monkeypatch.setattr(pymupdf, "open", MagicMock())
    monkeypatch.setattr(one_column_dict_parser, "parse_doc", lambda doc: [])

    def flatten_dictionary(pages, include_examples, include_entries):
        pairs = []
        if include_entries:
            pairs.append(("chat", "bagre", "cat"))
        if include_examples:
            pairs.append(("Un chat.", "Bagre.", "A cat."))
        return flatten.ParallelText(
            french=[p[0] for p in pairs],
            moore=[p[1] for p in pairs],
            english=[p[2] for p in pairs],
            source="niggli-dictionary-mos-fra-eng",
        )

    monkeypatch.setattr(flatten, "flatten_simple_parser", flatten_dictionary)
    output = tmp_path / "out.jsonl"

    result = CliRunner().invoke(
        app, ["e2e", "-s", source, "-i", str(input_path), "-o", str(output)] + options
    )

    assert result.exit_code == 0, result.output
    for pair in ("mos-fra", "mos-eng"):
        rows = [json.loads(line) for line in output.with_suffix(f".{pair}.jsonl").read_text().splitlines()]
        assert len(rows) == (2 if options == ["--entries"] else 1)
        for row in rows:
            expected = (0.8 if pair == "mos-fra" else 0.7) if row["source_text"] == "Bagre." else None
            assert row["laser_score"] == expected
    assert len(laser_calls) == (0 if "--no-examples" in options else 1)
    assert all(row["source_text"] == "Bagre." for call in laser_calls for row in call)


@pytest.mark.parametrize(
    ("options", "expected_count"),
    [([], 1), (["--no-terms", "--definitions"], 1), (["--definitions"], 2)],
)
def test_glossary_scores_definitions_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, options: list[str], expected_count: int, laser_calls
) -> None:
    fr_input, mo_input = tmp_path / "fr.pdf", tmp_path / "mo.pdf"
    fr_input.touch()
    mo_input.touch()
    monkeypatch.setattr(glossary_parser, "extract_moore_tables", lambda path: [])
    monkeypatch.setattr(glossary_parser, "extract_french_tables", lambda path: [])
    monkeypatch.setattr(
        glossary_parser,
        "align_glossaries",
        lambda *args: [
            glossary_parser.AlignedEntry(
                fr_term="chat", mos_term="bagre", fr_definition="Un chat.", mos_definition="Bagre."
            )
        ],
    )
    output = tmp_path / "out.jsonl"

    result = CliRunner().invoke(
        app,
        ["e2e", "-s", "digital", "--fr-input", str(fr_input), "--mo-input", str(mo_input), "-o", str(output)]
        + options,
    )

    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == expected_count
    for row in rows:
        assert row["laser_score"] == (0.8 if row["source_text"] == "Bagre." else None)
    assert len(laser_calls) == (1 if "--definitions" in options else 0)
    assert all(row["source_text"] == "Bagre." for call in laser_calls for row in call)


@pytest.mark.parametrize("explicit_laser", [False, True])
def test_selective_scoring_after_cleanup_runs_once(tmp_path: Path, laser_calls, explicit_laser: bool) -> None:
    aligned = flatten.AlignedCorpus(
        french=["chat", "Un chat. Proverbe : note"],
        moore=["bagre", "Bagre."],
        english=["cat", "A cat."],
        scores=[None, None],
        source="niggli-dictionary-mos-fra-eng",
    )

    def clean(rows):
        return [{**row, "french": row["french"].split(" Proverbe")[0]} for row in rows]

    output = tmp_path / "out.jsonl"
    _finalize_aligned(
        aligned, output, True, hf_private=False, add_lang_id=False, add_consistency=False,
        add_quality_warn=False, add_len_ratio=False, add_laser_score=explicit_laser, add_comet_qe=False,
        postprocess=clean, laser_score_mask=[False, True],
    )  # fmt: skip

    assert len(laser_calls) == 1
    assert len(laser_calls[0]) == (4 if explicit_laser else 2)
    assert any(row["target_text"] == "Un chat." for row in laser_calls[0])
    assert all("_compute_laser" not in row for row in laser_calls[0])
    rows = [json.loads(line) for line in output.with_suffix(".mos-fra.jsonl").read_text().splitlines()]
    assert [row["laser_score"] for row in rows] == ([0.8, 0.8] if explicit_laser else [None, 0.8])


def test_selective_scoring_keeps_aligned_corpus_json_format(tmp_path: Path, laser_calls) -> None:
    aligned = flatten.AlignedCorpus(
        french=["chat", "Un chat."],
        moore=["bagre", "Bagre."],
        english=["cat", "A cat."],
        scores=[None, None],
        source="niggli-dictionary-mos-fra-eng",
    )
    output = tmp_path / "out.json"

    _finalize_aligned(
        aligned, output, False, hf_private=False, add_lang_id=False, add_consistency=False,
        add_quality_warn=False, add_len_ratio=False, add_laser_score=False, add_comet_qe=False,
        laser_score_mask=[False, True],
    )  # fmt: skip

    assert [row["target_text"] for row in laser_calls[0]] == ["Un chat."]
    written = json.loads(output.read_text())
    assert written["english"] == ["cat", "A cat."]
    assert written["scores"] == [None, 0.8]


def test_no_default_laser_score_skips_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, laser_calls
) -> None:
    input_path = tmp_path / "dictionary.pdf"
    input_path.touch()
    monkeypatch.setattr(pymupdf, "open", MagicMock())
    monkeypatch.setattr(one_column_dict_parser, "parse_doc", lambda doc: [])
    monkeypatch.setattr(
        flatten,
        "flatten_simple_parser",
        lambda pages, include_examples, include_entries: flatten.ParallelText(
            french=["Un chat."], moore=["Bagre."], english=["A cat."], source="niggli-dictionary-mos-fra-eng"
        ),
    )
    output = tmp_path / "out.jsonl"

    result = CliRunner().invoke(
        app, ["e2e", "-s", "simple", "-i", str(input_path), "-o", str(output), "--no-default-laser-score"]
    )

    assert result.exit_code == 0, result.output
    assert laser_calls == []
    for pair in ("mos-fra", "mos-eng"):
        rows = [json.loads(line) for line in output.with_suffix(f".{pair}.jsonl").read_text().splitlines()]
        assert [row["laser_score"] for row in rows] == [None]
