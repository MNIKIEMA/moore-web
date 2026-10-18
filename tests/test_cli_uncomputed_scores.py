import json
from pathlib import Path
from unittest.mock import MagicMock

import pymupdf
import pytest
from typer.testing import CliRunner

from moore_web import flatten, glossary_parser, one_column_dict_parser
from moore_web.cli import app


@pytest.mark.parametrize("source", ["simple", "one-column-dict"])
def test_dictionary_exports_null_scores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str) -> None:
    input_path = tmp_path / "dictionary.pdf"
    input_path.touch()
    monkeypatch.setattr(pymupdf, "open", MagicMock())
    monkeypatch.setattr(one_column_dict_parser, "parse_doc", lambda doc: [])
    monkeypatch.setattr(
        flatten,
        "flatten_simple_parser",
        lambda *args, **kwargs: flatten.ParallelText(
            french=["chat"], moore=["bagre"], english=["cat"], source="niggli-dictionary-mos-fra-eng"
        ),
    )
    output = tmp_path / "out.jsonl"

    result = CliRunner().invoke(app, ["e2e", "-s", source, "-i", str(input_path), "-o", str(output)])

    assert result.exit_code == 0, result.output
    for pair in ("mos-fra", "mos-eng"):
        rows = [json.loads(line) for line in output.with_suffix(f".{pair}.jsonl").read_text().splitlines()]
        assert len(rows) == 1
        assert rows[0]["laser_score"] is None


@pytest.mark.parametrize(
    ("options", "expected_count"),
    [([], 1), (["--no-terms", "--definitions"], 1), (["--definitions"], 2)],
)
def test_glossary_exports_null_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, options: list[str], expected_count: int
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
    assert all(row["laser_score"] is None for row in rows)
