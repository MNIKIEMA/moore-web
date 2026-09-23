import json
from pathlib import Path

import pytest

from moore_web.new_year_message import prepare_new_year_pair, read_presegmented_text


def _write_collection(root: Path, include_moore: bool = True) -> None:
    (root / "2024").mkdir(parents=True)
    (root / "2024" / "fra.txt").write_text(
        "Titre\n\nUne phrase\nsur deux lignes.\n",
        encoding="utf-8",
    )
    documents = [{"languages": ["fra"], "text_file": "2024/fra.txt"}]
    if include_moore:
        (root / "2024" / "mos.txt").write_text(
            "Kibare\n\nGom-bil\nyĩnga.\n",
            encoding="utf-8",
        )
        documents.append({"languages": ["mos"], "text_file": "2024/mos.txt"})
    (root / "manifest.json").write_text(
        json.dumps({"subject_year": 2024, "documents": documents}),
        encoding="utf-8",
    )


def test_read_presegmented_text_uses_blank_lines_as_boundaries(tmp_path: Path) -> None:
    path = tmp_path / "message.txt"
    path.write_text("A\nline\n\nB\n", encoding="utf-8")
    assert read_presegmented_text(path) == ["A line", "B"]


def test_prepare_new_year_pair_reads_manifest_text_files(tmp_path: Path) -> None:
    _write_collection(tmp_path)
    parallel = prepare_new_year_pair(tmp_path)
    assert parallel.french == ["Titre", "Une phrase sur deux lignes."]
    assert parallel.moore == ["Kibare", "Gom-bil yĩnga."]
    assert parallel.source == "messages-nouvel-an"


def test_prepare_new_year_pair_requires_both_curated_texts(tmp_path: Path) -> None:
    _write_collection(tmp_path, include_moore=False)
    with pytest.raises(ValueError, match="mos"):
        prepare_new_year_pair(tmp_path)
