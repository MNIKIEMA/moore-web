"""Prepare curated French–Mooré New Year messages for alignment."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from moore_web.flatten import ParallelText


def read_presegmented_text(path: Path) -> list[str]:
    """Read blank-line-delimited UTF-8 segments, preserving Unicode text."""
    text = unicodedata.normalize("NFC", path.read_text(encoding="utf-8")).replace("\f", "")
    segments = []
    for block in re.split(r"\n\s*\n", text):
        normalized = re.sub(r"\s+", " ", block).strip()
        if normalized:
            segments.append(normalized)
    return segments


def prepare_new_year_pair(collection_dir: Path) -> ParallelText:
    """Load the curated French and Mooré text files named by the manifest."""
    manifest = json.loads((collection_dir / "manifest.json").read_text(encoding="utf-8"))
    text_files: dict[str, Path] = {}
    for document in manifest.get("documents", []):
        languages = document.get("languages") or []
        text_file = document.get("text_file")
        if len(languages) == 1 and languages[0] in {"fra", "mos"} and text_file:
            text_files[languages[0]] = collection_dir / text_file

    missing = {"fra", "mos"} - text_files.keys()
    if missing:
        raise ValueError(f"manifest is missing curated text for: {', '.join(sorted(missing))}")
    for path in text_files.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    return ParallelText(
        french=read_presegmented_text(text_files["fra"]),
        moore=read_presegmented_text(text_files["mos"]),
        source="messages-nouvel-an",
    )
