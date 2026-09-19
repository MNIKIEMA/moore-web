"""Parse the archived Mooré/French Proverbs Volume 1 app into proverb pairs.

Each page has three ordered content groups: Mooré proverb, French rendering,
and a repeated Mooré reading. The HTML identifies group boundaries; the
archived ``segments.jsonl`` supplies clean text, labels, and audio provenance.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import msgspec
from bs4 import BeautifulSoup, Tag


COLLECTION_ID = "mos-proverbes-volume-1"
PAGE_URL_BASE = "https://media.ipsapps.org/mos/ora/p1/"


class MooreProverb(msgspec.Struct):
    """One structurally paired proverb and French rendering."""

    id: str
    src_lang: str
    tgt_lang: str
    source_text: str
    target_text: str
    source: str
    doc_id: str
    proverb_number: int
    page: str
    page_url: str
    audio_url: str | None
    moore_segment_labels: list[str]
    french_segment_labels: list[str]
    replay_segment_labels: list[str]


def _clean(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def _without_number(text: str, number: int) -> str:
    return re.sub(rf"^{number}\s+", "", text, count=1).strip()


def _group_text(group: Tag, segments: dict[str, dict], used: set[str], page: str) -> tuple[str, list[str]]:
    labels: list[str] = []
    texts: list[str] = []
    for div in group.select("div.txs[id]"):
        element_id = div.get("id")
        if not isinstance(element_id, str) or not element_id.startswith("T"):
            raise ValueError(f"{page}: invalid segment ID {element_id!r}")
        label = element_id[1:]
        if label not in segments or label in used:
            raise ValueError(f"{page}: missing or duplicated segment {label!r}")
        used.add(label)
        labels.append(label)
        text = segments[label].get("text")
        if text is None:
            raise ValueError(f"{page}: segment {label!r} has no text")
        if text.strip():
            texts.append(text)
    return _clean(" ".join(texts)), labels


def parse_moore_proverbs(app_dir: str | Path) -> list[MooreProverb]:
    """Return one Mooré→French pair for every archived proverb page."""
    app_path = Path(app_dir)
    segments_by_page: dict[str, dict[str, dict]] = defaultdict(dict)
    with (app_path / "segments.jsonl").open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            segment = json.loads(line)
            if segment.get("collection_id") != COLLECTION_ID:
                raise ValueError(f"Segment line {line_number}: unexpected collection")
            page = segment.get("page")
            label = segment.get("label")
            if not isinstance(page, str) or not isinstance(label, str) or label in segments_by_page[page]:
                raise ValueError(f"Segment line {line_number}: invalid or duplicate page/label")
            segments_by_page[page][label] = segment

    pages = sorted((app_path / "html").glob("*.html"))
    if not pages or {p.name for p in pages} != set(segments_by_page):
        raise ValueError("Archived HTML pages and segment pages differ")

    records: list[MooreProverb] = []
    for html_path in pages:
        page = html_path.name
        soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "html.parser")
        content = soup.select_one("#content")
        if content is None:
            raise ValueError(f"{page}: missing #content")
        groups = content.find_all("div", class_="m", recursive=False)
        if len(groups) != 3:
            raise ValueError(f"{page}: expected Mooré/French/replay groups, found {len(groups)}")

        segments = segments_by_page[page]
        used: set[str] = set()
        moore, moore_labels = _group_text(groups[0], segments, used, page)
        french, french_labels = _group_text(groups[1], segments, used, page)
        replay, replay_labels = _group_text(groups[2], segments, used, page)
        if used != set(segments):
            raise ValueError(f"{page}: ungrouped segment labels {sorted(set(segments) - used)}")

        number_match = re.match(r"^(\d+)\s+", moore)
        if not number_match:
            raise ValueError(f"{page}: Mooré group has no proverb number")
        number = int(number_match.group(1))
        if number != int(html_path.stem.rsplit("-", 1)[-1]):
            raise ValueError(f"{page}: proverb number does not match page name")
        moore = _without_number(moore, number)
        if not moore or not french or moore != _without_number(replay, number):
            raise ValueError(f"{page}: missing text or repeated Mooré reading differs")

        audio_urls = {s.get("audio_url") for s in segments.values() if s.get("audio_url")}
        if len(audio_urls) > 1:
            raise ValueError(f"{page}: multiple audio URLs")

        records.append(
            MooreProverb(
                id=f"{COLLECTION_ID}-{number:03d}",
                src_lang="mos",
                tgt_lang="fra",
                source_text=moore,
                target_text=french,
                source=COLLECTION_ID,
                doc_id=COLLECTION_ID,
                proverb_number=number,
                page=page,
                page_url=PAGE_URL_BASE + page,
                audio_url=next(iter(audio_urls), None),
                moore_segment_labels=moore_labels,
                french_segment_labels=french_labels,
                replay_segment_labels=replay_labels,
            )
        )
    return records


def write_jsonl(records: list[MooreProverb], output_path: str | Path) -> None:
    """Write one JSONL record per proverb."""
    with Path(output_path).open("wb") as output:
        output.writelines(msgspec.json.encode(record) + b"\n" for record in records)
