"""Pair the archived Mooré/French tales of *Contes volume 5* (mooreburkina.com).

The app has 60 HTML pages: each of the 30 tales is a Mooré page followed by
its French translation. Both pages start with the numbered title
(``3 Katre ne wãamba`` / ``3 Le singe et l’hyène``) followed by ``div.m``
paragraphs. Only the Mooré pages are narrated, so only they have entries in
``segments.jsonl``.

The two sides were paragraphed independently (paragraph counts agree for only
3 of 30 tales) and the French is a free translation, so the tale is the only
reliable anchor. Sentence lists inside a tale are for later alignment and are
never paired by list position.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import msgspec
from bs4 import BeautifulSoup

from moore_web.flatten import segment_fr, segment_mo


COLLECTION_ID = "mos-contes-volume-5"
PAGE_URL_BASE = "https://media.ipsapps.org/mos/ora/vol5/"

# '3 Katre ne wãamba', '12. Chat et souris', '27.Ue femme…'
_TITLE_NUMBER_RE = re.compile(r"^(\d+)\s*\.?\s*")


class MooreTale(msgspec.Struct):
    """One tale: its Mooré page and French page, with unaligned sentence lists."""

    id: str
    src_lang: str
    tgt_lang: str
    source_title: str
    target_title: str
    source_text: str
    target_text: str
    source_sentences: list[str]
    target_sentences: list[str]
    source: str
    doc_id: str
    tale_number: int
    source_page: str
    target_page: str
    source_url: str
    target_url: str
    audio_url: str | None
    alignment_method: str


def _clean(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def _read_page(html_path: Path) -> tuple[int, str, list[str], list[str]]:
    """Return (tale number, title, paragraphs, segment labels) of one app page."""
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "html.parser")
    content = soup.select_one("#content")
    if content is None:
        raise ValueError(f"{html_path.name}: missing #content")
    blocks = [b for b in content.find_all("div", class_="m", recursive=False)]
    # Audio segments (div.txs) are separate phrases, sometimes with no whitespace
    # between them; spans inside one are fragments of it ('2' + '1 Kɩɩba').
    texts = [_clean(" ".join(d.get_text() for d in b.select("div.txs")) or b.get_text()) for b in blocks]
    labels = [d["id"][1:] for b in blocks for d in b.select("div.txs[id]")]
    texts = [t for t in texts if t]
    if len(texts) < 2:
        raise ValueError(f"{html_path.name}: expected a title and at least one paragraph")

    title, paragraphs = texts[0], texts[1:]
    match = _TITLE_NUMBER_RE.match(title)
    if not match:
        raise ValueError(f"{html_path.name}: title {title!r} has no tale number")
    return int(match.group(1)), title[match.end() :].strip(), paragraphs, labels


def _load_segments(app_path: Path) -> dict[str, dict[str, dict]]:
    by_page: dict[str, dict[str, dict]] = {}
    with (app_path / "segments.jsonl").open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            segment = json.loads(line)
            if segment.get("collection_id") != COLLECTION_ID:
                raise ValueError(f"Segment line {line_number}: unexpected collection")
            by_page.setdefault(segment["page"], {})[segment["label"]] = segment
    return by_page


def _sentences(title: str, paragraphs: list[str], segment) -> list[str]:
    return [title, *(s for p in paragraphs for s in segment(p))]


def parse_moore_tales(app_dir: str | Path) -> list[MooreTale]:
    """Return one Mooré→French record per tale, checked for pairing and coverage."""
    app_path = Path(app_dir)
    segments = _load_segments(app_path)
    pages = sorted((app_path / "html").glob("*.html"))
    if not pages or len(pages) % 2:
        raise ValueError(f"Expected Mooré/French page pairs, found {len(pages)} pages")

    tales: list[MooreTale] = []
    for mo_path, fr_path in zip(pages[::2], pages[1::2]):
        mo_number, mo_title, mo_paras, mo_labels = _read_page(mo_path)
        fr_number, fr_title, fr_paras, fr_labels = _read_page(fr_path)
        if mo_number != fr_number:
            raise ValueError(f"{mo_path.name} is tale {mo_number} but {fr_path.name} is tale {fr_number}")
        # Only Mooré pages are narrated: this is what tells the two sides apart.
        mo_segments = segments.get(mo_path.name)
        if not mo_segments or fr_path.name in segments:
            raise ValueError(f"{mo_path.name}/{fr_path.name}: expected audio segments on the Mooré page only")
        if set(mo_labels) != set(mo_segments):
            raise ValueError(f"{mo_path.name}: HTML segment labels differ from segments.jsonl")
        audio_urls = {s.get("audio_url") for s in mo_segments.values() if s.get("audio_url")}
        if len(audio_urls) > 1:
            raise ValueError(f"{mo_path.name}: multiple audio URLs")

        tales.append(
            MooreTale(
                id=f"{COLLECTION_ID}-{mo_number:02d}",
                src_lang="mos",
                tgt_lang="fra",
                source_title=mo_title,
                target_title=fr_title,
                source_text="\n".join(mo_paras),
                target_text="\n".join(fr_paras),
                source_sentences=_sentences(mo_title, mo_paras, segment_mo),
                target_sentences=_sentences(fr_title, fr_paras, segment_fr),
                source=COLLECTION_ID,
                doc_id=COLLECTION_ID,
                tale_number=mo_number,
                source_page=mo_path.name,
                target_page=fr_path.name,
                source_url=PAGE_URL_BASE + mo_path.name,
                target_url=PAGE_URL_BASE + fr_path.name,
                audio_url=next(iter(audio_urls), None),
                alignment_method="tale_number",
            )
        )

    numbers = [t.tale_number for t in tales]
    if numbers != list(range(1, len(tales) + 1)):
        raise ValueError(f"Tale numbers are not 1..{len(tales)} in page order: {numbers}")
    return tales


def write_jsonl(tales: list[MooreTale], output_path: str | Path) -> None:
    """Write one JSONL record per tale."""
    with Path(output_path).open("wb") as output:
        output.writelines(msgspec.json.encode(tale) + b"\n" for tale in tales)
