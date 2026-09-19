"""Segment the abcBurkina French/Mooré editions of *Les coépouses*.

These are two independently typeset translations. Their blank lines are not
parallel, so this parser uses hand-selected story-beat anchors instead of zipping
paragraphs or sentences. It keeps every source span, including songs and
credits, in 25 paired units. Sentence lists inside each unit are for later
alignment and are never truncated to the shorter language.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import msgspec

from moore_web.flatten import segment_fr, segment_mo


SOURCE = "abcburkina-contes"
DOCUMENT_ID = "les-coepouses"

# (unit type, first French words, first Mooré words). Each anchor must occur
# exactly once and in this order. A source-page change therefore fails loudly.
ANCHORS: tuple[tuple[str, str, str], ...] = (
    ("title", "Les coépouses", "Pʋg-taab a yiib solemde"),
    ("subtitle", "J’ai tué un bœuf", "Mam kʋʋ naafo"),
    ("narrative", "Il était une fois", "Rao a ye n da tar a paga"),
    ("narrative", "Un beau jour", "Daar a ye, a raame"),
    ("narrative", "Chaque jour", "Beoog fãa, wĩndgã"),
    ("narrative", "La deuxième femme était", "Pʋg-yao wã n da yaa"),
    ("narrative", "Elle eut sa première", "Pʋg-yao wã wa n dɩka"),
    ("narrative", "A cause de son accueil", "A sẽn mi ned deegr"),
    ("narrative", "C’était une femme", "A da yaa pag"),
    ("narrative", "C’est ainsi dit que", "Woto la pagb"),
    ("narrative", "Lorsqu’elle sortit", "A sẽn yi wã"),
    ("narrative", "Quand la première femme", "Pʋg-kẽemã sẽn ta"),
    ("narrative", "Dans sa fureur", "Sɩdã sũurã"),
    ("narrative", "Les jours passèrent", "Rasmã ne yʋndã"),
    ("narrative", "Désespérés", "Nebã lebg n dɩka"),
    ("narrative", "Il entendit la femme", "A wʋma pagã"),
    ("song", "« Bonjour", "« Ne f gãag"),
    ("narrative", "Après avoir chanté", "A sẽn yɩɩl woto"),
    ("narrative", "Elle commença à chanter", "A sɩnga yɩɩlg"),
    ("song", "« Arbre", "« Tɩɩga"),
    ("narrative", "Et la femme du sein", "La pagã paa"),
    ("song", "« Ayez pitié", "« Zoe-y m nimbãanega"),
    ("narrative", "Et soudain", "Zĩig pʋgẽ, tɩɩgã pakame"),
    ("moral", "Voici pour quoi", "Yaa rẽ yĩng"),
    ("attribution", "Conte Samo de Banso", "Samogemb solemde"),
)


class TaleUnit(msgspec.Struct):
    """A structurally paired story span and its unaligned sentence lists."""

    id: str
    src_lang: str
    tgt_lang: str
    source_text: str
    target_text: str
    source_sentences: list[str]
    target_sentences: list[str]
    source: str
    doc_id: str
    unit_index: int
    unit_type: str
    alignment_method: str
    source_url: str
    target_url: str
    source_start: int
    source_end: int
    target_start: int
    target_end: int


def _read_archive_text(path: str | Path, expected_lang: str) -> tuple[str, str]:
    """Return URL and whitespace-normalized article body after archive metadata."""
    raw = Path(path).read_text(encoding="utf-8-sig")
    header, sep, body = raw.partition("\n\n")
    if not sep:
        raise ValueError(f"{path}: missing archive header separator")
    fields = dict(line.split(": ", 1) for line in header.splitlines() if ": " in line)
    if fields.get("Languages") != expected_lang or not fields.get("Source"):
        raise ValueError(f"{path}: expected language {expected_lang} and a source URL")
    normalized = " ".join(unicodedata.normalize("NFC", body).split())
    if not normalized:
        raise ValueError(f"{path}: empty article body")
    return fields["Source"], normalized


def split_at_anchors(body: str, anchors: list[str]) -> list[tuple[str, int, int]]:
    """Partition a normalized article without dropping or reordering text."""
    positions: list[int] = []
    for anchor in anchors:
        if body.count(anchor) != 1:
            raise ValueError(f"Expected one occurrence of anchor {anchor!r}, found {body.count(anchor)}")
        position = body.index(anchor)
        if positions and position <= positions[-1]:
            raise ValueError(f"Out-of-order anchor {anchor!r}")
        positions.append(position)
    if not positions or positions[0] != 0:
        raise ValueError("First anchor must start the article body")

    spans: list[tuple[str, int, int]] = []
    for index, start in enumerate(positions):
        end = positions[index + 1] if index + 1 < len(positions) else len(body)
        text = body[start:end].strip()
        if not text:
            raise ValueError(f"Empty span at anchor {anchors[index]!r}")
        spans.append((text, start, end))
    return spans


def parse_abc_coepouses(french_path: str | Path, moore_path: str | Path) -> list[TaleUnit]:
    """Return reviewed French/Mooré story-beat units from the two text files."""
    french_url, french = _read_archive_text(french_path, "fra")
    moore_url, moore = _read_archive_text(moore_path, "mos")
    french_spans = split_at_anchors(french, [a[1] for a in ANCHORS])
    moore_spans = split_at_anchors(moore, [a[2] for a in ANCHORS])

    units: list[TaleUnit] = []
    for index, ((unit_type, _, _), fr_span, mo_span) in enumerate(
        zip(ANCHORS, french_spans, moore_spans, strict=True), start=1
    ):
        fr_text, fr_start, fr_end = fr_span
        mo_text, mo_start, mo_end = mo_span
        units.append(
            TaleUnit(
                id=f"{SOURCE}-{DOCUMENT_ID}-{index:02d}",
                src_lang="fra",
                tgt_lang="mos",
                source_text=fr_text,
                target_text=mo_text,
                source_sentences=segment_fr(fr_text),
                target_sentences=segment_mo(mo_text),
                source=SOURCE,
                doc_id=DOCUMENT_ID,
                unit_index=index,
                unit_type=unit_type,
                alignment_method="manual_story_anchors",
                source_url=french_url,
                target_url=moore_url,
                source_start=fr_start,
                source_end=fr_end,
                target_start=mo_start,
                target_end=mo_end,
            )
        )
    return units


def write_jsonl(units: list[TaleUnit], output_path: str | Path) -> None:
    """Write one JSONL record per paired story span."""
    with Path(output_path).open("wb") as output:
        output.writelines(msgspec.json.encode(unit) + b"\n" for unit in units)
