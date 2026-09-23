"""Pair the French and Mooré Universal Declaration of Human Rights by structure.

Both texts (e.g. ``udhr-fra.txt`` / ``udhr-mos.txt`` in ``faso-web-docs``) hold
one paragraph per blank-line-delimited block, headed by the preamble title and
one heading per article. Articles are matched by number and paragraphs by position
within an article, so no embedding-based alignment is needed.
"""

from __future__ import annotations

import re
from pathlib import Path

from moore_web.flatten import AlignedCorpus
from moore_web.new_year_message import read_presegmented_text

SOURCE = "udhr"

# Mooré numbers articles "Koɛɛg a 9 soaba." up to 9, then "Koɛɛg 10 soaba.".
_HEADINGS = {
    "fra": [
        (re.compile(r"^Préambule$"), lambda m: "preamble"),
        # The French preamble ends with the General Assembly's proclamation,
        # which the Mooré translation does not include.
        (re.compile(r"^L’Assemblée générale$"), lambda m: "proclamation"),
        (re.compile(r"^Article premier$"), lambda m: "article-01"),
        (re.compile(r"^Article (\d+)$"), lambda m: f"article-{int(m.group(1)):02d}"),
    ],
    "mos": [
        (re.compile(r"^Keoogre$"), lambda m: "preamble"),
        (re.compile(r"^Pipi koɛɛga\.?$"), lambda m: "article-01"),
        (re.compile(r"^Koɛɛg (?:a )?(\d+) soaba\.?$"), lambda m: f"article-{int(m.group(1)):02d}"),
    ],
}

# Upstream (wooorm/udhr) placeholder standing in for Mooré article 12, which
# the OHCHR Mooré PDF omits. It is not Mooré content.
_PLACEHOLDER_RE = re.compile(r"^&\d+$")
_LIST_NUMBER_RE = re.compile(r"^(\d+)\.\s+")


def split_sections(paragraphs: list[str], lang: str) -> dict[str, list[str]]:
    """Group paragraphs under their heading key; text before any heading is the title."""
    sections: dict[str, list[str]] = {}
    key = "title"
    for paragraph in paragraphs:
        for pattern, make_key in _HEADINGS[lang]:
            match = pattern.match(paragraph)
            if match:
                key = make_key(match)
                if key in sections:
                    raise ValueError(f"{lang}: duplicate section {key!r}")
                sections[key] = []
                break
        else:
            if not _PLACEHOLDER_RE.match(paragraph):
                sections.setdefault(key, []).append(paragraph)
    return {key: body for key, body in sections.items() if body}


def _strip_list_number(paragraph: str) -> tuple[str | None, str]:
    match = _LIST_NUMBER_RE.match(paragraph)
    if match is None:
        return None, paragraph
    return match.group(1), paragraph[match.end() :]


def pair_sections(
    fra: dict[str, list[str]],
    mos: dict[str, list[str]],
    segment: bool = False,
) -> tuple[AlignedCorpus, list[str]]:
    """Pair paragraphs of sections present on both sides with the same paragraph count.

    Returns the aligned corpus and a list of human-readable reasons for every
    section that was skipped. With ``segment``, each paragraph pair is split
    into sentences and paired sentence by sentence when both sides split into
    the same number; otherwise the paragraph pair is kept whole.
    """
    from moore_web.flatten import segment_fr, segment_mo

    french: list[str] = []
    moore: list[str] = []
    doc_ids: list[str] = []
    skipped: list[str] = []

    for key in [*fra, *(k for k in mos if k not in fra)]:
        fr_paragraphs, mo_paragraphs = fra.get(key), mos.get(key)
        if fr_paragraphs is None or mo_paragraphs is None:
            missing = "Mooré" if mo_paragraphs is None else "French"
            skipped.append(f"{key}: no {missing} text")
            continue
        if len(fr_paragraphs) != len(mo_paragraphs):
            skipped.append(f"{key}: {len(fr_paragraphs)} French vs {len(mo_paragraphs)} Mooré paragraphs")
            continue
        for fr_paragraph, mo_paragraph in zip(fr_paragraphs, mo_paragraphs):
            fr_number, fr_text = _strip_list_number(fr_paragraph)
            mo_number, mo_text = _strip_list_number(mo_paragraph)
            if fr_number != mo_number:
                raise ValueError(f"{key}: list item {fr_number} (French) paired with {mo_number} (Mooré)")
            fr_parts, mo_parts = [fr_text], [mo_text]
            if segment:
                fr_sentences, mo_sentences = segment_fr(fr_text), segment_mo(mo_text)
                if len(fr_sentences) == len(mo_sentences):
                    fr_parts, mo_parts = fr_sentences, mo_sentences
            french.extend(fr_parts)
            moore.extend(mo_parts)
            doc_ids.extend([key] * len(fr_parts))

    aligned = AlignedCorpus(
        french=french,
        moore=moore,
        scores=[None] * len(french),
        doc_ids=doc_ids,
        source=SOURCE,
    )
    return aligned, skipped


def pair_udhr_files(fr_path: Path, mo_path: Path, segment: bool = False) -> tuple[AlignedCorpus, list[str]]:
    """Pair a French and a Mooré UDHR text file (blank-line-delimited paragraphs)."""
    fra = split_sections(read_presegmented_text(fr_path), "fra")
    mos = split_sections(read_presegmented_text(mo_path), "mos")
    return pair_sections(fra, mos, segment=segment)
