from pathlib import Path

import pytest

from moore_web.udhr import pair_sections, pair_udhr_files, split_sections, udhr_review_units

FRA = """Déclaration universelle des droits de l’homme

Préambule

Considérant la dignité,

L’Assemblée générale

Proclame la présente Déclaration.

Article premier

Tous naissent libres. Ils sont doués de raison.

Article 11

1. Toute personne est présumée innocente.

2. Nul ne sera condamné.

Article 12

Nul ne sera l’objet d’immixtions.
"""

MOS = """Ninsaal yel-segdɩ noy gãnegr sebre

Keoogre

B wilgame tɩ ninsaal,

Pipi koɛɛga.

Ninsaalbã fãa so b mense. Nebã fãa tara yam.

Koɛɛg 11 soaba.

1. B sã n yeel tɩ ned maana bʊmb.

2. B pa tõe n bʊ ned.

Koɛɛg 12 soaba.

&1
"""


def _write_texts(root: Path) -> tuple[Path, Path]:
    fr_path, mo_path = root / "udhr-fra.txt", root / "udhr-mos.txt"
    fr_path.write_text(FRA, encoding="utf-8")
    mo_path.write_text(MOS, encoding="utf-8")
    return fr_path, mo_path


def test_split_sections_keys_articles_and_drops_placeholder() -> None:
    sections = split_sections(["Titre", "Koɛɛg a 2 soaba.", "Ned fãa.", "Koɛɛg 12 soaba.", "&1"], "mos")
    assert sections == {"title": ["Titre"], "article-02": ["Ned fãa."]}


def test_split_sections_rejects_duplicate_heading() -> None:
    with pytest.raises(ValueError, match="article-02"):
        split_sections(["Article 2", "A", "Article 2", "B"], "fra")


def test_pair_udhr_files_skips_sections_missing_on_one_side(tmp_path: Path) -> None:
    aligned, skipped = pair_udhr_files(*_write_texts(tmp_path))
    assert skipped == ["proclamation: no Mooré text", "article-12: no Mooré text"]
    assert aligned.doc_ids == ["title", "preamble", "article-01", "article-11", "article-11"]
    assert aligned.french[3] == "Toute personne est présumée innocente."
    assert aligned.moore[3] == "B sã n yeel tɩ ned maana bʊmb."
    assert aligned.scores == [None] * 5
    assert aligned.source == "udhr"


def test_pair_udhr_files_segments_when_sentence_counts_match(tmp_path: Path) -> None:
    aligned, _ = pair_udhr_files(*_write_texts(tmp_path), segment=True)
    article_1 = [
        (f, m) for f, m, d in zip(aligned.french, aligned.moore, aligned.doc_ids) if d == "article-01"
    ]
    assert article_1 == [
        ("Tous naissent libres.", "Ninsaalbã fãa so b mense."),
        ("Ils sont doués de raison.", "Nebã fãa tara yam."),
    ]


def test_pair_sections_skips_paragraph_count_mismatch() -> None:
    aligned, skipped = pair_sections({"article-02": ["A", "B"]}, {"article-02": ["A"]})
    assert skipped == ["article-02: 2 French vs 1 Mooré paragraphs"]
    assert aligned.french == []


def test_pair_sections_rejects_mismatched_list_numbers() -> None:
    with pytest.raises(ValueError, match="list item 1"):
        pair_sections({"article-11": ["1. A"]}, {"article-11": ["2. B"]})


def test_udhr_review_units_are_per_section_and_always_segmented(tmp_path: Path) -> None:
    units, skipped = udhr_review_units(*_write_texts(tmp_path))
    assert skipped == ["proclamation: no Mooré text", "article-12: no Mooré text"]
    assert [uid for uid, _ in units] == ["title", "preamble", "article-01", "article-11"]
    article_1 = dict(units)["article-01"]
    assert article_1.french == ["Tous naissent libres.", "Ils sont doués de raison."]
    assert article_1.moore == ["Ninsaalbã fãa so b mense.", "Nebã fãa tara yam."]
    assert dict(units)["article-11"].french == [
        "Toute personne est présumée innocente.",
        "Nul ne sera condamné.",
    ]
