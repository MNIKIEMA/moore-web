import json

import pytest

from moore_web.moore_tales_parser import COLLECTION_ID, parse_moore_tales


def _page(blocks: str) -> str:
    return f'<html><body><div id="content"><div class="b"></div>{blocks}</div></body></html>'


MOORE_1 = _page(
    '<div class="m"><div id="Ta" class="txs"><span class="bd">1 Yõens </span><span class="bd">yelle</span></div></div>'
    '<div class="m"><div id="Tb" class="txs">Daar a yembre,</div>'
    '<div id="Tc" class="txs">yõensã tigma taab.</div><div id="Td" class="txs">B yeelame.</div></div>'
)
FRENCH_1 = _page(
    '<div class="m"><span class="bd">1 Problème des Souris</span></div>'
    '<div class="m">Un jour, les souris se réunirent. Elles dirent.</div>'
)
MOORE_2 = _page(
    '<div class="m"><div id="Ta" class="txs"><span class="bd">2 Kɩɩba.</span></div></div>'
    '<div class="m"><div id="Tb" class="txs">Kɩɩb a ye.</div></div>'
)
FRENCH_2 = _page('<div class="m">2. Un orphelin</div><div class="m">Un orphelin vivait.</div>')


def _app(tmp_path, pages: dict[str, str], segments: dict[str, list[str]]):
    (tmp_path / "html").mkdir()
    for name, html in pages.items():
        (tmp_path / "html" / name).write_text(html, encoding="utf-8")
    with (tmp_path / "segments.jsonl").open("w", encoding="utf-8") as f:
        for page, labels in segments.items():
            for label in labels:
                row = {
                    "collection_id": COLLECTION_ID,
                    "page": page,
                    "label": label,
                    "audio_url": f"{page}.mp3",
                }
                f.write(json.dumps(row) + "\n")
    return tmp_path


def _valid_app(tmp_path):
    return _app(
        tmp_path,
        {
            "01-B021-001.html": MOORE_1,
            "02-B022-001.html": FRENCH_1,
            "03-B043-001.html": MOORE_2,
            "04-B044-001.html": FRENCH_2,
        },
        {"01-B021-001.html": ["a", "b", "c", "d"], "03-B043-001.html": ["a", "b"]},
    )


def test_pairs_moore_and_french_pages_by_tale_number(tmp_path):
    tales = parse_moore_tales(_valid_app(tmp_path))

    assert [t.id for t in tales] == [f"{COLLECTION_ID}-01", f"{COLLECTION_ID}-02"]
    first = tales[0]
    assert (first.source_title, first.target_title) == ("Yõens yelle", "Problème des Souris")
    # Adjacent audio segments stay separate words; the title opens both sentence lists.
    assert first.source_text == "Daar a yembre, yõensã tigma taab. B yeelame."
    assert first.source_sentences[0] == "Yõens yelle"
    assert first.target_sentences == [
        "Problème des Souris",
        "Un jour, les souris se réunirent.",
        "Elles dirent.",
    ]
    assert first.audio_url == "01-B021-001.html.mp3"
    assert tales[1].target_title == "Un orphelin"


def test_title_split_across_spans_keeps_its_number(tmp_path):
    # As on the real page of tale 21: '2' + '1 Kɩɩba'.
    split = _page(
        '<div class="m"><div id="Ta" class="txs"><span class="bd">2</span><span class="bd">1 Kɩɩba</span></div></div>'
        '<div class="m"><div id="Tb" class="txs">Daar a yembre.</div></div>'
    )
    french = _page('<div class="m">21 Un orphelin</div><div class="m">Il vivait.</div>')
    app = _app(tmp_path, {"01-a.html": split, "02-b.html": french}, {"01-a.html": ["a", "b"]})
    with pytest.raises(ValueError, match=r"not 1\.\.1 in page order: \[21\]"):
        parse_moore_tales(app)  # read as 21, not '2 1 Kɩɩba'; only the numbering check fails


def test_mismatched_tale_numbers_fail(tmp_path):
    app = _app(tmp_path, {"01-a.html": MOORE_1, "02-b.html": FRENCH_2}, {"01-a.html": ["a", "b", "c", "d"]})
    with pytest.raises(ValueError, match="is tale 1 but"):
        parse_moore_tales(app)


def test_audio_segments_on_the_french_page_fail(tmp_path):
    app = _app(
        tmp_path,
        {"01-a.html": MOORE_1, "02-b.html": FRENCH_1},
        {"01-a.html": ["a", "b", "c", "d"], "02-b.html": ["a"]},
    )
    with pytest.raises(ValueError, match="Mooré page only"):
        parse_moore_tales(app)
