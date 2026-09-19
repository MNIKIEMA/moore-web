"""Tests for pairing the three groups of a Mooré proverb app page."""

import json

import pytest

from moore_web.moore_proverbs_parser import parse_moore_proverbs


def _make_app(tmp_path, *, replay: str = "Baag ye.", include_french: bool = True):
    app = tmp_path / "app"
    html = app / "html"
    html.mkdir(parents=True)
    french_group = '<div class="m"><div class="txs" id="T1b">French</div></div>' if include_french else ""
    (html / "01-001-001.html").write_text(
        '<div id="content">'
        '<div class="m"><div class="txs" id="T1a">Mooré</div></div>'
        + french_group
        + '<div class="m"><div class="txs" id="T1c">Replay</div></div>'
        + "</div>"
    )
    rows = [
        {
            "collection_id": "mos-proverbes-volume-1",
            "page": "01-001-001.html",
            "label": "1a",
            "text": "1 Baag ye.",
            "audio_url": "https://example.test/audio.mp3",
        },
        {
            "collection_id": "mos-proverbes-volume-1",
            "page": "01-001-001.html",
            "label": "1b",
            "text": "(Un chien).",
            "audio_url": "https://example.test/audio.mp3",
        },
        {
            "collection_id": "mos-proverbes-volume-1",
            "page": "01-001-001.html",
            "label": "1c",
            "text": replay,
            "audio_url": "https://example.test/audio.mp3",
        },
    ]
    (app / "segments.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return app


def test_pairs_one_proverb_without_repeated_reading(tmp_path) -> None:
    record = parse_moore_proverbs(_make_app(tmp_path))[0]

    assert (record.source_text, record.target_text) == ("Baag ye.", "(Un chien).")
    assert (record.src_lang, record.tgt_lang) == ("mos", "fra")
    assert record.proverb_number == 1
    assert record.moore_segment_labels == ["1a"]
    assert record.french_segment_labels == ["1b"]
    assert record.replay_segment_labels == ["1c"]
    assert record.audio_url == "https://example.test/audio.mp3"


def test_replay_mismatch_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="repeated Mooré reading differs"):
        parse_moore_proverbs(_make_app(tmp_path, replay="Different proverb."))


def test_missing_french_group_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="expected Mooré/French/replay groups"):
        parse_moore_proverbs(_make_app(tmp_path, include_french=False))
