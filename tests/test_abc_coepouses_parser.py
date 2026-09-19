"""Checks for lossless, ordered story-beat segmentation."""

import pytest

from moore_web.abc_coepouses_parser import split_at_anchors


def test_split_at_anchors_preserves_all_text() -> None:
    body = "Title. First episode. Second episode. End."
    spans = split_at_anchors(body, ["Title.", "First episode.", "Second episode."])

    assert [text for text, _, _ in spans] == ["Title.", "First episode.", "Second episode. End."]
    assert spans[0][1] == 0
    assert spans[-1][2] == len(body)
    assert spans[0][2] == spans[1][1]
    assert spans[1][2] == spans[2][1]


def test_missing_anchor_is_rejected() -> None:
    with pytest.raises(ValueError, match="Expected one occurrence"):
        split_at_anchors("Title. End.", ["Title.", "Missing."])


def test_out_of_order_anchor_is_rejected() -> None:
    with pytest.raises(ValueError, match="Out-of-order"):
        split_at_anchors("Title. End.", ["End.", "Title."])
