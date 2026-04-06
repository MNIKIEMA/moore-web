"""Tests for book_parser_facilitateur: NUMBERED_ITEM_RE, collect_numbered_items,
and start_after / stop_before sentinels in split_and_parse_by_sections."""

import re

from moore_web.book_parser_facilitateur import (
    NUMBERED_ITEM_RE,
    collect_numbered_items,
    split_and_parse_by_sections,
)

# ---------------------------------------------------------------------------
# NUMBERED_ITEM_RE — pattern matching
# ---------------------------------------------------------------------------


class TestNumberedItemRE:
    def test_plain_number(self):
        m = NUMBERED_ITEM_RE.match("3. Some item text")
        assert m is not None
        assert m.group(1) == "3"
        assert m.group(2) == "Some item text"

    def test_role_prefix_ayo(self):
        m = NUMBERED_ITEM_RE.match("(Ayo) 2. Tɩ ges pʋg-kõapa la kɩɩbsa yelle")
        assert m is not None
        assert m.group(1) == "2"
        assert m.group(2) == "Tɩ ges pʋg-kõapa la kɩɩbsa yelle"

    def test_role_prefix_nyee(self):
        m = NUMBERED_ITEM_RE.match("(Nyẽe) 6. Rũng sẽn deng n dum bãada")
        assert m is not None
        assert m.group(1) == "6"
        assert m.group(2) == "Rũng sẽn deng n dum bãada"

    def test_leading_whitespace(self):
        m = NUMBERED_ITEM_RE.match("  (Ayo) 1. Item with leading spaces")
        assert m is not None
        assert m.group(1) == "1"

    def test_no_match_role_only(self):
        assert NUMBERED_ITEM_RE.match("(Ayo)") is None

    def test_no_match_plain_text(self):
        assert NUMBERED_ITEM_RE.match("Some text without number") is None


# ---------------------------------------------------------------------------
# collect_numbered_items — integration
# ---------------------------------------------------------------------------


class TestCollectNumberedItems:
    def test_plain_numbered_items(self):
        lines = ["1. First item", "2. Second item", "3. Third item"]
        items = collect_numbered_items(lines)
        assert len(items) == 3
        assert items[0].number == 1 and items[0].text == "First item"
        assert items[1].number == 2
        assert items[2].number == 3

    def test_role_prefixed_items(self):
        lines = [
            "(Ayo) 2. Tɩ ges pʋg-kõapa la kɩɩbsa yelle",
            "(Nyẽe) 3. Yãmb sãn n dɩk a wamdã",
            "(Nyẽe) 6. Rũng sẽn deng n dum bãada",
        ]
        items = collect_numbered_items(lines)
        assert len(items) == 3
        assert items[0].number == 2
        assert items[0].text == "Tɩ ges pʋg-kõapa la kɩɩbsa yelle"
        assert items[1].number == 3
        assert items[2].number == 6

    def test_role_prefix_not_in_text(self):
        lines = ["(Ayo) 1. Item text here"]
        items = collect_numbered_items(lines)
        assert "(Ayo)" not in items[0].text

    def test_multi_line_item(self):
        lines = ["(Ayo) 2. First line of item", "continuation of item"]
        items = collect_numbered_items(lines)
        assert len(items) == 1
        assert "continuation" in items[0].text

    def test_mixed_plain_and_prefixed(self):
        lines = [
            "1. Plain item",
            "(Ayo) 2. Prefixed item",
            "3. Another plain",
        ]
        items = collect_numbered_items(lines)
        assert len(items) == 3
        assert [i.number for i in items] == [1, 2, 3]


# ---------------------------------------------------------------------------
# Minimal section regex list — just one pattern so tests stay readable
# ---------------------------------------------------------------------------


# Minimal section regex list — just one pattern so tests stay readable
_SEC_RE = [re.compile(r"^Section \d+$")]
_SEC_TITLES = ["Section"]


def _parse(text, *, start_after=None, stop_before=None):
    return split_and_parse_by_sections(
        text,
        _SEC_RE,
        _SEC_TITLES,
        start_after=start_after,
        stop_before=stop_before,
    )
