"""Tests for book_parser_facilitateur: NUMBERED_ITEM_RE, collect_numbered_items,
and start_after / stop_before sentinels in split_and_parse_by_sections."""

from moore_web.book_parser_facilitateur import (
    NUMBERED_ITEM_RE,
    BulletItem,
    NumberedItem,
    collect_items,
    collect_numbered_items,
    flatten_content,
)

# ---------------------------------------------------------------------------
# NUMBERED_ITEM_RE — pattern matching
# ---------------------------------------------------------------------------


class TestNumberedItemRE:
    def test_plain_number(self):
        s = """2. 
Yãmb sãn tʋm ne pĩim b sẽn zoe n dɩk n tʋm ne ned sẽn tar

SIDAwã n yaol n ka dʋg-a ? (Nyẽe)"""
        m = NUMBERED_ITEM_RE.match(s)
        assert m is not None
        assert m.group(1) == "2"
        assert m.group(2) == "Yãmb sãn tʋm ne pĩim b sẽn zoe n dɩk n tʋm ne ned sẽn tar"

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
# flatten_content — document order of items and bullets (issue #44)
# ---------------------------------------------------------------------------


class TestFlattenContentOrder:
    def test_bullets_nested_under_an_item_stay_after_it(self):
        # Mooré ch5 "Bũmb d sẽn tõe n zãmse": three bullets under item 2, then
        # the next question (absorbed into the last bullet) and its own items.
        lines = [
            "1. Tõnd segd n tagsame.",
            "2. Yãmb sãn n dɩk tẽeb kãense, yãmb na yã :",
            "• Tẽeb sẽn ya sõngre.",
            "• Tẽeb sẽn wat ne bãag saagre.",
            "Tẽeb bʋs n ya sõama ?",
            "1. Tẽms wʋsog pʋsẽ.",
        ]
        items, bullets = collect_items(lines)
        assert flatten_content(items, bullets, "") == [
            "Tõnd segd n tagsame.",
            "Yãmb sãn n dɩk tẽeb kãense, yãmb na yã :",
            "Tẽeb sẽn ya sõngre.",
            "Tẽeb sẽn wat ne bãag saagre. Tẽeb bʋs n ya sõama ?",
            "Tẽms wʋsog pʋsẽ.",
        ]

    def test_items_without_positions_keep_items_before_bullets(self):
        # A book JSON saved before `line` existed decodes with line == -1.
        items = [NumberedItem(1, "Item one"), NumberedItem(2, "Item two")]
        bullets = [BulletItem("Bullet")]
        assert flatten_content(items, bullets, "Body") == ["Body", "Item one", "Item two", "Bullet"]
