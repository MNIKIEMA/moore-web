from moore_web.book_parser import process_page_blocks


class _FakePage:
    """Stands in for a pymupdf.Page: only `get_text("blocks")` is used."""

    def __init__(self, blocks: list[tuple[float, float, float, float, str]]) -> None:
        self._blocks = blocks

    def get_text(self, option: str) -> list[tuple[float, float, float, float, str]]:
        assert option == "blocks"
        return self._blocks


def test_blocks_are_ordered_by_top_edge() -> None:
    # Page 42: the continuation paragraph's box overhangs the next heading, so
    # ordering by the bottom edge put the heading first.
    page = _FakePage(
        [
            (39.2, 266.2, 206.8, 281.7, "4. Boẽ ne boẽ n wiligd tɩ"),
            (30.7, 46.6, 220.3, 336.9, "Nin-kãng toẽ n tara laafɩ"),
        ]
    )
    moore, french = process_page_blocks(page, middle_x=229.1)
    assert moore == ["Nin-kãng toẽ n tara laafɩ", "4. Boẽ ne boẽ n wiligd tɩ"]
    assert french == []


def test_blocks_are_assigned_to_a_column_by_midpoint() -> None:
    # Page 39 has no drawn separator, so the page centre (210) is used and the
    # French blocks start just left of it.
    page = _FakePage(
        [
            (34.9, 46.6, 192.1, 500.0, "1. SIDAwã bãag ya boẽ?"),
            (207.8, 48.6, 383.3, 90.0, "1. Qu'est-ce que le SIDA"),
            (204.8, 552.2, 217.4, 565.5, "39"),
        ]
    )
    moore, french = process_page_blocks(page, middle_x=210.0)
    assert moore == ["1. SIDAwã bãag ya boẽ?"]
    assert french == ["1. Qu'est-ce que le SIDA"]
