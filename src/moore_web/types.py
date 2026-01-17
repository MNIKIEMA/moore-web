import msgspec


class ChapterPage(msgspec.Struct):
    """Represents a single page within a chapter"""

    page_number: int
    french_text: str
    moore_text: str


class Chapter(msgspec.Struct):
    """Represents a complete chapter with metadata"""

    chapter_number: int
    title_french: str
    title_moore: str
    start_page: int
    pages: list[ChapterPage]

    @property
    def end_page(self) -> int:
        """Last page number of the chapter"""
        return self.pages[-1].page_number if self.pages else self.start_page

    @property
    def page_count(self) -> int:
        """Total number of pages in the chapter"""
        return len(self.pages)
