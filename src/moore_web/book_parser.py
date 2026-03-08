import unicodedata
import re
import pymupdf

import msgspec

from moore_web.book_enum_parser import group_chapter5_enums, EnumItem, ENUM_RAW


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
    enums: list[EnumItem] = msgspec.field(default_factory=list)

    @property
    def end_page(self) -> int:
        """Last page number of the chapter"""
        return self.pages[-1].page_number if self.pages else self.start_page

    @property
    def page_count(self) -> int:
        """Total number of pages in the chapter"""
        return len(self.pages)


def compile_robust_regex(pattern: str):
    normalized = pattern.replace(r"\s+", r"[\s-]+")
    return re.compile(normalized, re.IGNORECASE | re.MULTILINE)


CHAPTER_TITLES: dict[int, tuple[str, str]] = {
    1: ("Les secrets de Maman", "M ma solga bũmbu"),
    2: ("Davantage de problèmes dans la famille de Poko", "Yɛla ket n paasda a Pok rãmb zak n wã"),
    3: ("Des dangers pour la famille de Poko", "A Pok rãmb zaka zu-loeese"),
    4: ("Poko retrouve espoir", "A Pok ne a yaopa tõog n gesa b meng yelle"),
    5: (
        "La communauté de Poko s'informe sur le SIDA",
        "A Pok rãmb tẽnga rãmb baome n na n bãng sẽn kẽed SIDAwã bãag wɛɛngẽ",
    ),
}

CHAPTER_PAGES: dict[int, int] = {1: 3, 2: 10, 3: 17, 4: 25, 5: 32}

CHAPTERS_RAW: list[tuple[str, str]] = [
    (r"Chapitre\s+1\s+Les\s+secrets\s+de\s+Maman", r"Sak\s+a\s+1\s+soaba\s+M\s+ma\s+solga\s+bũmbu"),
    (
        r"Chapitre\s+2\s+Davantage\s+de\s+problèmes\s+dans\s+la\s+famille\s+de\s+Poko",
        r"Sak\s+a\s+2\s+soaba\s+Yɛla\s+ket\s+n\s+paasda\s+a\s+Pok\s+rãmb\s+zak\s+n\s+wã",
    ),
    (
        r"Chapitre\s+3\s+Des\s+dangers\s+pour\s+la\s+famille\s+de\s+Poko",
        r"Sak\s+a\s+3\s+soaba\s+A\s+Pok\s+rãmb\s+zaka\s+zu-loeese",
    ),
    (
        r"Chapitre\s+4\s+Poko\s+retrouve\s+espoir",
        r"Sak\s+a\s+4\s+soaba\s+A\s+Pok\s+ne\s+a\s+yaopa\s+tõog\s+n\s+gesa\s+b\s+meng\s+yelle",
    ),
    (
        r"Chapitre\s+5\s+La\s+communauté\s+de\s+Poko\s+s'informe\s+sur\s+le\s+SIDA\s*\.?\s*",
        r"Sak\s+a\s+5\s+soaba\s+A\s+Pok\s+rãmb\s+tẽnga\s+rãmb\s+baome\s+n\s+na\s+n\s+bãng\s+sẽn\s+kẽed\s+SIDAwã\s+bãag\s+wɛɛngẽ",
    ),
]


CHAPTERS_RE = [(compile_robust_regex(fr), compile_robust_regex(mo)) for fr, mo in CHAPTERS_RAW]

PAGE = {
    7: "Juste avant le commence-",
    8: "Quelques mois plus tard, Poko et sa mère   ramas-",
    45: """Même Jésus, lorsqu'Il  était sur la croix a crié à 
son Père en disant : 
"Père, pourquoi m'as-tu 
abandonné ?" Les 
gens enlèvent la douleur 
contenue dans leur 
cœur en parlant à Dieu 
ou à d'autres per-""",
}

PAGE_RE = {
    7: r"Juste\s+avant\s+le\s+commence-",
    8: r"Quelques\s+mois\s+plus\s+tard,\s+Poko\s+et\s+sa\s+mère\s+ramas-",
    45: r"Même\s+Jésus,\s+lorsqu'Il\s+"
    r"était\s+sur\s+la\s+croix\s+a\s+crié\s+à\s+"
    r"son\s+Père\s+en\s+disant\s+:\s+"
    r"\"\s*Père,\s+pourquoi\s+m'as-tu\s+"
    r"abandonné\s+\?\s+\"\s+Les\s+"
    r"gens\s+enlèvent\s+la\s+douleur\s+"
    r"contenue\s+dans\s+leur\s+"
    r"cœur\s+en\s+parlant\s+à\s+Dieu\s+"
    r"ou\s+à\s+d'autres\s+per-",
}

FRENCH_CLITICS = r"(?:ce|il|ils|elle|elles|on|je|tu|nous|vous|le|la|les|lui|y|en|t|ci|là|moi|toi|soi)"


def find_column_separator(page: pymupdf.Page):
    page_center = page.rect.width / 2
    candidates = []

    for d in page.get_drawings():
        for item in d["items"]:
            if item[0] == "l":
                _, (x0, y0), (x1, y1) = item
                if abs(x0 - x1) < 1 and abs(y1 - y0) > 200:
                    candidates.append(x0)

    if not candidates:
        return page_center

    return min(candidates, key=lambda x: abs(x - page_center))


def process_page_blocks(page: pymupdf.Page, middle_x: float) -> tuple[list[str], list[str]]:
    """
    Extracts text blocks and identifies if this page triggers a chapter start.

    Returns:
        tuple[list[str], list[str]]
    """
    moore_parts: list[str] = []
    french_parts: list[str] = []

    blocks = page.get_text("blocks", sort=True)

    for b in blocks:
        x0, y0, x1, y1, text = b[:5]
        text = text.strip()

        if not text or text.isdigit():
            continue
        if x0 < middle_x:
            moore_parts.append(text)
        else:
            french_parts.append(text)

    return moore_parts, french_parts


def normalize_mos_text(text: str) -> str:
    """Complete text normalization pipeline"""

    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u2019", "'")  # Right single quotation mark (')
    text = text.replace("\u2018", "'")  # Left single quotation mark (')
    text = text.replace("\u201c", '"')  # Left double quotation mark (")
    text = text.replace("\u201d", '"')  # Right double quotation mark (")
    text = text.replace("\u00ab", '"')  # Left-pointing double angle quotation mark («)
    text = text.replace("\u00bb", '"')

    text = text = re.sub(r"-\s*\n\s*", "", text)

    text = re.sub(r" +", " ", text)

    text = re.sub(r"\n\n+", "\n\n", text)

    return text.strip()


def normalize_fr_text(text: str) -> str:
    """Complete text normalization pipeline"""

    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u2019", "'")  # Right single quotation mark (')
    text = text.replace("\u2018", "'")  # Left single quotation mark (')
    text = text.replace("\u201c", '"')  # Left double quotation mark (")
    text = text.replace("\u201d", '"')  # Right double quotation mark (")
    text = text.replace("\u00ab", '"')  # Left-pointing double angle quotation mark («)
    text = text.replace("\u00bb", '"')

    text = re.sub(rf"(\w)-\s*\n\s*({FRENCH_CLITICS})\b", r"\1-\2", text)
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    text = re.sub(r" +", " ", text)
    text = re.sub(r"\s*([!?:;])", r" \1", text)

    text = re.sub(r"\n\n+", "\n\n", text)

    return text.strip()


def group_chapters(documents: pymupdf.Document) -> list[Chapter]:
    """
    Group document pages into chapters based on chapter start pages.

    Returns:
        List of Chapter objects with metadata and page content
    """
    chapters: list[Chapter] = []
    current_chapter_pages = []
    current_chapter_num = None
    current_start_page = None

    for page_num, page in enumerate(documents, start=1):  # type: ignore
        if page_num > 47:
            break
        x = find_column_separator(page)
        moore_text, french_text = process_page_blocks(page=page, middle_x=x)

        french_text = normalize_fr_text("\n".join(french_text))
        moore_text = normalize_mos_text("\n".join(moore_text))
        if page_num == 39:
            moore_marker_pattern = (
                r"1\.\s+SIDAwã\s+bãag\s+ya\s+boẽ\?\s+La\s+a\s+maanda\s+a\s+wãn\s+n\s+kʋʋd\s+nebã\?"
            )
            french_marker_pattern = r"Mais\s+quand\s+le\s+VIH\s+entre\s+dans\s+le\s+corps\s+d'une\s+personne"

            moore_match = re.search(moore_marker_pattern, moore_text)
            french_match = re.search(french_marker_pattern, moore_text)

            if moore_match and french_match:
                french_part_1 = moore_text[: moore_match.start()].strip()
                moore_only = moore_text[moore_match.start() : french_match.start()].strip()
                french_part_2 = moore_text[french_match.start() :].strip()
                french_text = french_part_1 + "\n\n" + french_part_2 + "\n\n" + french_text
                moore_text = moore_only = re.sub(
                    r"Ba\s+yẽ\s+bã'abiire\s+zɩɩm\s+pʋam\.", "", moore_only
                ).strip()

        if page_num in CHAPTER_PAGES.values():
            if current_chapter_num is not None and current_chapter_pages:
                title_fr, title_mo = CHAPTER_TITLES[current_chapter_num]
                chapters.append(
                    Chapter(
                        chapter_number=current_chapter_num,
                        title_french=title_fr,
                        title_moore=title_mo,
                        start_page=current_start_page,
                        pages=current_chapter_pages,
                    )
                )

            current_chapter_num = [k for k, v in CHAPTER_PAGES.items() if v == page_num][0]
            current_start_page = page_num
            current_chapter_pages = []
            chapter_idx = list(CHAPTER_PAGES.values()).index(page_num)
            regex_fr, regex_mo = CHAPTERS_RE[chapter_idx]

            french_text = re.sub(regex_fr, "", french_text, count=1).strip()
            moore_text = re.sub(regex_mo, "", moore_text, count=1).strip()
            if current_chapter_num == 5:
                moore_text = re.sub(regex_fr, "", moore_text, count=1).strip()

        current_chapter_pages.append(
            ChapterPage(page_number=page_num, french_text=french_text, moore_text=moore_text)
        )

    if current_chapter_num is not None and current_chapter_pages:
        title_fr, title_mo = CHAPTER_TITLES[current_chapter_num]
        chapters.append(
            Chapter(
                chapter_number=current_chapter_num,
                title_french=title_fr,
                title_moore=title_mo,
                start_page=current_start_page,  # type: ignore
                pages=current_chapter_pages,
            )
        )

    return chapters


def fix_hyphenated_sentences(pages: list[ChapterPage]) -> list[ChapterPage]:
    """
    Fix sentences that end with hyphen and continue on next page.
    Removes duplicated text from the next page.
    """
    fixed_pages = pages.copy()

    for i, page in enumerate(fixed_pages[:-1]):
        current_text = page.french_text.rstrip()
        page_number = page.page_number

        if page_number not in PAGE:
            continue
        next_page = fixed_pages[i + 1]
        next_text = next_page.french_text.lstrip()

        incomplete_text = PAGE[page_number]
        pattern = PAGE_RE[page_number]

        if not re.search(pattern, current_text):
            continue

        remainder = remainder = re.sub(rf"\s*{pattern}\s*$", "", current_text)

        merged_french = incomplete_text[:-1] + next_text

        fixed_pages[i] = ChapterPage(
            page_number=page.page_number,
            french_text=remainder,
            moore_text=page.moore_text,
        )

        fixed_pages[i + 1] = ChapterPage(
            page_number=next_page.page_number,
            french_text=merged_french,
            moore_text=next_page.moore_text,
        )

    return fixed_pages


def parse_pdf_to_json(
    input_pdf: str, output_path: str | None = None
) -> list[Chapter]:
    with pymupdf.open(input_pdf) as doc:
        chapters = group_chapters(doc)

    for chapter in chapters:
        chapter.pages = fix_hyphenated_sentences(chapter.pages)

    chapter5 = next((ch for ch in chapters if ch.chapter_number == 5), None)
    if chapter5:
        chapter5.enums = group_chapter5_enums(chapter5.pages, ENUM_RAW, enum_start_page=39)

    if output_path:
        with open(output_path, "wb") as f:
            f.write(msgspec.json.encode(chapters))

    return chapters


if __name__ == "__main__":
    input_pdf = "data/2 SIDA mooré - français.pdf"
    out_path = "aligned_parsed.json"

    chapters = parse_pdf_to_json(input_pdf, out_path)
    chapter5 = next((ch for ch in chapters if ch.chapter_number == 5), None)
    enum_count = len(chapter5.enums) if chapter5 else 0
    print(f"Parsed {len(chapters)} chapters, {enum_count} Chapter-5 enum items → {out_path}")

    chapter_lines = []
    for chapter in chapters:
        for page in chapter.pages:
            text = page.french_text.replace("\n", "").strip()
            if text:
                chapter_lines.append(text)

    with open("french_lines.txt", "w") as f:
        f.write("\n".join(chapter_lines))
    print(f"Wrote {len(chapter_lines)} French lines → french_lines.txt")
