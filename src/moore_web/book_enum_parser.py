from moore_web.book_parser import ChapterPage
import re
from dataclasses import dataclass
from typing import Optional, TypeAlias


ENUM_RAW: list[tuple[str, str]] = [
    # Enum 1
    (
        r"1\.\s+Qu'est-ce\s+que\s+le\s+SIDA\s+et\s+comment\s+cause-t-il\s+la\s+mort\s*\?",
        r"1\.\s+SIDAwã\s+bãag\s+ya\s+boẽ\?\s+La\s+a\s+maanda\s+a\s+wãn\s+n\s+kʋʋd\s+nebã\?",
    ),
    # Enum 2
    (
        r"2\.\s+Comment\s+une\s+personne\s+attrape-t-elle\s+le\s+VIH\s*\?",
        r"2\.\s+SIDAwã\s+bãag\s+maanda\s+a\s+wãn\s+n\s+yõk\s+neda\?",
    ),
    # Enum 3
    (
        r"3\.\s+Comment\s+une\s+personne\s+peut-elle\s+savoir\s+si\s+elle\s+a\s+attrapé\s+le\s+VIH\s*\?",
        r"3\.\s+Ned\s+na\s+n\s+maana\s+a\s+wãn\s+n\s+bãng\s+tɩ\s+SIDAwã\s+bãag\s+tar-a\s+lame\?",
    ),
    # Enum 4
    (
        r"4\.\s+Quels\s+sont\s+quelques-uns\s+des\s+signes\s+qui\s+peuvent\s+indiquer\s+qu'une\s+personne\s+peut\s+avoir\s+le\s+SIDA\s*\?",
        r"4\.\s+Boẽ\s+ne\s+boẽ\s+n\s+wiligd\s+tɩ\s+ned\s+tara\s+SIDAwã\s+bãagã\?",
    ),
    # Enum 5
    (
        r"5\.\s+Y\s+a-t-il\s+des\s+médicaments\s+qu'une\s+personne\s+atteinte\s+du\s+SIDA\s+peut\s+prendre\s+pour\s+se\s+sentir\s+mieux\s*\?",
        r"5\.\s+Tɩɩm\s+beeme\s+tɩ\s+a\s+soab\s+ninga\s+SIDAwã\s+bãag\s+sẽn\s+tara\s+wã\s+toẽ\s+n\s+dɩkdẽ\s+tɩ\s+sãõog\s+bii\?",
    ),
    # Enum 6
    (
        r"6\.\s+Nous\s+connaissons\s+une\s+personne\s+qui\s+a\s+le\s+SIDA\.\s+Elle\s+va\s+bientôt\s+mourir\.\s+Comment\s+l'aider\s*\?",
        r"6\.\s+Tõnd\s+sã\s+n\s+mi\s+ned\s+SIDAwã\s+bãag\s+sẽn\s+tar-a,\s+tɩ\s+d\s+mii\s+t'a\s+na\s+n\s+kiimi;\s+d\s+na\s+n\s+maana\s+a\s+wãn\s+n\s+sõng-a\?",
    ),
]


@dataclass
class EnumItem:
    """Represents a single enumeration item in Chapter 5"""

    enum_number: int
    french_title: str
    moore_title: str
    french_text: str
    moore_text: str
    start_page: int
    end_page: int


def compile_robust_enum_regex(pattern: str) -> re.Pattern:
    """
    Compile regex with robust whitespace and hyphen handling.
    Allows for line breaks, extra spaces, and hyphens in place of spaces.
    """
    normalized = pattern.replace(r"\s+", r"[\s\-]+")
    normalized = normalized.replace("quelques-uns", "quelques-?uns")

    return re.compile(normalized, re.IGNORECASE | re.MULTILINE | re.DOTALL)


def compile_enum_patterns(enum_raw: list[tuple[str, str]]) -> list[tuple[re.Pattern, re.Pattern]]:
    """
    Compile robust regex patterns for each enum definition.

    Args:
        enum_raw: List of (french_pattern, moore_pattern) tuples

    Returns:
        List of compiled pattern tuples
    """
    patterns = []
    for fr_pattern, mo_pattern in enum_raw:
        fr_compiled = compile_robust_enum_regex(fr_pattern)
        mo_compiled = compile_robust_enum_regex(mo_pattern)
        patterns.append((fr_compiled, mo_compiled))

    return patterns


def extract_enum_number(text: str) -> Optional[int]:
    """Extract the enumeration number from text like '1. ' or '2. '"""
    match = re.match(r"^(\d+)\.\s", text.strip())
    return int(match.group(1)) if match else None


def find_enum_in_text(
    text: str, patterns: list[tuple[re.Pattern, re.Pattern, str]], enum_number: int
) -> Optional[tuple[int, int]]:
    """
    Find the position of an enum pattern in text.

    Returns:
        Tuple of (start_pos, end_pos) if found, None otherwise
    """
    if enum_number < 1 or enum_number > len(patterns):
        return None

    fr_pattern, mo_pattern, _ = patterns[enum_number - 1]

    fr_match = fr_pattern.search(text)
    if fr_match:
        return fr_match.span()

    return None


def group_chapter5_enums(
    chapter5_pages: list[ChapterPage], enum_raw: list[tuple[str, str]] = ENUM_RAW, enum_start_page: int = 39
) -> list[EnumItem]:
    """
    Group enumeration items in Chapter 5, handling cases where items span multiple pages.

    Args:
        chapter5_pages: List of ChapterPage objects for chapter 5
        enum_raw: List of (french_pattern, moore_pattern) tuples with raw regex
        enum_start_page: Absolute page number where enums start (default: 39)

    Returns:
        List of EnumItem objects with complete text and page ranges
    """

    filtered_pages: list[ChapterPage] = [p for p in chapter5_pages if p.page_number >= enum_start_page]
    print(f"Filtered to {len(filtered_pages)} pages starting from page {enum_start_page}.")

    if not filtered_pages:
        return []

    enum_patterns = compile_enum_patterns(enum_raw)
    enum_items: list[EnumItem] = []
    tok_regex_fr: list[re.Pattern] = [p[0] for p in enum_patterns]
    tok_regex_mo: list[re.Pattern] = [p[1] for p in enum_patterns]

    tok_regex_fr_str = "(" + "|".join([p.pattern for p in tok_regex_fr]) + ")"
    tok_regex_mo_str = "(" + "|".join([p.pattern for p in tok_regex_mo]) + ")"

    tok_regex_fr = re.compile(tok_regex_fr_str, re.IGNORECASE | re.MULTILINE | re.DOTALL)
    tok_regex_mo = re.compile(tok_regex_mo_str, re.IGNORECASE | re.MULTILINE | re.DOTALL)

    all_french_list: list[str] = []
    all_moore_list: list[str] = []

    for page in filtered_pages:
        all_french_list.append(page.french_text)
        all_moore_list.append(page.moore_text)

    all_french_text = "\n".join(all_french_list)
    all_moore_text = "\n".join(all_moore_list)

    split_fr = tok_regex_fr.split(all_french_text)
    split_mo = tok_regex_mo.split(all_moore_text)

    print(f"French split into {len(split_fr)} parts")
    print(f"Moore split into {len(split_mo)} parts")

    fr_pairs = []
    for i in range(1, len(split_fr), 2):
        title = split_fr[i].strip()
        content = split_fr[i + 1].strip() if i + 1 < len(split_fr) else ""
        enum_num = extract_enum_number(title)
        fr_pairs.append((enum_num, title, content))

    mo_pairs = []
    for i in range(1, len(split_mo), 2):
        title = split_mo[i].strip()
        content = split_mo[i + 1].strip() if i + 1 < len(split_mo) else ""
        enum_num = extract_enum_number(title)
        mo_pairs.append((enum_num, title, content))

    for fr_num, fr_title, fr_content in fr_pairs:
        if fr_num is None:
            continue

        mo_content = ""
        for mo_num, mo_title, mo_text in mo_pairs:
            if mo_num == fr_num:
                mo_content = mo_text
                break

        enum_items.append(
            EnumItem(
                enum_number=fr_num,
                french_text=fr_content,
                moore_text=mo_content,
                french_title=fr_title,
                moore_title=mo_title,
                start_page=enum_start_page,
                end_page=enum_start_page,
            )
        )

    return enum_items


EnumPattern: TypeAlias = tuple[re.Pattern, re.Pattern, re.Pattern | None]


if __name__ == "__main__":
    import pymupdf
    from moore_web.book_parser import group_chapters

    input_pdf = "data/2 SIDA mooré - français.pdf"
    with pymupdf.open(input_pdf) as doc:
        chapters = group_chapters(doc)

    chapter5 = chapters[-1]

    enum_items = group_chapter5_enums(chapter5.pages, ENUM_RAW, enum_start_page=39)

    print(f"\nFound {len(enum_items)} enumeration items in Chapter 5:")
    for item in enum_items:
        print(f"\n{'=' * 80}")
        print(f"Enum {item.enum_number}: Pages {item.start_page}-{item.end_page}")
        print(f"\nFrench: {item.french_text}")
        print(f"\nMoore: {item.moore_text}")
    pass
