from __future__ import annotations

import re
import msgspec
from typing import Optional, TypeAlias, TYPE_CHECKING

if TYPE_CHECKING:
    from moore_web.book_parser import ChapterPage


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


class EnumItem(msgspec.Struct):
    """Represents a single enumeration item in Chapter 5."""

    enum_number: int
    french_title: str
    moore_title: str
    french_text: str
    moore_text: str
    start_page: int
    end_page: int


EnumPattern: TypeAlias = tuple[re.Pattern, re.Pattern, re.Pattern | None]


def compile_robust_enum_regex(pattern: str) -> re.Pattern:
    """Compile regex with robust whitespace and hyphen handling."""
    normalized = pattern.replace(r"\s+", r"[\s\-]+")
    normalized = normalized.replace("quelques-uns", "quelques-?uns")
    return re.compile(normalized, re.IGNORECASE | re.MULTILINE | re.DOTALL)


def compile_enum_patterns(enum_raw: list[tuple[str, str]]) -> list[tuple[re.Pattern, re.Pattern]]:
    """Compile robust regex patterns for each enum definition."""
    return [
        (compile_robust_enum_regex(fr), compile_robust_enum_regex(mo))
        for fr, mo in enum_raw
    ]


def extract_enum_number(text: str) -> Optional[int]:
    """Extract the enumeration number from text like '1. ' or '2. '"""
    match = re.match(r"^(\d+)\.\s", text.strip())
    return int(match.group(1)) if match else None


def group_chapter5_enums(
    chapter5_pages: list[ChapterPage],
    enum_raw: list[tuple[str, str]] = ENUM_RAW,
    enum_start_page: int = 39,
) -> list[EnumItem]:
    """
    Group enumeration items in Chapter 5, handling cases where items span multiple pages.

    Args:
        chapter5_pages: List of ChapterPage objects for chapter 5.
        enum_raw: List of (french_pattern, moore_pattern) tuples with raw regex.
        enum_start_page: Absolute page number where enums start (default: 39).

    Returns:
        List of EnumItem objects with complete text and page ranges.
    """
    filtered_pages = [p for p in chapter5_pages if p.page_number >= enum_start_page]
    if not filtered_pages:
        return []

    enum_patterns = compile_enum_patterns(enum_raw)
    tok_regex_fr_str = "(" + "|".join(p[0].pattern for p in enum_patterns) + ")"
    tok_regex_mo_str = "(" + "|".join(p[1].pattern for p in enum_patterns) + ")"
    tok_regex_fr = re.compile(tok_regex_fr_str, re.IGNORECASE | re.MULTILINE | re.DOTALL)
    tok_regex_mo = re.compile(tok_regex_mo_str, re.IGNORECASE | re.MULTILINE | re.DOTALL)

    all_french_text = "\n".join(p.french_text for p in filtered_pages)
    all_moore_text = "\n".join(p.moore_text for p in filtered_pages)

    split_fr = tok_regex_fr.split(all_french_text)
    split_mo = tok_regex_mo.split(all_moore_text)

    fr_pairs = [
        (extract_enum_number(split_fr[i].strip()), split_fr[i].strip(), split_fr[i + 1].strip() if i + 1 < len(split_fr) else "")
        for i in range(1, len(split_fr), 2)
    ]
    mo_pairs = [
        (extract_enum_number(split_mo[i].strip()), split_mo[i].strip(), split_mo[i + 1].strip() if i + 1 < len(split_mo) else "")
        for i in range(1, len(split_mo), 2)
    ]
    mo_by_num = {num: (title, text) for num, title, text in mo_pairs if num is not None}

    enum_items: list[EnumItem] = []
    for fr_num, fr_title, fr_content in fr_pairs:
        if fr_num is None:
            continue
        mo_title, mo_content = mo_by_num.get(fr_num, ("", ""))
        enum_items.append(
            EnumItem(
                enum_number=fr_num,
                french_title=fr_title,
                moore_title=mo_title,
                french_text=fr_content,
                moore_text=mo_content,
                start_page=enum_start_page,
                end_page=enum_start_page,
            )
        )

    return enum_items
