from moore_web.book_enum_parser import EnumItem
from moore_web.book_parser import ChapterPage, group_chapters
import re
import pymupdf
import msgspec

from moore_web.book_enum_parser import group_chapter5_enums, ENUM_RAW


class SentencePair(msgspec.Struct):
    french: str
    moore: str
    source: str
    index: int


def remove_newlines(text: str) -> str:
    """Remove all newlines and normalize whitespace"""
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_semicolons_outside_quotes(sentence: str) -> list[str]:
    parts, current, in_quote = [], [], False
    for ch in sentence:
        if ch in {'"', "«", "»"}:
            in_quote = not in_quote
        if ch == ";" and not in_quote:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current).strip())
    return parts


def split_sentences(text: str) -> list[str]:
    """
    Split text into sentences:
    1. Split on . ! ? followed by optional closing quote and whitespace
    2. Split on ; only if outside quotes
    """
    text = remove_newlines(text)

    boundary = r'([.!?]["»”]?)\s+'
    parts = re.split(boundary, text)

    sentences = []
    i = 0
    while i < len(parts):
        if i + 1 < len(parts) and re.fullmatch(r'[.!?]["»”]?', parts[i + 1]):
            sentences.append((parts[i] + parts[i + 1]).strip())
            i += 2
        else:
            if parts[i].strip():
                sentences.append(parts[i].strip())
            i += 1

    final = []
    for s in sentences:
        if ";" in s:
            final.extend(split_semicolons_outside_quotes(s))
        else:
            final.append(s)

    return [s for s in final if s]


def segment_enum(enum_items: list[EnumItem]) -> tuple[list[str], list[str]]:
    """
    Prepare enumeration items for alignment by splitting into sentences.
    Returns (french_sentences, moore_sentences)
    """
    french_sentences = []
    moore_sentences = []

    for item in enum_items:
        fr_sents = split_sentences(item.french_text)
        french_sentences.extend(fr_sents)

        mo_sents = split_sentences(item.moore_text)
        moore_sentences.extend(mo_sents)

    return french_sentences, moore_sentences


def segment_pages(
    pages: list[ChapterPage], start_page: int | None = None, end_page: int | None = None
) -> tuple[list[str], list[str]]:
    """
    Prepare regular chapter pages for alignment.
    Returns (french_sentences, moore_sentences)
    """
    filtered_pages = pages

    if start_page is not None:
        filtered_pages = [p for p in filtered_pages if p.page_number >= start_page]
    if end_page is not None:
        filtered_pages = [p for p in filtered_pages if p.page_number < end_page]

    all_french = " ".join([p.french_text for p in filtered_pages])
    all_moore = " ".join([p.moore_text for p in filtered_pages])

    french_sentences = split_sentences(all_french)
    moore_sentences = split_sentences(all_moore)

    return french_sentences, moore_sentences


def save_sentence_pairs_jsonl(sentence_pairs: list[SentencePair], output_path: str):
    encoder = msgspec.json.Encoder()

    with open(output_path, "wb") as f:
        for sents_pair in sentence_pairs:
            f.write(encoder.encode(sents_pair))
            f.write(b"\n")


if __name__ == "__main__":
    input_pdf = "data/2 SIDA mooré - français.pdf"

    with pymupdf.open(input_pdf) as doc:
        chapters = group_chapters(doc)
    chapter5 = chapters[-1]
    enum_items = group_chapter5_enums(chapter5.pages, ENUM_RAW, enum_start_page=39)
    fr_sents, mo_sents = segment_enum(enum_items)
    enum_sentence_pairs = [
        SentencePair(french=fr, moore=mo, source="enum", index=i)
        for i, (fr, mo) in enumerate(zip(fr_sents, mo_sents))
    ]

    all_pages = []
    for chapter in chapters:
        all_pages.extend(chapter.pages)
    fr_sents, mo_sents = segment_pages(all_pages, end_page=39)
    page_sentence_pairs = [
        SentencePair(french=fr, moore=mo, source="page", index=i)
        for i, (fr, mo) in enumerate(zip(fr_sents, mo_sents))
    ]

    sentence_pairs = page_sentence_pairs + enum_sentence_pairs
    save_sentence_pairs_jsonl(sentence_pairs, "data/sentence_pairs.jsonl")
