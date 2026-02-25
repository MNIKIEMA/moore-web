from moore_web.book_enum_parser import group_chapter5_enums, ENUM_RAW, EnumItem
from moore_web.book_parser import parse_pdf_to_json, Chapter
from argparse import ArgumentParser
from pathlib import Path
import re
import msgspec
from sacremoses import MosesTokenizer, MosesDetokenizer


mt = MosesTokenizer(lang="fr")
md = MosesDetokenizer(lang="fr")

def clean(s: str) -> str:
    s = re.sub(r"\n+", " ", s)
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def flatten_content(
    items: list[EnumItem],
) -> tuple[list[str], list[str]]:
    """
    Flattens numbered items, bullet items, and body text into a single list of strings.
    Numbered items are prefixed with their number, body is split into sentences.
    """
    enums_fr: list[str] = []
    enums_mo: list[str] = []

    for item in items:
        if item.french_text:
            enums_fr.append(clean(item.french_text))
        if item.moore_text:
            enums_mo.append(clean(item.moore_text))
    return enums_fr, enums_mo


def flatten_book_to_list(chapters: list[Chapter], enum_start_page: int = 39) -> tuple[list[str], list[str]]:
    result_fr: list[str] = []
    result_mo: list[str] = []
    for chapter in chapters:
        for page in chapter.pages:
            page_number = page.page_number
            if page_number >= enum_start_page:
                break
            french = page.french_text
            mos = page.moore_text
            if french:
                french_tokens = mt.tokenize(french)
                french = md.detokenize(french_tokens)
                result_fr.append(clean(french))
            if mos:
                result_mo.append(clean(mos))
    return result_fr, result_mo


def parse_book_from_json(json_path: str) -> list[Chapter]:
    json_data = Path(json_path).read_text(encoding="utf-8")
    book = msgspec.json.decode(json_data, type=list[Chapter])
    return book


def main():
    parser = ArgumentParser(description="Flatten a book JSON into a list of strings.")
    parser.add_argument("--input", "-i", required=True, help="Path to the input book JSON file.")
    parser.add_argument("--output", "-o", required=True, help="Path to the output text file.")
    args = parser.parse_args()

    chapters = parse_pdf_to_json(args.input)
    chapter5 = chapters[-1]
    enum_items = group_chapter5_enums(chapter5.pages, ENUM_RAW, enum_start_page=39)

    output_path = Path(args.output)
    flattened_fr, flattened_mo = flatten_book_to_list(chapters)
    enums_fr, enums_mo = flatten_content(enum_items)

    all_fr = "\n".join(flattened_fr + enums_fr)
    all_mo = "\n".join(flattened_mo + enums_mo)

    output_path_fr = output_path.with_suffix(".fr.txt")
    output_path_mo = output_path.with_suffix(".mo.txt")
    output_path_fr.write_text(all_fr, encoding="utf-8")
    output_path_mo.write_text(all_mo, encoding="utf-8")


if __name__ == "__main__":
    main()
