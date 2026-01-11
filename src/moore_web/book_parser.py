import re
import pymupdf
import json

from moore_web.constants import CHAPTERS, CHAPTER_FIVE_ENUM


def normalize(text):
    """Normalize whitespace in text."""
    return re.sub(r"\s+", " ", text).strip()


def normalize_for_matching(text):
    """Normalize text for chapter matching."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def is_incomplete_sentence(text):
    """Check if text ends with incomplete sentence (no proper ending punctuation)."""
    if not text:
        return False
    text = text.rstrip()
    return not text.endswith((".", "!", "?", "»", '"', '"'))


def split_sentences(text):
    """Split text into sentences, handling French and Mooré punctuation."""
    sentences = re.split(r'(?<=[.!?»""])\s+(?=[A-ZÀÂÄÇÈÉÊËÏÎÔÙÛÜŸŒÆ«""A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]


def align_sentences(fr_text, mo_text):
    """
    Align French and Mooré sentences intelligently.
    """
    fr_sents = split_sentences(fr_text)
    mo_sents = split_sentences(mo_text)

    aligned = []

    if len(fr_sents) == len(mo_sents):
        for f, m in zip(fr_sents, mo_sents):
            aligned.append({"fr": f, "mo": m})

    elif abs(len(fr_sents) - len(mo_sents)) <= 2 and min(len(fr_sents), len(mo_sents)) > 0:
        min_len = min(len(fr_sents), len(mo_sents))

        for i in range(min_len):
            aligned.append({"fr": fr_sents[i], "mo": mo_sents[i]})

        if len(fr_sents) > min_len:
            remaining_fr = " ".join(fr_sents[min_len:])
            aligned.append({"fr": remaining_fr, "mo": ""})
        elif len(mo_sents) > min_len:
            remaining_mo = " ".join(mo_sents[min_len:])
            aligned.append({"fr": "", "mo": remaining_mo})

    else:
        if fr_text.strip() or mo_text.strip():
            aligned.append({"fr": fr_text.strip(), "mo": mo_text.strip()})

    return aligned


def split_page(page):
    """Extract text from left (Mooré) and right (French) columns."""
    middle_x = page.rect.width / 2
    moore_blocks = []
    french_blocks = []

    blocks = page.get_text("blocks", sort=True)

    for block in blocks:
        x0, y0, x1, y1, text, *_ = block
        text = text.strip()

        if not text or (text.isdigit() and len(text) <= 3):
            continue

        if x0 < middle_x:
            moore_blocks.append(text)
        else:
            french_blocks.append(text)

    return "\n".join(moore_blocks), "\n".join(french_blocks)


def detect_chapter(fr_text, mo_text):
    """Detect chapter heading on page."""
    fr_norm = normalize_for_matching(fr_text)
    mo_norm = normalize_for_matching(mo_text)

    for idx, (fr_pattern, mo_pattern) in enumerate(CHAPTERS, start=1):
        fr_pat_norm = normalize_for_matching(fr_pattern)
        mo_pat_norm = normalize_for_matching(mo_pattern)

        if fr_pat_norm in fr_norm or mo_pat_norm in mo_norm:
            return idx, fr_pattern, mo_pattern

    return None, None, None


def remove_heading(text, heading):
    """Remove chapter heading from text."""
    if not heading or not text:
        return text
    heading_norm = normalize_for_matching(heading)
    text_norm = normalize_for_matching(text)

    if heading_norm in text_norm:
        idx = text_norm.index(heading_norm)

        return text[idx + len(heading) :].strip()

    return text


def create_chapter_5_enumerations():
    """Create enumeration items for Chapter 5."""
    items = []
    for i, (fr1, mo, fr2) in enumerate(CHAPTER_FIVE_ENUM, start=1):
        fr_complete = normalize(fr1 + " " + fr2)
        mo_normalized = normalize(mo)

        items.append({"item_id": f"5.{i}", "type": "enumeration", "fr": fr_complete, "mo": mo_normalized})
    return items


def parse_pdf_to_json(pdf_path, out_path):
    """
    Parse bilingual PDF with proper handling of text continuation across pages.
    """
    corpus = []

    with pymupdf.open(pdf_path) as doc:
        current_chapter_id = None
        current_chapter_data = None

        carry_fr = ""
        carry_mo = ""

        chapter_5_enum_added = False

        print(f"Processing {len(doc)} pages...\n")

        for page_num, page in enumerate(doc, start=1):
            mo_text, fr_text = split_page(page)

            chap_id, fr_heading, mo_heading = detect_chapter(fr_text, mo_text)

            if chap_id is not None:
                print(f"Page {page_num}: *** CHAPTER {chap_id} DETECTED ***")

                if carry_fr or carry_mo:
                    if current_chapter_data is not None:
                        alignment = align_sentences(carry_fr, carry_mo)
                        if alignment:
                            current_chapter_data["pages"].append(
                                {
                                    "page_num": page_num - 1,
                                    "alignment": alignment,
                                    "note": "carried_from_previous",
                                }
                            )
                    carry_fr, carry_mo = "", ""

                if current_chapter_data is not None:
                    corpus.append(current_chapter_data)

                current_chapter_id = chap_id
                current_chapter_data = {"chapter_id": chap_id, "pages": []}

                if chap_id == 5 and not chapter_5_enum_added:
                    current_chapter_data["enumerations"] = create_chapter_5_enumerations()
                    chapter_5_enum_added = True
                    print(f"  -> Added {len(CHAPTER_FIVE_ENUM)} enumeration items")

                fr_text = remove_heading(fr_text, fr_heading)
                mo_text = remove_heading(mo_text, mo_heading)

            if current_chapter_id is None:
                continue

            fr_text = normalize(fr_text)
            mo_text = normalize(mo_text)

            if carry_fr:
                fr_text = carry_fr + " " + fr_text
                carry_fr = ""
            if carry_mo:
                mo_text = carry_mo + " " + mo_text
                carry_mo = ""

            fr_incomplete = is_incomplete_sentence(fr_text)
            mo_incomplete = is_incomplete_sentence(mo_text)

            if fr_incomplete or mo_incomplete:
                print(
                    f"Page {page_num}: Text continues to next page (FR: {fr_incomplete}, MO: {mo_incomplete})"
                )

                carry_fr = fr_text
                carry_mo = mo_text
                continue

            if fr_text or mo_text:
                alignment = align_sentences(fr_text, mo_text)

                if alignment:
                    current_chapter_data["pages"].append({"page_num": page_num, "alignment": alignment})
                    print(f"Page {page_num}: Added {len(alignment)} segment(s)")

        if carry_fr or carry_mo:
            if current_chapter_data is not None:
                alignment = align_sentences(carry_fr, carry_mo)
                if alignment:
                    current_chapter_data["pages"].append(
                        {"page_num": page_num, "alignment": alignment, "note": "final_carry"}
                    )

        if current_chapter_data is not None:
            corpus.append(current_chapter_data)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"✓ Parsed {len(corpus)} chapters")
    print(f"✓ Output: {out_path}")
    print(f"{'=' * 60}\n")

    for chapter in corpus:
        chap_id = chapter["chapter_id"]
        num_pages = len(chapter.get("pages", []))
        num_segments = sum(len(p["alignment"]) for p in chapter.get("pages", []))
        num_enums = len(chapter.get("enumerations", []))

        print(f"Chapter {chap_id}:")
        print(f"  - {num_pages} pages")
        print(f"  - {num_segments} alignment segments")
        if num_enums > 0:
            print(f"  - {num_enums} enumeration items")


if __name__ == "__main__":
    input_pdf = "data/2 SIDA mooré - français.pdf"
    out_path = "aligned_parsed.json"

    parse_pdf_to_json(input_pdf, out_path)
