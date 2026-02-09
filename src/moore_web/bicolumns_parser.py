import pymupdf
from argparse import ArgumentParser
import json
import re
from loguru import logger


def parse_dictionary_entries(entries: list[str]) -> list[dict]:
    """
    Parses a list of split strings into structured dictionary data.
    Expected format: [Lemma, POS, Text+Lemma, POS, Text+Lemma...]
    """
    if not entries:
        return []

    pos_tags = entries[1::2]
    text_blocks = entries[2::2]
    current_lemma = entries[0].strip()

    lemma_match = re.match(r"^(.*?)\s*\[([^\]]+)\]", current_lemma)
    if lemma_match:
        lemma_name = lemma_match.group(1).strip()
        ipa = lemma_match.group(2).strip()
    else:
        lemma_name = re.split(r"\s+", current_lemma)[0].strip()
        ipa = ""

    all_data = []

    for pos, text in zip(pos_tags, text_blocks):
        current_pos = pos.strip()
        current_content = text.strip()

        dots = current_content.count(".")
        if dots > 1:
            current_definition, raw_next_lemma = split_entry(current_content)
        elif dots == 1:
            parts = current_content.rsplit(".", 1)
            current_definition = parts[0].strip()
            raw_next_lemma = parts[1].strip() if len(parts) > 1 else ""
        else:
            current_definition = current_content
            raw_next_lemma = ""

        if has_multiple_senses(current_definition):
            senses = split_senses(current_definition)
        else:
            senses = [("1", current_definition)]

        all_data.append(
            {
                "lemma": lemma_name,
                "ipa": ipa,
                "pos": current_pos,
                "definition": current_definition,
                "senses": senses,
            }
        )

        if raw_next_lemma:
            next_lemma_match = re.match(r"^(.*?)\s*\[([^\]]+)\]", raw_next_lemma)
            if next_lemma_match:
                next_lemma_name = next_lemma_match.group(1).strip()
                next_ipa = next_lemma_match.group(2).strip()
            else:
                next_lemma_name = re.split(r"\s+", raw_next_lemma)[0].strip()
                next_ipa = ""
            if re.match(r"^\d+$", next_lemma_name):
                pass
            else:
                lemma_name = next_lemma_name
                ipa = next_ipa

    return all_data


def has_multiple_senses(entry: str) -> bool:
    sense_pattern = r"\b\d+\s*•"
    senses = re.findall(sense_pattern, entry)
    return len(senses) >= 2


def remove_section_header(text: str) -> str:
    text = re.sub(r"^\s*[A-Za-zÀ-ÿ]\s*-\s*[A-Za-zÀ-ÿ]\s*$", "", text, flags=re.MULTILINE)
    return text


def remove_page_number(text: str) -> str:
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    return text


def remove_date(text: str) -> str:
    text = re.sub(r"\b\d{2}/\d{2}/\d{4}\b", "", text)
    return text


def extract_page_numbers(text):
    """
    Extract page numbers and their positions from the text.

    Args:
        text (str): Dictionary text

    Returns:
        list: List of tuples (page_number, position_in_text)
    """
    page_numbers = []

    pattern = r"^\s*\d{2}/\d{2}/\d{4}\s*$\s*^\s*(\d+)\s*$"

    for match in re.finditer(pattern, text, flags=re.MULTILINE):
        page_num = int(match.group(1))
        position = match.start()
        page_numbers.append((page_num, position))

    return page_numbers


def split_senses(entry: str) -> list[tuple[str, str]]:
    sense_pattern = r"\b(\d+)\s*•"
    matches = list(re.finditer(sense_pattern, entry))

    if not matches:
        return []

    results: list[tuple[str, str]] = []
    for i in range(len(matches)):
        sense_num = matches[i].group(1)
        start_index = matches[i].end()
        if i + 1 < len(matches):
            end_index = matches[i + 1].start()
        else:
            end_index = len(entry)
        first, sep, _ = entry[start_index:end_index].strip().rpartition(".")
        sense_text = first + sep
        results.append((sense_num, sense_text))
    return results


def extract_metadata_components(metadata_str):
    pos_tags = r"(Verbe|Pronom|Nom|expression|interj|particule grammaticale|préposition|préfixe|Adverbe|auxiliaire|Adjectif|conjonction|indéfinie|démonstratif|v\.inaccompli|interrogatif|Déterminant|v|n)"
    pos_match = re.search(rf"{pos_tags}\.", metadata_str)
    pos = pos_match.group(0) if pos_match else None
    pre_pos = metadata_str
    if pos_match:
        pre_pos = metadata_str[: pos_match.start()].strip()

    if "[" in pre_pos:
        lemma = pre_pos.split("[")[0].strip()
    else:
        parts = re.split(r"(?:Varinat|Comparez):", pre_pos)
        lemma = parts[0].strip()

    variant_match = re.search(r"(?:Varinat|Comparez):\s*([^.]+)", metadata_str)
    variants = variant_match.group(1).strip() if variant_match else None

    return {"lemma": lemma, "variants": variants, "pos": pos}


def split_entry(text: str) -> list[str]:
    """
    Split definition from lemma+variants.
    Find the last ". " before a lemma (not before field keywords)
    """
    dot_positions = [m.start() for m in re.finditer(r"\.\s+", text)]

    if not dot_positions:
        return [text, ""]
    field_keywords = (
        r"^(Comparez|Varinat|Variant|racine|emprunt|Plural|infinitif|Inaccompli|nominal|Category):"
    )

    for pos in reversed(dot_positions):
        after_dot = text[pos + 2 :].lstrip()
        if after_dot and not re.match(field_keywords, after_dot, re.IGNORECASE):
            definition = text[: pos + 1].strip()
            lemma_part = text[pos + 1 :].strip()
            return [definition, lemma_part]
    last_dot = dot_positions[-1]
    return [text[: last_dot + 1].strip(), text[last_dot + 1 :].strip()]


def parse_complex_definition(text: str):
    pattern = r"^(?P<french>[^;]+;)\s*(?P<english>[^.]+?\.)\s*(?P<remaining>.*)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        french_gloss = match.group("french").strip()
        english_gloss = match.group("english").strip()
        remaining = match.group("remaining").strip()
    else:
        logger.warning(f"Failed to match pattern: {text}\n English gloss will be empty")
        gloss, remaining = text.split(".", 1)
        french_gloss = gloss.strip()
        english_gloss = ""
    if not remaining:
        return {"french": french_gloss, "english": english_gloss}
    cat_sci_pattern = r"Category:\s*([^.]+?)\.\s*(?:([A-Za-z][a-z]+(?:\s+[a-z]+){1,2}))?"
    cat_sci_match = re.search(cat_sci_pattern, remaining)
    category = None
    scientific_name = None
    print("\nRemaining", remaining, "\n\n")
    if cat_sci_match:
        category = cat_sci_match.group(1).strip()
        scientific_name = cat_sci_match.group(2)
        remaining = remaining.replace(cat_sci_match.group(0), "")
    syn_match = re.search(r"synonyme:\s*([^.]+)\.", remaining)
    ant_match = re.search(r"antonyme:\s*([^.]+)\.", remaining)

    clean_text = re.sub(r"(synonyme|antonyme):.*?\.", "", remaining, flags=re.DOTALL)

    definition_part = ""
    examples = []

    if clean_text:
        definition_part = clean_text.strip()
        examples = extract_examples(definition_part)

    return {
        "french": french_gloss,
        "english": english_gloss,
        "examples": examples,
        "category": category,
        "scientific_name": scientific_name,
        "synonym": syn_match.group(1).strip() if syn_match else None,
        "antonym": ant_match.group(1).strip() if ant_match else None,
    }


def extract_examples(text: str):
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("- ", "-")

    segments = re.findall(r"[^.?!]+[.?!](?:\s*»\s*\.\s*\.)?", text)
    segments = [s.strip() for s in segments]

    num_segments = len(segments)
    results = []
    if num_segments >= 2:
        if num_segments % 3 == 0:
            per_lang = num_segments // 3
            moore = " ".join(segments[0:per_lang])
            french = " ".join(segments[per_lang : per_lang * 2])
            english = " ".join(segments[per_lang * 2 : per_lang * 3])
        elif num_segments == 5:
            per_lang = num_segments // 3
            moore = " ".join(segments[0:per_lang])
            french = " ".join(segments[per_lang : per_lang * 2 + 1])
            english = " ".join(segments[per_lang * 2 + 1 : per_lang * 3 + 2])
        elif num_segments % 2 == 0:
            per_lang = num_segments // 2
            moore = " ".join(segments[0:per_lang])
            french = " ".join(segments[per_lang : per_lang * 2])
            english = None
        else:
            logger.warning(f"Failed to parse example: {text}")
            return []
        results.append({"moore": moore, "french": french, "english": english})
    else:
        logger.warning(f"Failed to parse example: {text} ({segments} segments)")
        return []

    return results


def merge_page(page: pymupdf.Page, num_columns: int = 2) -> str:
    blocks = page.get_text("blocks", sort=True)
    page_width = page.rect.width
    column_width = page_width / 2
    columns = [[] for _ in range(num_columns)]

    for b in blocks:
        x0, y0, x1, y1, text = b[:5]
        text = text.strip()
        if not text or text.isdigit():
            continue
        col_idx = int(x0 / column_width)
        col_idx = min(col_idx, num_columns - 1)
        columns[col_idx].append(text)

    return "\n".join("\n".join(col) for col in columns)


def parse_doc(doc: pymupdf.Document):
    # TODO: page 509: Take into account some variation like postposition.
    all_parsed = []
    pos_pattern = r"((?:Verbe|Pronom|Nom|expression|interj|particule grammaticale|préfixe|Adverbe|auxiliaire|Adjectif|conjonction|indéfinie|démonstratif|v\.inaccompli|interrogatif|Déterminant)\.)"
    num_columns = 2
    for i, page in enumerate(doc, start=1):
        text = merge_page(page, num_columns)
        if i == 1:
            _, text = re.split(r"A\s*-\s*a", text)
        else:
            text = "\n".join(text.split("\n")[2:])
        page_number = extract_page_numbers(text)
        print(page_number)
        text = remove_date(text)
        text = remove_page_number(text)
        text = re.sub(r"\n+", " ", text)
        entries = re.split(pos_pattern, text)
        parsed_entries = parse_dictionary_entries(entries)
        all_parsed.extend(parsed_entries)
    # TODO: parse lemma to get variant and IPA
    # TODO: parse sense to get gloss and examples,category, scientific name, synonyms and antonyms
    return all_parsed


def main():
    parser = ArgumentParser(description="Parse dictionary entries from a PDF file.")
    parser.add_argument("--input-path", "-i", type=str, required=True, help="Path to the PDF file")
    parser.add_argument("--output-path", "-o", type=str, required=True, help="Path to the output JSONL file")
    args = parser.parse_args()

    pdf_path = args.input_path
    out_path = args.output_path
    doc = pymupdf.open(pdf_path)
    parsed = parse_doc(doc)

    with open(out_path, "w", encoding="utf-8") as f:
        for p in parsed:
            f.write(json.dumps(p) + "\n")


if __name__ == "__main__":
    main()
