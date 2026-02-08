import pymupdf
from argparse import ArgumentParser
import json
import re


def split_dictionary_entries(text) -> list[str]:
    """
    Split dictionary text into individual lemma entries.
    """
    lines = text.split("\n")

    pos_pattern = r"(?:Verbe|Pronom|Nom|expression|interj|particule grammaticale|préposition|préfixe|Adverbe|auxiliaire|Adjectif|conjonction|indéfinie|démonstratif|v\.inaccompli|interrogatif|Déterminant)\.|(?:\s|^)[vn]\.(?=\s|$)"

    entries: list[str] = []
    current_entry_lines = []

    for i, line in enumerate(lines):
        if re.match(r"^[a-zA-Z0-9\-ã].+?\s{2,}", line) and not line.startswith(" "):
            if current_entry_lines:
                entry_text = "\n".join(current_entry_lines).strip()
                if re.search(pos_pattern, entry_text):
                    entries.append(entry_text)
            current_entry_lines = [line]
        else:
            current_entry_lines.append(line)
    if current_entry_lines:
        entry_text = "\n".join(current_entry_lines).strip()
        if re.search(pos_pattern, entry_text):
            entries.append(entry_text)

    return entries


def split_at_pos(entry: str) -> tuple[str, str]:
    """
    Split a dictionary entry at the end of the POS tag.

    Returns the metadata (lemma + IPA + variants + POS) and the definition part.

    Args:
        entry (str): A single dictionary entry

    Returns:
        tuple: (metadata_part, definition_part)

    Example:
        >>> entry = "a1 [à] Varinat: yẽ. Pronom.\\n1 • il, elle"
        >>> meta, defn = split_at_pos(entry)
        >>> "Pronom." in meta
        True
    """
    pos_tags = r"(?:Verbe|Pronom|Nom|expression|interj|particule grammaticale|préposition|préfixe|Adverbe|auxiliaire|Adjectif|conjonction|indéfinie|démonstratif|v\.inaccompli|interrogatif|Déterminant|v|n)\."
    m = re.search(pos_tags, entry)

    if m:
        split_pos = m.end()
        metadata = entry[:split_pos].strip()
        definition = entry[split_pos:].strip()
        return metadata, definition

    return entry.strip(), ""


def has_multiple_senses(entry: str) -> bool:
    """
    Check if a dictionary entry has multiple numbered senses.

    Senses are marked with numbers followed by bullet: "1 •", "2 •", etc.

    Args:
        entry (str): A single dictionary entry

    Returns:
        bool: True if entry has 2 or more senses, False otherwise

    Example:
        >>> entry = "a1 [à] Pronom.\\n1 • first sense\\n2 • second sense"
        >>> has_multiple_senses(entry)
        True
    """
    sense_pattern = r"(?:^|\s)\d+\s*•"

    senses = re.findall(sense_pattern, entry, flags=re.MULTILINE)

    return len(senses) >= 2


def remove_duplicate_lemma_lines(entry):
    """
    Remove duplicate lemma lines at the start of an entry.

    Some entries have format:
    lemma1
    lemma2
    lemma1
    [IPA] ...

    This keeps only the lemma line that has IPA/variants/POS.

    Args:
        entry (str): A single dictionary entry

    Returns:
        str: Entry with duplicate lemma lines removed
    """
    lines = entry.strip().split("\n")

    if len(lines) < 3:
        return entry

    first_line = lines[0].strip()
    second_line = lines[1].strip()

    has_metadata = r"\[.*\]|(?:Plural|singulier|Varinat|Comparez|Inaccompli|nominal|racine|emprunt):|(?:Verbe|Pronom|Nom|expression|interj|particule grammaticale|préposition|préfixe|Adverbe|auxiliaire|Adjectif|conjonction|indéfinie|démonstratif|v\.inaccompli|interrogatif|Déterminant|v|n)\."

    if not re.search(has_metadata, first_line) and not re.search(has_metadata, second_line):
        return "\n".join(lines[2:]).strip()

    return entry


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


def split_senses(entry):
    sense_pattern = r"((?:^|\s)\d+\s*•)"

    parts = re.split(sense_pattern, entry)

    metadata = parts[0].strip()
    it = iter(parts[1:])
    senses = [f"{m.strip()} {t.strip()}" for m, t in zip(it, it)]

    return metadata, senses


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


def parse_complex_definition(text: str):
    gloss, remaining = text.split(".", 1)

    french_gloss, english_gloss = gloss.split(";", 1)
    if not remaining:
        return {"french": french_gloss, "english": english_gloss}
    cat_sci_pattern = r"Category:\s*([^.]+?)(?:\.\s*([A-Z][a-z]+ [a-z]+))?\."
    cat_sci_match = re.search(cat_sci_pattern, remaining)
    category = None
    scientific_name = None

    if cat_sci_match:
        category = cat_sci_match.group(1).strip()
        scientific_name = cat_sci_match.group(2)
    syn_match = re.search(r"synonyme:\s*([^.]+)\.", remaining)
    ant_match = re.search(r"antonyme:\s*([^.]+)\.", remaining)

    clean_text = re.sub(r"Category:.*?\.(?:\s*[A-Z][a-z]+ [a-z]+\.)?", "", remaining, flags=re.DOTALL)
    clean_text = re.sub(r"(synonyme|antonyme):.*?\.", "", clean_text, flags=re.DOTALL)

    definition_part = ""
    examples = []

    if clean_text:
        definition_part = clean_text.strip()
        examples = extract_examples(definition_part)

    return {
        "french": french_gloss,
        "english": english_gloss,
        "definition": definition_part,
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
    if num_segments >= 3 and num_segments % 3 == 0:
        per_lang = num_segments // 3

        moore = " ".join(segments[0:per_lang])
        french = " ".join(segments[per_lang : per_lang * 2])
        english = " ".join(segments[per_lang * 2 : per_lang * 3])

        results.append({"moore": moore, "french": french, "english": english})

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
    all_parsed = []
    num_columns = 2
    for i, page in enumerate(doc, start=1):
        text = merge_page(page, num_columns)
        if i == 1:
            _, text = re.split(r"A\s*-\s*a", text)
        page_number = extract_page_numbers(text)
        text = remove_date(text)
        text = remove_page_number(text)

        entries = split_dictionary_entries(text)
        for entry in entries:
            try:
                metadata, definition = split_at_pos(entry)
            except Exception as e:
                print(e)
                continue
            has_sense = has_multiple_senses(entry)
            if has_sense:
                metadata, definition = split_senses(entry)
            metadata_components = extract_metadata_components(metadata)
            if isinstance(definition, list):
                for d in definition:
                    def_struct = parse_complex_definition(d)
                    print("Definition structure:", def_struct)
            else:
                def_struct = parse_complex_definition(definition)

            all_parsed.append(
                {
                    "page_number": page_number,
                    "metadata": metadata_components,
                    "definition": def_struct,
                }
            )

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
