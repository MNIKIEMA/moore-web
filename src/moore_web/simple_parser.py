import re
from dataclasses import dataclass
import pymupdf
from loguru import logger

SUB_SPLIT_RE = re.compile(r"\s+\d+\)\s+(?=Frn)")
MOORE_EXAMPLE_RE = re.compile(r"\{e\.g\.\s*(.*?)\}", flags=re.DOTALL)
EXTRA_FIELDS = [
    (
        "variant",
        r"var\.?:\s*(.*?)\s*(?=$|syn|Nominal|scient|Racine|catégorie|infinitif|Empr|sg)",
    ),
    (
        "synonym",
        r"syn\.?:\s*(.*?)\s*(?=$|var|Nominal|scient|Racine|catégorie|infinitif|Empr|sg)",
    ),
    (
        "nominal",
        r"Nominal\.?:\s*(.*?)\s*(?=$|syn|var|scient|Racine|catégorie|infinitif|Empr|sg)",
    ),
    (
        "scientific",
        r"scient\.?:\s*(.*?)\s*(?=$|syn|var|Nominal|Racine|catégorie|infinitif|Empr|sg)",
    ),
    (
        "racine",
        r"Racine\.?:\s*(.*?)\s*(?=$|syn|var|Nominal|scient|catégorie|infinitif|Empr|sg)",
    ),
    (
        "infinitif",
        r"infinitif\.?:\s*(.*?)\s*(?=$|syn|var|Nominal|scient|Racine|catégorie|Empr|sg)",
    ),
    (
        "empr",
        r"Empr\.?:\s*(.*?)\s*(?=$|syn|var|Nominal|scient|Racine|catégorie|infinitif|sg)",
    ),
    (
        "sg",
        r"sg\.?:\s*(.*?)\s*(?=$|syn|var|Nominal|scient|Racine|catégorie|infinitif|Empr)",
    ),
    ("category", r"\(catégorie\s*:\s*(.*?)\)"),
]

EXTRA_FIELDS_BETTER = [
    ("variant", r"var\.?:\s*(.*?)(?=\s*(?:syn|Nominal|scient|Racine|catégorie|\n|$))"),
    ("synonym", r"syn\.?:\s*(.*?)(?=\s*(?:var|Nominal|scient|Racine|catégorie|\n|$))"),
    ("nominal", r"Nominal\.?:\s*(.*?)(?=\s*(?:syn|var|scient|Racine|catégorie|\n|$))"),
    (
        "scientific",
        r"scient\.?:\s*(.*?)(?=\s*(?:syn|var|Nominal|Racine|catégorie|\n|$))",
    ),
    ("racine", r"Racine\.?:\s*(.*?)(?=\s*(?:syn|var|Nominal|scient|catégorie|\n|$))"),
    ("category", r"\(catégorie\s*:\s*(.*?)\)"),
]

GRAMMAR = r"""
    part\.gram|
    expr\.|
    indéf\.|
    num|
    n\.pl|
    interj|
    aux|
    \<Not Sure\>|
    n\.propre|
    postpos|
    Verbe|
    adj|
    n|
    v(?::Any)?|
    dém|
    Nom|
    inter\.|
    Adjectif|
    verbe\.it|
    v\.inacc|
    conj|
    pron|
    adv
"""
GRAMMAR_PATTERN = (
    r"part\.gram|expr\.|indéf\.|num|n\.pl|interj|aux|<Not Sure>|"
    r"n\.propre|postpos|Verbe|adj|n\b|v(?::Any)?|dém|Nom|inter\.|"
    r"Adjectif|verbe\.it|v\.inacc|conj|pron|adv"
)

ENTRY_START = rf"""
(?=
    ^\s*                       # must start at beginning of line
    \S+                         # entry token
    (?:\s+\[[^\]]+\])?          # optional tone
    \s+
    (?:{GRAMMAR})\b             # grammar
)
"""


entry_header = re.compile(
    rf"""
    (?P<header>
        ^\s*
        (?P<entry>\S+)                   # main entry token
        (?:\s+(?P<tone>\[[^\]]+\]))?     # optional tone
        (?:\s+|[\r\n]+)                  # space OR newline before grammar
        (?P<grammar>{GRAMMAR})\b         # grammar tag
    )
    """,
    re.MULTILINE | re.VERBOSE,
)


@dataclass
class Entry:
    french: str
    english: str
    moore: str


def split_sub_entries(entry_text):
    parts = re.split(SUB_SPLIT_RE, entry_text)
    return [p.strip() for p in parts if p.strip()]


def has_sub_entries(entry_text):
    return bool(SUB_SPLIT_RE.search(entry_text))


def split_french_english(text):
    blocks = re.findall(
        rf"Frn(.*?)Eng(.*?)(?=Frn|{ENTRY_START}|\Z)",
        text,
        flags=re.DOTALL | re.VERBOSE,
    )
    return [(fr.strip(), eng.strip()) for fr, eng in blocks]


def extract_moore_examples(eng_text):
    return (
        MOORE_EXAMPLE_RE.findall(eng_text)[-1]
        if MOORE_EXAMPLE_RE.search(eng_text)
        else None
    )


def clean_english_remove_examples(eng_text):
    return MOORE_EXAMPLE_RE.sub("", eng_text).strip()


def extract_extra_fields(text):
    info = {}
    for name, pattern in EXTRA_FIELDS:
        m = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            value = m.group(1).strip()
            if value:
                info[name] = value
    return info


def clean_text(t):
    t = re.sub(r"\b[A-Z]\s+[a-z]\b", "", t)
    t = re.sub(r"\n\s*\d+\s*\n", "\n", t)
    t = re.sub(r"\n{2,}", "\n", t)
    t = re.sub(r"^\s*-\s*$", "", t, flags=re.MULTILINE)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\b\d+\s+of\s+\d+\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\b\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}\b", "", t)
    t = re.sub(r"file:///\S+", "", t)
    t = re.sub(
        r"Dictionnaire\s+Mooré.*?français.*?English",
        "",
        t,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return t.strip()


def clean_english_text(eng_text):
    """
    Remove metadata patterns from English text
    """
    eng_text = MOORE_EXAMPLE_RE.sub("", eng_text)

    for _, pattern in EXTRA_FIELDS:
        eng_text = re.sub(pattern, "", eng_text, flags=re.DOTALL | re.IGNORECASE)

    eng_text = re.sub(r"\s+", " ", eng_text)

    return eng_text.strip()


def analyze_body(body):
    """
    Process a body block:
      - split sub-entries
      - split FR/ENG
      - if Moore example exists, extract FR/Eng parts separately
      - extract extras from the second FR/ENG tuple
    Returns list of senses.
    """
    senses = []
    blocks = split_sub_entries(body) if has_sub_entries(body) else [body]

    for block in blocks:
        fr_eng_pairs = split_french_english(block)
        print("*"*60 + "\n", fr_eng_pairs, len(fr_eng_pairs), "*"*60 + "\n", sep="\n")
        if not fr_eng_pairs:
            continue

        sense_data = []

        if len(fr_eng_pairs) == 2:
            fr_entry, eng_entry = fr_eng_pairs[0]
            fr_example, eng_rest = fr_eng_pairs[1]

            moore_example = extract_moore_examples(eng_entry)
            eng_entry_rest = clean_english_remove_examples(eng_entry)
            extras = extract_extra_fields(eng_rest)
            eng_entry_clean = clean_english_text(eng_entry_rest)

            sense_data.append(
                {
                    "fr_entry": fr_entry,
                    "eng_entry": eng_entry_clean,
                    "moore_example": moore_example,
                    "french_example": fr_example,
                    "english_example": clean_english_text(eng_rest),
                    **extras,
                }
            )
        else:
            for fr, eng in fr_eng_pairs:
                eng_clean = clean_english_remove_examples(eng)
                extras = extract_extra_fields(eng)
                eng_clean = clean_english_text(eng_clean)
                sense_data.append(
                    {
                        "fr_entry": fr,
                        "eng_entry": eng_clean,
                        "moore_example": None,
                        "french_example": None,
                        "english_example": None,
                        **extras,
                    }
                )
        senses.append(sense_data)

    return senses


def split_first_entry(text: str):
    """
    Split the text into two parts:
    1. Text before the first valid dictionary entry
    2. Text from the first valid dictionary entry onward

    Returns:
        [before, from_entry_onward] -> list of two strings
    """

    entry_start_re = re.compile(
        rf"""
        ^                                           # start of line
        ([^\s\n][^\n]*?)                           # token (any non-whitespace start)
        (?:\s+|\n\s*)                              # whitespace or newline+whitespace
        (?:\[[^\]]+\]\s+)?                         # optional tone in brackets
        ({GRAMMAR_PATTERN})                        # grammar tag
        \s+                                        # whitespace
        (?=Frn|\d+\))                              # lookahead for Frn or digit)
        """,
        re.MULTILINE | re.VERBOSE,
    )
    
    match = entry_start_re.search(text)
    if match:
        before = text[:match.start()].rstrip()
        from_entry = text[match.start():].lstrip()
        return [before, from_entry]
    else:
        return [text, ""]


def split_dictionary_entries(content) -> list[tuple[str, str, str, str]]:
    """
    Split dictionary content into individual entries.
    Each entry format:
        headword [tone]? grammar (number)? definition
    
    Returns:
        list of tuples: [(token, tone, grammar, rest), ...]
    """
    entries = []
    
    entry_start_pattern = rf'^([^\s\n][^\n]*?)\s+(?:\[([^\]]+)\]\s+)?({GRAMMAR_PATTERN})\s+(?=Frn|\d+\))'
    print(content)

    matches = list(re.finditer(entry_start_pattern, content, re.MULTILINE))
    
    if not matches:
        return []
    
    for i, match in enumerate(matches):
        token = match.group(1).strip()
        tone = match.group(2).strip() if match.group(2) else ""
        grammar = match.group(3).strip()

        start_of_rest = match.end()
        
        if i + 1 < len(matches):
            end_of_rest = matches[i + 1].start()
        else:
            end_of_rest = len(content)
        
        rest = content[start_of_rest:end_of_rest].strip()
        
        entries.append((token, tone, grammar, rest))
    
    return entries


def parse_page(page: str):
    page_entries = []
    unfinished_block, entries = split_first_entry(page)
    entries = split_dictionary_entries(entries)
    logger.warning(entries)
    for token, tone, grammar, rest in entries:
        senses = analyze_body(rest)
        page_entries.append(
            {
                "body": rest,
                "entry": token,
                "tone": tone,
                "grammar": grammar,
                "senses": senses,
            }
        )

    return page_entries, unfinished_block


def parse_doc(doc: pymupdf.Document):
    parsed_entries = []
    for i, page in enumerate(doc):
        if i <= 1:
            pass
        if i > 2:
            break
        blocks = page.get_text("blocks", sort = True)
        text = "\n".join([b[4] for b in blocks])

        text = clean_text(text)
        print("text \nxxxx", text, "clean text\nxxxx")
        entries, valid_start = parse_page(text)
        if not valid_start:
            parsed_entries.append(entries)
        else:
            if parsed_entries:
                parsed_entries.pop()
            index = max(0, i - 1)
            previous_page = doc[index]
            previous_text = previous_page.get_text("text")
            previous_text = clean_text(previous_text) + valid_start
            entries, valid_start = parse_page(previous_text)  # type: ignore
            parsed_entries.append(entries)
    return parsed_entries



if __name__ == "__main__":
    import json
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pdf", "-i", type=str, default="Dictionnaire-Moore-français-English-avec-images.pdf")
    parser.add_argument("--output_json", "-o", type=str, default="output.json")
    args = parser.parse_args()

    input_pdf = args.input_pdf
    output_json = args.output_json

    with pymupdf.open(input_pdf) as doc:
        res = parse_doc(doc)
        for e in res:
            print("-" * 60)
            print("header :", json.dumps(e, indent=2, ensure_ascii=False))