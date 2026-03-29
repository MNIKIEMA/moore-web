import re
import pymupdf
from loguru import logger

from moore_web.models import DictionaryEntry, Example, Sense
from moore_web.pdf_extractor import _expand_ligatures


MOORE_EXAMPLE_RE = re.compile(r"\{e\.g\.\s*(.*?)\}", flags=re.DOTALL)
# Detects: "Frn <def> {e.g. <moore>} Frn <fr_ex>" — no Eng between the two Frn blocks
EMBEDDED_EG_FRN_RE = re.compile(r"^(.*?)\{e\.g\.\s*(.*?)\}\s*Frn\s*(.*)", flags=re.DOTALL)
EXTRA_FIELDS = [
    (
        "variant",
        r"var\.?:\s*(.*?)\s*(?=$|syn|Nominal|scient|Racine|catégorie|infinitif|Empr|sg|Inaccompli)",
    ),
    (
        "synonym",
        r"syn\.?:\s*(.*?)\s*(?=$|var|Nominal|scient|Racine|catégorie|infinitif|Empr|sg|Inaccompli)",
    ),
    (
        "nominal",
        r"Nominal\.?:\s*(.*?)\s*(?=$|syn|var|scient|Racine|catégorie|infinitif|Empr|sg|Inaccompli)",
    ),
    (
        "scientific",
        r"scient\.?:\s*(.*?)\s*(?=$|syn|var|Nominal|Racine|catégorie|infinitif|Empr|sg|Inaccompli)",
    ),
    (
        "racine",
        r"Racine\.?:\s*(.*?)\s*(?=$|syn|var|Nominal|scient|catégorie|infinitif|Empr|sg|Inaccompli)",
    ),
    (
        "infinitif",
        r"infinitif\.?:\s*(.*?)\s*(?=$|syn|var|Nominal|scient|Racine|catégorie|Empr|sg|Inaccompli)",
    ),
    (
        "empr",
        r"Empr\.?:\s*(.*?)\s*(?=$|syn|var|Nominal|scient|Racine|catégorie|infinitif|sg|Inaccompli)",
    ),
    (
        "sg",
        r"sg\.?:\s*(.*?)\s*(?=$|syn|var|Nominal|scient|Racine|catégorie|infinitif|Empr|Inaccompli)",
    ),
    (
        "inaccompli",
        r"Inaccompli\.?:\s*(.*?)\s*(?=$|syn|var|Nominal|scient|Racine|catégorie|infinitif|Empr|sg)",
    ),
    (
        "antonym",
        r"ant\.?:\s*(.*?)\s*(?=$|syn|var|Nominal|scient|Racine|catégorie|infinitif|Empr|sg|Inaccompli)",
    ),
    ("category", r"\(catégorie\s*:\s*(.*?)\.\)"),
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
    ("category", r"\(catégorie\s*:\s*(.*?)\.\)"),
]


GRAMMAR_PATTERN = (
    r"part\.gram|expr\.|indéf\.|num|n\.pl|interj|aux|<Not Sure>|"
    r"n\.propre|postpos|Verbe|adj|n\b|v(?::Any)?|dém|Nom|inter\.|"
    r"Adjectif|verbe\.it|v\.inacc|conj|pron|adv"
)

SUB_SPLIT_RE = re.compile(rf"\s+\d+\)\s+(?:(?:{GRAMMAR_PATTERN})\s+)?(?=Frn)")
# Same pattern with a capturing group so split_sub_entries can recover the grammar tag
_SUB_SPLIT_CAP_RE = re.compile(rf"\s+\d+\)\s+(?:({GRAMMAR_PATTERN})\s+)?(?=Frn)")

# ---------------------------------------------------------------------------
# S6 artifact patch — temporary until SUB_SPLIT_RE handles all "N) grammar Frn"
# sub-entry splits correctly.
#
# Three failure modes leak through as bogus headwords:
#   1. "poorẽ1)"   — a real lemma with a trailing sub-entry index ("1)") bled in
#                    because split_sub_entries couldn't cut before it.
#   2. "Frnnous Engwe" — a French/English definition block that wasn't split off
#                    and was mistakenly captured as the next headword.
#   3. "about what?" — punctuation-heavy fragment from an unsplit definition line.
#
# These are dropped / cleaned here as a best-effort workaround.
# TODO (S6): Remove this patch once split_sub_entries reliably handles every
#            "N) <grammar> Frn" variant so none of these fragments reach parse_page.
# ---------------------------------------------------------------------------
_S6_TRAILING_SUBENTRY_RE = re.compile(r"\d+\)$")
_INLINE_TONE_RE = re.compile(r"^(.*?)\s*\[([^\]]+)\]$")


def _s6_clean_token(token: str) -> str:
    """Strip a trailing sub-entry index (e.g. '1)') bled into the lemma."""
    return _S6_TRAILING_SUBENTRY_RE.sub("", token).strip()


def _extract_inline_tone(token: str, tone: str) -> tuple[str, str]:
    """If the tone was merged into the token (e.g. 'gẽ[é]'), split it out."""
    if not tone:
        m = _INLINE_TONE_RE.match(token)
        if m:
            return m.group(1).strip(), m.group(2)
    return token, tone


def _s6_is_garbage(token: str) -> bool:
    """Return True for tokens that are parsing artifacts, not valid Moore lemmas.

    Covers the known S6 failure modes; other cases may still slip through until
    the root cause in SUB_SPLIT_RE / split_sub_entries is fully resolved.
    """
    # "Frnnous Engwe" — Frn…Eng definition text mistaken for a headword
    if token.startswith("Frn"):
        return True
    # "about what?" — interrogative fragment that cannot be a Moore headword
    if "?" in token:
        return True
    return False


ENTRY_START = rf"""
(?=
    ^\s*                       # must start at beginning of line
    \S+                         # entry token
    (?:\s+\[[^\]]+\])?          # optional tone
    \s+
    (?:{GRAMMAR_PATTERN})\b             # grammar
)
"""


_VARIANT_EXTRAS = frozenset({"variant", "nominal", "racine", "infinitif", "empr", "sg", "inaccompli"})
_UNSPEC_SENSE_RE = re.compile(r"^\(unspec\.\s+var\.\s+of\s+(\S+)\)$")


def _make_entry(d: dict) -> DictionaryEntry:
    """Convert an internal parse dict to a DictionaryEntry struct."""
    senses: list[Sense] = []
    variants: dict[str, list[str]] = {}
    sense_id = 1

    is_unspec_entry = False

    for block in d["senses"]:
        for s in block:
            # Detect synthetic unspec. var. stub — record as variant, skip as sense
            unspec_match = _UNSPEC_SENSE_RE.match(s.get("fr_entry", ""))
            if unspec_match:
                is_unspec_entry = True
                variants.setdefault("variant", []).append(unspec_match.group(1))
                continue

            for key in _VARIANT_EXTRAS:
                if key in s:
                    variants[key] = [v.strip() for v in str(s[key]).split(",")]

            examples: list[Example] = []
            moore_ex = s.get("moore_example")
            fr_ex = s.get("french_example")
            if moore_ex or fr_ex:
                examples.append(
                    Example(
                        moore=moore_ex or "",
                        french=fr_ex or "",
                        english=s.get("english_example") or None,
                    )
                )
            for extra in s.get("extra_examples", []):
                examples.append(
                    Example(
                        moore=extra.get("moore", ""),
                        french=extra.get("french", ""),
                        english=extra.get("english") or None,
                    )
                )

            senses.append(
                Sense(
                    id=str(sense_id),
                    french=s.get("fr_entry", ""),
                    english=s.get("eng_entry", ""),
                    examples=examples,
                    category=s.get("category"),
                    scientific_name=s.get("scientific"),
                    synonym=s.get("synonym"),
                    antonym=s.get("antonym"),
                    pos=s.get("grammar") or d["grammar"],
                )
            )
            sense_id += 1

    return DictionaryEntry(
        lemma=d["entry"],
        ipa=d["tone"],
        pos="" if is_unspec_entry else d["grammar"],
        senses=senses,
        variants=variants if variants else None,
    )


def split_sub_entries(entry_text: str) -> list[tuple[str | None, str]]:
    """Split on sub-entry delimiters, returning (grammar_tag | None, block) pairs.
    The first block always has grammar=None (its POS comes from the entry level)."""
    parts = re.split(_SUB_SPLIT_CAP_RE, entry_text)
    # re.split with one capturing group: [block0, cap1, block1, cap2, block2, ...]
    result = [(None, parts[0].strip())]
    for i in range(1, len(parts), 2):
        grammar = parts[i]  # None when the optional grammar group didn't match
        block = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if block:
            result.append((grammar, block))
    return result


def has_sub_entries(entry_text):
    return bool(SUB_SPLIT_RE.search(entry_text))


def split_french_english(text):
    blocks = re.findall(
        rf"Frn(.*?)Eng(.*?)(?=Frn|{ENTRY_START}|\Z)",
        text,
        flags=re.DOTALL | re.VERBOSE,
    )
    return [(_normalize(fr), _normalize(eng)) for fr, eng in blocks]


def extract_moore_examples(eng_text):
    return _normalize(MOORE_EXAMPLE_RE.findall(eng_text)[-1]) if MOORE_EXAMPLE_RE.search(eng_text) else None


def clean_english_remove_examples(eng_text):
    return MOORE_EXAMPLE_RE.sub("", eng_text).strip()


# TODO: _normalize is redundant — flatten_simple_parser._clean() already collapses newlines
#       before writing output. Consider removing _normalize and its call sites once
#       DictionaryEntry / Sense are consumed exclusively through the flatten layer.
def _normalize(text: str) -> str:
    """Collapse newlines and surrounding whitespace into a single space."""
    return re.sub(r"\s*\n\s*", " ", text).strip()


def extract_extra_fields(text):
    info = {}
    category_pattern = r"\(catégorie\s*:\s*(.*?)\.\)[.,\s]*"
    category_match = re.search(category_pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if category_match:
        info["category"] = _normalize(category_match.group(1))
        text = re.sub(category_pattern, "", text, flags=re.DOTALL | re.IGNORECASE)
    for name, pattern in EXTRA_FIELDS:
        if name == "category":
            continue
        m = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            value = _normalize(m.group(1))
            if value:
                info[name] = value
    return info


_UNSPEC_VAR_RE = re.compile(
    r"^([^\s\n][^\n]*?)\s+(?:\[([^\]]+)\]\s+)?unspec\.\s+var\.\s+of\s+(\S[^\n]*?)$",
    re.MULTILINE,
)


def _normalise_unspec_var(t: str) -> str:
    """Rewrite 'lemma [tone] unspec. var. of X' stubs into a synthetic
    grammar-bearing line so split_dictionary_entries can cut there."""
    # Cat C: "lemma\n[tone] unspec." → join onto one line
    t = re.sub(r"(\S)\n(\[[^\]]+\]\s+unspec\.)", r"\1 \2", t)
    # Cat D: "lemma\nunspec. var." → join onto one line
    t = re.sub(r"(\S)\n(unspec\.)", r"\1 \2", t)
    # Cat D: strip nested "(unspec. var. of X)" inside the target
    t = re.sub(r"\s*\(unspec\.[^)]+\)", "", t)

    def _replace(m: re.Match) -> str:
        lemma = m.group(1).strip()
        tone = f"[{m.group(2)}] " if m.group(2) else ""
        target = m.group(3).strip()
        return f"{lemma} {tone}n Frn (unspec. var. of {target}) Eng"

    return _UNSPEC_VAR_RE.sub(_replace, t)


def clean_text(t):
    t = _expand_ligatures(t)
    t = re.sub(r"\b[A-Z]\s+[a-z]\b", "", t)
    t = re.sub(r"\n\s*\d+\s*\n", "\n", t)
    t = re.sub(r"\n{2,}", "\n", t)
    t = re.sub(r"^\s*-\s*$", "", t, flags=re.MULTILINE)
    t = re.sub(r"(\w)- +(\w)", r"\1-\2", t)
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
    t = _normalise_unspec_var(t)
    return t.strip()


def clean_english_text(eng_text):
    """
    Remove metadata patterns from English text
    """
    eng_text = MOORE_EXAMPLE_RE.sub("", eng_text)

    eng_text = re.sub(r"\(catégorie\s*:.*?\.\)[.,\s]*", "", eng_text, flags=re.IGNORECASE | re.DOTALL)

    for name, pattern in EXTRA_FIELDS:
        if name != "category":
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
    blocks = split_sub_entries(body) if has_sub_entries(body) else [(None, body)]

    for sub_grammar, block in blocks:
        fr_eng_pairs = split_french_english(block)
        if not fr_eng_pairs:
            continue

        sense_data = []

        if len(fr_eng_pairs) >= 2:
            fr_entry, eng_entry = fr_eng_pairs[0]

            # Pair 0's fr may contain an embedded "{e.g. <moore>} Frn <fr_ex>" with no Eng
            embedded0 = EMBEDDED_EG_FRN_RE.match(fr_entry)
            if embedded0:
                fr_entry = embedded0.group(1).strip()
                embedded_moore = embedded0.group(2).strip()
                embedded_fr_ex = embedded0.group(3).strip()
                embedded_en_ex = clean_english_text(eng_entry)
                extra_from_embedded = [
                    {"moore": embedded_moore, "french": embedded_fr_ex, "english": embedded_en_ex}
                ]
                eng_entry = ""
            else:
                extra_from_embedded = []

            eng_entry_clean = clean_english_text(clean_english_remove_examples(eng_entry))
            extras = extract_extra_fields(fr_eng_pairs[-1][1])

            # Chain: pair[j-1]'s {e.g.} is the Moore example; pair[j] provides its fr/en
            examples = list(extra_from_embedded)
            for j in range(1, len(fr_eng_pairs)):
                m_ex = extract_moore_examples(fr_eng_pairs[j - 1][1])
                fr_ex, en_ex = fr_eng_pairs[j]
                en_ex_clean = clean_english_text(en_ex)
                if m_ex or fr_ex or en_ex_clean:
                    examples.append({"moore": m_ex or "", "french": fr_ex, "english": en_ex_clean})

            first_ex = examples[0] if examples else {}
            sense_data.append(
                {
                    "fr_entry": fr_entry,
                    "eng_entry": eng_entry_clean,
                    "moore_example": first_ex.get("moore"),
                    "french_example": first_ex.get("french"),
                    "english_example": first_ex.get("english"),
                    "extra_examples": examples[1:],
                    "grammar": sub_grammar,
                    **extras,
                }
            )
        else:
            fr, eng = fr_eng_pairs[0]
            embedded = EMBEDDED_EG_FRN_RE.match(fr)
            if embedded:
                fr_def = embedded.group(1).strip()
                moore_ex = embedded.group(2).strip()
                fr_ex = embedded.group(3).strip()
                en_ex = clean_english_text(eng)
                extras = extract_extra_fields(en_ex)
                sense_data.append(
                    {
                        "fr_entry": fr_def,
                        "eng_entry": "",
                        "moore_example": moore_ex,
                        "french_example": fr_ex,
                        "english_example": en_ex,
                        "grammar": sub_grammar,
                        **extras,
                    }
                )
            else:
                moore_example = extract_moore_examples(eng)
                eng_clean = clean_english_text(clean_english_remove_examples(eng))
                extras = extract_extra_fields(eng)
                sense_data.append(
                    {
                        "fr_entry": fr,
                        "eng_entry": eng_clean,
                        "moore_example": moore_example,
                        "french_example": None,
                        "english_example": None,
                        "grammar": sub_grammar,
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
        (?!\d+\)\s)                                # not a bare sub-entry number e.g. "2) "
        ([^\s\n](?:(?![,\.]\s)[^\n])*?)            # token (no sentence-style ", " or ". " inside)
        (?:\s+|\n\s*)                              # whitespace or newline+whitespace
        (?:\[[^\]]+\]\s+)?                         # optional tone in brackets
        (?:\([^)]+\)\s+)?                          # optional parenthetical e.g. (unspec. var. X)
        (?:\d+\)\s+)?                              # optional sub-entry number before grammar
        ({GRAMMAR_PATTERN})                        # grammar tag
        \s+                                        # whitespace
        (?=Frn|\d+\))                              # lookahead for Frn or digit)
        """,
        re.MULTILINE | re.VERBOSE,
    )

    match = entry_start_re.search(text)
    if match:
        before = text[: match.start()].rstrip()
        from_entry = text[match.start() :].lstrip()
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

    entry_start_pattern = rf"^(?!\d+\)\s)([^\s\n](?:(?![,\.]\s)[^\n])*?)\s+(?:\[([^\]]+)\]\s+)?(?:\([^)]+\)\s+)?(?:\d+\)\s+)?({GRAMMAR_PATTERN})\s+(?=Frn|\d+\))"

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


def strip_trailing_entry_name(body: str, entry: str) -> str:
    """
    Remove a repeated entry name appearing as the final line of the body.
    Empty lines and internal formatting are preserved exactly.
    """
    lines = body.splitlines()

    if not lines:
        return body

    if lines[-1].strip() == entry.strip():
        lines.pop()
        return "\n".join(lines)

    return body


def parse_page(page: str) -> tuple[list[dict], str]:
    """Return (internal_entry_dicts, spillover_text)."""
    page_entries: list[dict] = []
    unfinished_block, entries = split_first_entry(page)
    entries = split_dictionary_entries(entries)
    for token, tone, grammar, rest in entries:
        token = _s6_clean_token(token)
        token, tone = _extract_inline_tone(token, tone)
        if _s6_is_garbage(token):
            logger.warning("S6 artifact dropped: {!r}", token)
            continue
        cleaned_body = strip_trailing_entry_name(rest, token)
        senses = analyze_body(cleaned_body)
        page_entries.append(
            {
                "body": cleaned_body,
                "entry": token,
                "tone": tone,
                "grammar": grammar,
                "senses": senses,
            }
        )

    return page_entries, unfinished_block


def parse_doc(doc: pymupdf.Document) -> list[DictionaryEntry]:
    parsed_entries: list[list[dict]] = []
    for i, page in enumerate(doc, start=1):  # type: ignore
        blocks = page.get_text("blocks", sort=True)
        text = "\n".join([b[4] for b in blocks])

        text = clean_text(text)
        entries, valid_start = parse_page(text)
        if not valid_start:
            parsed_entries.append(entries)
            continue
        if not parsed_entries:
            parsed_entries.append(entries)
            continue
        last_page_entries = parsed_entries[-1]
        if not last_page_entries:
            parsed_entries.append(entries)
            continue
        last_entry = last_page_entries.pop()

        # FIXME: Page with image has the entry name repeated at the end
        merged_body = last_entry["body"] + valid_start
        last_entry["body"] = merged_body
        last_entry["senses"] = analyze_body(merged_body)
        last_page_entries.append(last_entry)
        last_page_entries.extend(entries)

    return [_make_entry(d) for page_entries in parsed_entries for d in page_entries]


if __name__ == "__main__":
    import argparse
    import msgspec.json

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_pdf",
        "-i",
        type=str,
        default="data/Dictionnaire-Moore-français-English-avec-images.pdf",
    )
    parser.add_argument("--output_jsonl", "-o", type=str, default="output.jsonl")
    args = parser.parse_args()

    with pymupdf.open(args.input_pdf) as doc:
        res = parse_doc(doc)

    with open(args.output_jsonl, "wb") as f:
        for entry in res:
            f.write(msgspec.json.encode(entry) + b"\n")
