"""Unicode fixes for the text of parallel pairs, applied to every source at build time.

These are data errors, not style: characters that look like Mooré letters but
are other code points (typed with the wrong keyboard, or produced by a font or
OCR), and accents stored as separate combining marks. They are wrong for any
use of the data, and a model sees them as different tokens (it learns that ɩ
can also be written with a Greek ι, and produces it).

Mooré side (`normalize_moore`), measured on moore-web-parallel v1.0.0:

    ι Greek iota      -> ɩ    606 rows (mafand)
    ɡ script g        -> g    481 rows (du-moore)
    ʊ upsilon         -> ʋ     36 rows (udhr)
    ɭ l with retroflex hook -> Ɩ (capital ɩ)   21 rows (all-caps conseils headings)
    ε Greek epsilon   -> ɛ      6 rows (mafand)
    U+0342 Greek perispomeni used as a tilde -> U+0303 combining tilde
    ā ē ī ō ū -> ã ẽ ĩ õ ũ   212 rows (nasal vowels typed with a macron: expert, kade)
    ǎ ă -> ã, ª -> ẽ, ƴ -> y, ṅ -> n   ~15 rows (conseils font encoding, typos)
    then NFC: decomposed accents (3,677 rows, mostly mafand) become one letter

French side (`normalize_french`): only the letter-shape fixes (ʊ ɡ ɭ, the
perispomeni) for Mooré names in French text ("Tindaoɡo", 5 rows), then NFC.
Replacements confirmed with the user (2026-09-29).

Ambiguous characters (ə, 6 rows; œ in udhr "pœga"; š, 3 rows) are left for
manual review: `scripts/check_orthography.py` lists them. Typography (’ « ») is kept;
the NLLB-specific mapping lives in mt-training.
"""

from __future__ import annotations

import unicodedata
from collections import Counter

# Look-alikes of Mooré letters and other unambiguous substitutions.
MOORE_LOOKALIKES = {
    "ι": "ɩ",  # GREEK SMALL LETTER IOTA
    "ε": "ɛ",  # GREEK SMALL LETTER EPSILON
    "ʊ": "ʋ",  # LATIN SMALL LETTER UPSILON
    "Ʊ": "Ʋ",  # LATIN CAPITAL LETTER UPSILON
    "ɡ": "g",  # LATIN SMALL LETTER SCRIPT G
    "ɭ": "Ɩ",  # capital ɩ in all-caps conseils headings ("Tɭ" = "TƖ")
    "\u0342": "\u0303",  # COMBINING GREEK PERISPOMENI -> COMBINING TILDE
    # Nasal vowels typed with a macron (expert translations, kade): "tēedame" = tẽedame
    "ā": "ã",
    "ē": "ẽ",
    "ī": "ĩ",
    "ō": "õ",
    "ū": "ũ",
    "Ā": "Ã",
    "Ē": "Ẽ",
    "Ī": "Ĩ",
    "Ō": "Õ",
    "Ū": "Ũ",
    # Nasal a with a caron or breve ("ǎbriyolozi", "zenralãă")
    "ǎ": "ã",
    "ă": "ã",
    "Ǎ": "Ã",
    "Ă": "Ã",
    "ª": "ẽ",  # conseils font encoding: "sªn" = sẽn
    "ƴ": "y",
    "Ƴ": "Y",  # hooked y: "ƴʋʋm" = yʋʋm
    "ṅ": "n",
    "Ṅ": "N",  # "sẽṅ" = sẽn
}
_MOORE_TABLE = str.maketrans(MOORE_LOOKALIKES)
# French text quotes Mooré names ("Tindaoɡo"): only the letter-shape fixes; Greek
# letters, macrons, carons and hooked letters may be real in foreign names.
FRENCH_LOOKALIKES = {k: MOORE_LOOKALIKES[k] for k in ("ʊ", "Ʊ", "ɡ", "ɭ", "\u0342")}
_FRENCH_TABLE = str.maketrans(FRENCH_LOOKALIKES)

# Letters a Mooré sentence may use besides Latin letters with an ASCII base
# carrying an allowed mark (a, é, ẽ, …): the Mooré-specific letters.
MOORE_LETTERS = set("ɛɔɩʋƐƆƖƲ")
# French: ligatures, plus Mooré letters in quoted Mooré names.
FRENCH_LETTERS = MOORE_LETTERS | set("œŒæÆ")


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def normalize_moore(text: str) -> str:
    # NFC first so decomposed forms (a + U+0304) become the precomposed letters the
    # table maps (ā), then NFC again to compose the replacements (o + U+0303 -> õ).
    return _nfc(_nfc(text).translate(_MOORE_TABLE))


def normalize_french(text: str) -> str:
    return _nfc(_nfc(text).translate(_FRENCH_TABLE))


def lookalikes_in(text: str) -> set[str]:
    """Known look-alike characters still present (should be empty after normalize_moore)."""
    return {c for c in text if c in MOORE_LOOKALIKES}


# Marks Mooré spelling uses (tilde) or keeps in French names (acute, grave,
# circumflex, diaeresis, cedilla). Anything else (macron, caron, breve, dot…)
# on a Latin letter is flagged.
ALLOWED_MARKS = {"\u0303", "\u0301", "\u0300", "\u0302", "\u0308", "\u0327"}


def unusual_letters(
    text: str, allowed: set[str] = MOORE_LETTERS, allowed_marks: set[str] | None = ALLOWED_MARKS
) -> set[str]:
    """Letters that are not Latin (ASCII or `allowed` base, with `allowed_marks`) nor in `allowed`.

    `allowed_marks=None` accepts any mark on a Latin letter.
    """
    found = set()
    for char in set(text):
        if not char.isalpha() or char in allowed:
            continue
        base, *marks = unicodedata.normalize("NFD", char)
        latin = unicodedata.name(char, "").startswith("LATIN") and (base.isascii() or base in allowed)
        if latin and (allowed_marks is None or all(m in allowed_marks for m in marks)):
            continue
        found.add(char)
    return found


def scan(
    texts: list[str], allowed: set[str] = MOORE_LETTERS, allowed_marks: set[str] | None = ALLOWED_MARKS
) -> tuple[Counter, int]:
    """(rows per unusual letter, rows not in NFC form)."""
    letters: Counter = Counter()
    not_nfc = 0
    for text in texts:
        letters.update(unusual_letters(text, allowed, allowed_marks))
        not_nfc += unicodedata.normalize("NFC", text) != text
    return letters, not_nfc
