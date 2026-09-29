import unicodedata

from moore_web.orthography import lookalikes_in, normalize_french, normalize_moore, scan, unusual_letters


def test_lookalikes_become_moore_letters():
    assert (
        normalize_moore("vιtamin, Eropeyεn, tʊʊmẽ, ɡes-y, BÕN-VɭɭLɭ")
        == "vɩtamin, Eropeyɛn, tʋʋmẽ, ges-y, BÕN-VƖƖLƖ"
    )
    assert not lookalikes_in(normalize_moore("vιtamin ɡes"))


def test_decomposed_and_perispomeni_tildes_become_one_letter():
    decomposed = "sẽn Inyo͂"
    fixed = normalize_moore(decomposed)
    assert fixed == "sẽn Inyõ" and unicodedata.is_normalized("NFC", fixed)


def test_moore_text_already_correct_is_unchanged():
    text = "Tõnd na n kẽnga yiri, t'a sẽn yeel « ayo » ɛ ɔ ɩ ʋ Ɩ Ʋ."
    assert normalize_moore(text) == text


def test_french_gets_nfc_and_non_greek_lookalikes():
    assert normalize_french("e\u0301te\u0301 Tindaoɡo ι") == "été Tindaogo ι"


def test_unusual_letters_and_scan():
    assert unusual_letters("Brigaad Laaɓal, tɩ ɛ ẽ é") == {"ɓ"}
    letters, not_nfc = scan(["vιtamin", "sẽn", "tɩ"])
    assert letters == {"ι": 1} and not_nfc == 1


def test_macron_caron_and_encoding_errors_become_moore_letters():
    assert normalize_moore("tēedame Gāand Yūngo yīnga tōndã ĒE") == "tẽedame Gãand Yũngo yĩnga tõndã ẼE"
    assert (
        normalize_moore("ǎbriyolozi zenralãă sªn ƴʋʋm Ƴinivɛrsite sẽṅ")
        == "ãbriyolozi zenralãã sẽn yʋʋm Yinivɛrsite sẽn"
    )


def test_french_keeps_foreign_accents():
    assert normalize_french("Tōkyō, Škoda, Hausa ƴ") == "Tōkyō, Škoda, Hausa ƴ"


def test_scanner_flags_marks_moore_does_not_use():
    assert unusual_letters("tēedame") == {"ē"}
    assert unusual_letters("sẽn été naïf ça") == set()
    assert unusual_letters("Tōkyō", allowed_marks=None) == set()


def test_decomposed_macron_is_mapped_too():
    assert normalize_moore("ga\u0304and si\u0304") == "gãand sĩ"
