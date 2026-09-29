from moore_web.hplt_mono import (
    clean_sentences,
    has_foreign_script,
    host_matches,
    normalize,
    sentence_id,
    split_document,
    distinct_word_ratio,
    tone_marked_words,
)


def _lang(texts):
    # Stand-in for GlotLID: English if it contains "the", otherwise Mooré.
    langs = ["eng_Latn" if " the " in f" {t.lower()} " else "mos_Latn" for t in texts]
    return langs, [0.95] * len(texts)


def test_host_matches_subdomains_only():
    assert host_matches("https://incubator.wikimedia.org/wiki/Wp/mos/X", ["incubator.wikimedia.org"])
    assert host_matches("https://mos.wikipedia.org/wiki/Y", ["wikipedia.org"])
    assert not host_matches("https://www.jw.org/mos/", ["wikipedia.org"])
    assert not host_matches("https://notwikipedia.org/", ["wikipedia.org"])


def test_split_document_keeps_line_index():
    doc = {
        "id": "d1",
        "u": "https://mos.wikipedia.org/wiki/A",
        "text": "A ye.[1] A yiibu. 3] [a]\n\n8] A tãabo[ 12].[DM 1]",
    }
    rows = split_document(doc)
    assert [(r["line"], r["text"]) for r in rows] == [(0, "A ye."), (0, "A yiibu."), (2, "A tãabo.")]


def test_clean_sentences_filters_dedups_and_excludes():
    docs = [
        {
            "id": "w",
            "u": "https://mos.wikipedia.org/wiki/A",
            "text": (
                "Turkmen haly yaa buud a ye sẽn yaa ne nug tʋʋma.\n"
                "Retrieved from the web on March 28.\n"
                "B sɩnga.\n"
                "Turkmen haly yaa buud a ye sẽn yaa ne nug tʋʋma!\n"
                "Sõng-kãnga sẽn be sõng-kãrã pʋgẽ wã.\n"
                "Wẽnnaam naana tẽngã ne saasã fãa."
            ),
        },
        {"id": "j", "u": "https://www.jw.org/mos/x", "text": "Wẽnnaam naana tẽngã bõe yĩnga, a Zeova?"},
        {
            "id": "m",
            "u": "https://incubator.wikimedia.org/wiki/Wp/mos/B",
            "text": ("Karen-biisã zãmsda b karen-saamb yɩɩlã wakat fãa. mos_Latnmos_Latn be be be."),
        },
    ]
    rows, stats = clean_sentences(docs, _lang, exclude_texts=["Sõng-kãnga sẽn be sõng-kãrã pʋgẽ wã"])
    assert [r["text"] for r in rows] == [
        "Turkmen haly yaa buud a ye sẽn yaa ne nug tʋʋma.",
        "Wẽnnaam naana tẽngã ne saasã fãa.",
    ]
    assert (stats.documents, stats.mt_tagged_documents) == (1, 1)
    assert list(stats.steps.values()) == [6, 5, 5, 5, 5, 5, 4, 4, 3, 2]
    assert rows[0]["doc_id"] == "w" and rows[0]["words"] == 11 and "lang" not in rows[0]
    assert list(rows[0])[:4] == ["id", "text", "source", "license"]
    assert rows[0]["id"] == sentence_id(rows[0]["text"])
    assert (rows[0]["source"], rows[0]["license"]) == ("wikipedia", "CC-BY-SA-4.0")


def test_sentence_id_depends_only_on_normalized_content():
    a = sentence_id("Turkmen haly yaa buud a ye.")
    assert a == sentence_id("  turkmen HALY, yaa buud a ye! ")
    assert a != sentence_id("Turkmen haly yaa buud a yiibu.")
    assert a.startswith("hplt-") and len(a) == len("hplt-") + 16


def test_normalize():
    assert normalize("  Turkmen HALY, yaa! ") == normalize("turkmen haly yaa")


def test_foreign_script_and_ipa_but_not_moore_letters():
    assert not has_foreign_script("Sõng-kãnga sẽn be sõng-kãrã pʋgẽ wã, yɛl wʋsg, bɩ ɔ.")
    assert not has_foreign_script("A Rövşən ne a Kérékou.")
    assert has_foreign_script("Yeondeunghoe (Korean: 연등회) yaa kibay.")
    assert has_foreign_script("Castel (Catalan: [kəsˈteʎ]) yaa bur.")
    assert has_foreign_script("Kaлyшapи bɩ Pycaлии.")


def test_tone_marked_words_count_lowercase_words_only():
    assert tone_marked_words("A Pedro Martínez ne a Román Díaz sẽn yiis album.") == 0
    assert tone_marked_words("À galóùnting (vil) kãsngã lã Wùgdgù") == 1
    assert tone_marked_words("A goùnbga noûg goabgã windg sì lœtin nē Burkina Faso.") == 4


def test_distinct_word_ratio_flags_loops_only_in_longer_sentences():
    loop = "B sẽn yaa b sẽn yaa b to wã yaa b sẽn tar b sẽn yaa b to."
    assert distinct_word_ratio(loop) < 0.55
    assert distinct_word_ratio("Turkmen haly yaa buud a ye sẽn yaa ne nug tʋʋma.") >= 0.55
    assert distinct_word_ratio("B yãk n yãk b.") == 1.0  # under 8 words: not judged
