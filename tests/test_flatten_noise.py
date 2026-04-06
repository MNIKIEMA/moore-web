"""Tests for noise-filtering logic in moore_web.flatten.

Covers:
- _PAGE_REF_RE  : strip inline page / author-page references
- _STANDALONE_NUM_RE : reject segments that are only numbers / ranges
- _URL_RE / _COPYRIGHT_RE : reject bibliography and credits segments
- flatten_facilitateur_pair : titles must also be cleaned (regression tests
  for the bug where titles bypassed _PAGE_REF_RE.sub)
"""

import pytest

from moore_web.flatten import (
    _COPYRIGHT_RE,
    _PAGE_REF_RE,
    _STANDALONE_NUM_RE,
    _URL_RE,
)


# ---------------------------------------------------------------------------
# _PAGE_REF_RE — should MATCH (to be stripped)
# ---------------------------------------------------------------------------


class TestPageRefRe:
    @pytest.mark.parametrize(
        "text",
        [
            "(p. 3)",
            "(p.3)",
            "(p.17)",
            "(p. 25)",
            "(p.70)",
            "(Yamamori p.70)",
            "(Yamamori p. 70)",
            "(Smith p. 10)",
        ],
    )
    def test_matches(self, text):
        assert _PAGE_REF_RE.search(text), f"expected match for {text!r}"

    @pytest.mark.parametrize(
        "text",
        [
            "Les secrets de Maman (p. 3)",
            "A Pok zaka rãmb tara zu-loeese (p.17)",
            "M ma solga yɛla (p.3)",
            "A Pok ne a yaopa tõog n gesa b mens yelle (p.25)",
        ],
    )
    def test_ref_found_in_title(self, text):
        assert _PAGE_REF_RE.search(text), f"ref not found in {text!r}"

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("Les secrets de Maman (p. 3)", "Les secrets de Maman "),
            ("A Pok zaka rãmb tara zu-loeese (p.17)", "A Pok zaka rãmb tara zu-loeese "),
            ("M ma solga yɛla (p.3)", "M ma solga yɛla "),
            ("(Yamamori p.70) some text", " some text"),
        ],
    )
    def test_substitution_removes_ref(self, text, expected):
        assert _PAGE_REF_RE.sub("", text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "Il est malade.",
            "smartphones (3 modèles)",  # number inside parens but no 'p.'
            "(top.70)",  # 'p' not at word boundary
        ],
    )
    def test_no_match(self, text):
        assert not _PAGE_REF_RE.search(text), f"unexpected match for {text!r}"


# ---------------------------------------------------------------------------
# _STANDALONE_NUM_RE — should MATCH (to be dropped)
# ---------------------------------------------------------------------------


class TestStandaloneNumRe:
    @pytest.mark.parametrize(
        "text",
        [
            "44",
            "14-24",
            "14–24",
            "27,",
            "2.",
            "27, 2.",
            "1, 2, 3",
            "1",
            "0",
        ],
    )
    def test_matches(self, text):
        assert _STANDALONE_NUM_RE.match(text), f"expected match for {text!r}"

    @pytest.mark.parametrize(
        "text",
        [
            "verse 14-24 says",
            "chapitre 3 de la Bible",
            "Il y a 3 raisons.",
            "Karm-y Zak sebrã 1 :27 ne a 2 :14-24 1.",
        ],
    )
    def test_no_match(self, text):
        assert not _STANDALONE_NUM_RE.match(text), f"unexpected match for {text!r}"


# ---------------------------------------------------------------------------
# _URL_RE — should MATCH (to be dropped)
# ---------------------------------------------------------------------------


class TestUrlRe:
    @pytest.mark.parametrize(
        "text",
        [
            "http://www.example.org",
            "https://unaids.org/en/",
            "www.truelovewaits.org.za",
            "<www.shellbook.com>",
            "see http://foo.bar for details",
        ],
    )
    def test_matches(self, text):
        assert _URL_RE.search(text), f"expected match for {text!r}"

    @pytest.mark.parametrize(
        "text",
        [
            "Il est parti.",
            "A yɛɛ wʋsgo.",
        ],
    )
    def test_no_match(self, text):
        assert not _URL_RE.search(text), f"unexpected match for {text!r}"


# ---------------------------------------------------------------------------
# _COPYRIGHT_RE — should MATCH (to be dropped)
# ---------------------------------------------------------------------------


class TestCopyrightRe:
    def test_matches_copyright_symbol(self):
        assert _COPYRIGHT_RE.search("© Shellbook Publishing 2004")

    def test_no_match_normal_sentence(self):
        assert not _COPYRIGHT_RE.search("Il est parti.")


# ---------------------------------------------------------------------------
# flatten_facilitateur_pair — titles must also be cleaned
# ---------------------------------------------------------------------------


def _make_book(chapters):
    """Build a minimal Book from a list of (chapter_title, [(sec_title, body)]) tuples."""
    from moore_web.book_parser_facilitateur import Book, Chapter, Section

    book = Book()
    for i, (ch_title, sections) in enumerate(chapters, start=1):
        ch = Chapter(number=i, title=ch_title)
        for sec_title, body in sections:
            sec = Section(title=sec_title, body=body)
            ch.sections.append(sec)
        book.chapters.append(ch)
    return book


class TestFlattenFacilitateurPairTitles:
    """Regression: titles with page refs must have the ref stripped."""

    def test_chapter_title_page_ref_stripped(self):
        from moore_web.flatten import flatten_facilitateur_pair

        fr_book = _make_book([("Les secrets de Maman (p. 3)", [])])
        mo_book = _make_book([("M ma solga yɛla (p.3)", [])])

        result = flatten_facilitateur_pair(fr_book, mo_book, segment=False)

        assert all("(p." not in s for s in result.french), (
            f"Page ref not stripped from french titles: {result.french}"
        )
        assert all("(p." not in s for s in result.moore), (
            f"Page ref not stripped from moore titles: {result.moore}"
        )

    def test_section_title_page_ref_stripped(self):
        from moore_web.flatten import flatten_facilitateur_pair

        fr_book = _make_book([("Chapitre 1", [("A Pok zaka rãmb tara zu-loeese (p.17)", "Some body text.")])])
        mo_book = _make_book(
            [("Chapitre 1", [("A Pok ne a yaopa tõog n gesa b mens yelle (p.25)", "Yɛɛ wʋsgo.")])]
        )

        result = flatten_facilitateur_pair(fr_book, mo_book, segment=False)

        assert all("(p." not in s for s in result.french), (
            f"Page ref not stripped from french section titles: {result.french}"
        )
        assert all("(p." not in s for s in result.moore), (
            f"Page ref not stripped from moore section titles: {result.moore}"
        )

    def test_author_page_ref_stripped(self):
        from moore_web.flatten import flatten_facilitateur_pair

        fr_book = _make_book([("Titre (Yamamori p.70)", [])])
        mo_book = _make_book([("Yɛɛ wʋsgo.", [])])

        result = flatten_facilitateur_pair(fr_book, mo_book, segment=False)

        assert all("Yamamori" not in s for s in result.french), (
            f"Author-page ref not stripped: {result.french}"
        )
