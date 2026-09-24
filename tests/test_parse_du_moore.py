from moore_web.parse_du_moore import (
    Lesson,
    _merge_wrapped,
    lesson_pairs,
    lesson_review_units,
    pair_tagged_pages,
)


def _lesson(**kw) -> Lesson:
    defaults = dict(
        book=2,
        number=30,
        key=("Alain a de jolies ceintures.", "A Alẽ tara sẽbds neeba."),
        vocab=({1: "les ceintures", 2: "le sein"}, {1: "sẽbdsã", 2: "paɡ bĩisri"}),
        sentences=([], []),
        passage=([], []),
    )
    return Lesson(**{**defaults, **kw})


def test_pages_pair_on_lesson_number_despite_header_typos():
    tagged = [
        ("kaoreng a tãab (3) soaba", ["fr3"]),
        ("Kaoreng a tãab (3) soaba", ["mo3"]),
        ("Kaoreng pisi a la ye (21) soaba", ["fr21"]),
        ("Kaoreng pisi la a ye (21) soaba", ["mo21"]),
    ]
    assert pair_tagged_pages(tagged) == [(3, ["fr3"], ["mo3"]), (21, ["fr21"], ["mo21"])]


def test_unpaired_page_is_skipped():
    tagged = [
        ("Kaoreng a yiib (2) soaba", ["x"]),
        ("Kaoreng a naas (4) soaba", ["fr"]),
        ("Kaoreng a naas (4) soaba", ["mo"]),
    ]
    assert pair_tagged_pages(tagged) == [(4, ["fr"], ["mo"])]


def test_merge_wrapped_joins_wrapped_lines_and_splits_packed_ones():
    chunks = [
        "Ce matin le teinturier partait au marigot",
        "pour sa teinture.",
        "Par finir Wango a aimé le théâtre. Selon lui, c’était beau.",
    ]
    assert _merge_wrapped(chunks) == [
        "Ce matin le teinturier partait au marigot pour sa teinture.",
        "Par finir Wango a aimé le théâtre.",
        "Selon lui, c’était beau.",
    ]


def test_pairs_skip_uneven_sentences_and_never_emit_passages():
    lesson = _lesson(sentences=(["a.", "b."], ["x.", "y.", "z."]), passage=(["p."], ["q."]))
    sections = {r["section"] for r in lesson_pairs(lesson)}
    assert sections == {"key", "vocab"}


def test_review_units_keep_everything_and_line_up_vocab():
    lesson = _lesson(
        vocab=(
            {1: "les ceintures", 8: "une fleur", 2: "le sein"},
            {2: "paɡ bĩisri", 1: "sẽbdsã", 14: "yiibu"},
        ),
        sentences=(["a.", "b."], ["x.", "y.", "z."]),
    )
    units = {uid: (fra, mos) for uid, fra, mos in lesson_review_units(lesson)}
    assert list(units) == ["du-moore-30-key", "du-moore-30-vocab", "du-moore-30-sentences"]
    assert units["du-moore-30-vocab"] == (
        ["les ceintures", "le sein", "une fleur"],
        ["sẽbdsã", "paɡ bĩisri", "yiibu"],
    )
    assert units["du-moore-30-sentences"] == (["a.", "b."], ["x.", "y.", "z."])
