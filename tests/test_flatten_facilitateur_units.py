import pytest

from moore_web.book_parser_facilitateur import Book, Chapter, Section, Subsection
from moore_web.flatten import flatten_facilitateur_pair_per_unit


def _book(number, title, sections):
    return Book(chapters=[Chapter(number=number, title=title, sections=sections)])


def test_sections_are_matched_by_title_role_not_position():
    fr_book = _book(
        1,
        "Les secrets de Kadé",
        [
            Section(title="Questions à discuter", body="Question française."),
            Section(title="L'histoire de Kadé", body="Histoire française."),
        ],
    )
    mo_book = _book(
        1,
        "M ma solga yɛla",
        [
            Section(title="Kibarã", body="Kibar sẽn ya moore."),
            Section(title="Sõaseg sokdse", body="Sokr sẽn ya moore."),
        ],
    )

    units = dict(flatten_facilitateur_pair_per_unit(fr_book, mo_book, segment=False))

    assert list(units) == ["kade-ch1-title", "kade-ch1-story", "kade-ch1-questions"]
    assert "Histoire française." in units["kade-ch1-story"].french
    assert "Kibar sẽn ya moore." in units["kade-ch1-story"].moore
    assert "Question française." in units["kade-ch1-questions"].french
    assert "Sokr sẽn ya moore." in units["kade-ch1-questions"].moore


def test_repeated_preface_sections_and_subsections_are_folded_by_role():
    fr_manual = Section(
        title="Comment utiliser ce manuel",
        body="Mode d'emploi.",
        subsections=[Subsection(title="L'histoire", body="Lire l'histoire.")],
    )
    mo_manual = Section(
        title="Sẽn n kẽed ne seb kãngã",
        body="Sebrã kibare.",
        subsections=[Subsection(title="Karem-y kibarã", body="Karem-y kibara.")],
    )
    fr_book = _book(
        0,
        "Manuel",
        [
            Section(title="Quel est le problème ?", body="Le problème."),
            Section(
                title="Comment l'église pourrait-elle répondre à ce problème ?",
                body="La réponse.",
            ),
            Section(title="À propos de ce manuel", body="À propos."),
            fr_manual,
        ],
    )
    mo_book = _book(
        0,
        "Sebre",
        [
            Section(title="Yellã yaa bʋgo ?", body="Yellã la leokre."),
            mo_manual,
        ],
    )

    units = dict(flatten_facilitateur_pair_per_unit(fr_book, mo_book, segment=False))

    assert list(units) == ["kade-ch0-title", "kade-ch0-context", "kade-ch0-manual"]
    assert "Le problème." in units["kade-ch0-context"].french
    assert "La réponse." in units["kade-ch0-context"].french
    assert "Yellã la leokre." in units["kade-ch0-context"].moore
    assert "L'histoire" in units["kade-ch0-manual"].french
    assert "Karem-y kibarã" in units["kade-ch0-manual"].moore
    assert "Lire l'histoire." in units["kade-ch0-manual"].french
    assert "Karem-y kibara." in units["kade-ch0-manual"].moore


def test_structure_mismatch_is_not_silently_exported():
    fr_book = _book(
        2,
        "Chapitre deux",
        [Section(title="Questions à discuter", body="Une question.")],
    )
    mo_book = _book(
        2,
        "Sak a yiib soaba",
        [Section(title="Kibarã", body="Kibare.")],
    )

    with pytest.raises(ValueError, match="section roles differ in chapter 2"):
        flatten_facilitateur_pair_per_unit(fr_book, mo_book, segment=False)


def test_unknown_section_title_has_actionable_error():
    fr_book = _book(3, "Chapitre trois", [Section(title="Unexpected", body="Texte.")])
    mo_book = _book(3, "Sak a tãab soaba", [Section(title="Kibarã", body="Kibare.")])

    with pytest.raises(ValueError, match="Unknown Kadé section title in chapter 3: 'Unexpected'"):
        flatten_facilitateur_pair_per_unit(fr_book, mo_book, segment=False)


@pytest.mark.parametrize("source", ["facilitateur", "kade"])
def test_export_cli_registers_paired_facilitateur_sources(source):
    import runpy
    from pathlib import Path

    script = Path(__file__).parents[1] / "scripts" / "export_review_units.py"
    namespace = runpy.run_path(str(script))

    args = namespace["build_parser"]().parse_args(
        ["--source", source, "--fr-input", "fr.txt", "--mo-input", "mo.txt", "-o", "units.jsonl"]
    )

    assert args.source == source
    assert source in namespace["PAIR_EXPORTERS"]
