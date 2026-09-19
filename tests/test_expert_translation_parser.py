"""Checks for the expert translation PDF's wrapped IDs and QA columns."""

import pytest

from moore_web.expert_translation_parser import HEADERS, parse_table_rows


def test_table_row_keeps_pair_and_review_fields() -> None:
    row = [
        "manually_crea\nted_cmn_Hans\n_253",
        "instruction-\nresponse",
        "A short\ncontext",
        "Le vieil\nhomme.",
        "Nin-kẽem\nkʋdrã.",
        None,
        "reviewed",
        "dots instead of\ncommas",
    ]

    result = parse_table_rows([list(HEADERS), row], page=1, doc_id="batch2", seen_ids=set())

    assert len(result) == 1
    record = result[0]
    assert record.id == "manually_created_cmn_Hans_253"
    assert (record.src_lang, record.tgt_lang) == ("fra", "mos")
    assert (record.source_text, record.target_text) == (
        "Le vieil homme.",
        "Nin-kẽem kʋdrã.",
    )
    assert record.domain == "instruction-response"
    assert record.context == "A short context"
    assert record.qa_check == "reviewed"
    assert record.qa_comments == "dots instead of commas"
    assert record.comments is None
    assert record.page == 1


def test_missing_translation_is_rejected() -> None:
    row = ["id-1", "dialogue", None, "Bonjour", "", None, "ok", None]

    with pytest.raises(ValueError, match="missing ID, source, or translation"):
        parse_table_rows([row], page=2, doc_id="batch2", seen_ids=set())


def test_duplicate_id_across_pages_is_rejected() -> None:
    row = ["id-1", "dialogue", None, "Bonjour", "Ne y yibeoogo", None, "ok", None]
    seen_ids: set[str] = set()
    parse_table_rows([row], page=1, doc_id="batch2", seen_ids=seen_ids)

    with pytest.raises(ValueError, match="duplicate ID"):
        parse_table_rows([row], page=2, doc_id="batch2", seen_ids=seen_ids)
