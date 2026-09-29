import pytest

from moore_web.corrections import apply_corrections, load_corrections


def _tsv(tmp_path, lines):
    p = tmp_path / "c.tsv"
    p.write_text(
        "id\tcolumn\twrong\tright\tnote\n" + "".join(f"{line}\n" for line in lines), encoding="utf-8"
    )
    return p


def test_applies_in_the_given_column_and_counts(tmp_path):
    rows = [
        {"id": "a", "moore": "nin-zaals vššm yellẽ", "french": "vššm"},
        {"id": "b", "moore": "x", "french": "y"},
    ]
    corr = load_corrections(_tsv(tmp_path, ["a\tmoore\tvššm\tvɩɩm\tš for ɩ"]))
    assert apply_corrections(rows, corr) == 1
    assert rows[0] == {"id": "a", "moore": "nin-zaals vɩɩm yellẽ", "french": "vššm"}


def test_fails_when_text_or_id_is_missing(tmp_path):
    with pytest.raises(ValueError, match="not in"):
        apply_corrections(
            [{"id": "a", "moore": "fixed already"}],
            load_corrections(_tsv(tmp_path, ["a\tmoore\tvššm\tvɩɩm\t"])),
        )
    with pytest.raises(ValueError, match="ids not in the dataset"):
        apply_corrections(
            [{"id": "a", "moore": "x"}], load_corrections(_tsv(tmp_path, ["gone\tmoore\tx\ty\t"]))
        )


def test_rejects_bad_column_and_no_op(tmp_path):
    with pytest.raises(ValueError, match="column"):
        load_corrections(_tsv(tmp_path, ["a\tenglish\tx\ty\t"]))
    with pytest.raises(ValueError, match="no-op"):
        load_corrections(_tsv(tmp_path, ["a\tmoore\tx\tx\t"]))


def test_repository_corrections_file_loads():
    from pathlib import Path

    corr = load_corrections(Path(__file__).resolve().parents[1] / "corrections" / "moore.tsv")
    assert sum(len(v) for v in corr.values()) == 10
