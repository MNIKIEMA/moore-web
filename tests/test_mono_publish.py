import pytest
import yaml

from moore_web.mono_publish import dataset_card, group_by_source, write_folder


def _row(i, source="wikipedia", license="CC-BY-SA-4.0"):
    return {
        "id": f"hplt-{i:016x}",
        "text": f"Sẽn yaa {i}.",
        "source": source,
        "license": license,
        "doc_id": "d",
        "url": "https://mos.wikipedia.org/wiki/A",
        "line": i,
        "lang_prob": 1.0,
        "words": 3,
    }


def test_card_has_default_and_per_source_configs():
    card = dataset_card(group_by_source([_row(1), _row(2)]))
    meta = yaml.safe_load(card.split("---")[1])
    assert meta["license"] == "cc-by-sa-4.0"
    assert [(c["config_name"], c["data_files"][0]["path"]) for c in meta["configs"]] == [
        ("default", "data/*/*.parquet"),
        ("wikipedia", "data/wikipedia/*.parquet"),
    ]
    assert "| `wikipedia` | Mooré Wikipedia | 2 |" in card


def test_rejects_unknown_source_wrong_license_and_duplicate_ids():
    with pytest.raises(ValueError, match="Unknown source"):
        group_by_source([_row(1, source="jw")])
    with pytest.raises(ValueError, match="license"):
        group_by_source([_row(1, license="CC-BY-4.0")])
    with pytest.raises(ValueError, match="Duplicate"):
        group_by_source([_row(1), _row(1)])


def test_write_folder_layout(tmp_path):
    from datasets import load_dataset

    write_folder(group_by_source([_row(1), _row(2)]), tmp_path)
    assert (tmp_path / "README.md").exists()
    ds = load_dataset("parquet", data_files=str(tmp_path / "data/wikipedia/train.parquet"), split="train")
    assert ds.column_names[:4] == ["id", "text", "source", "license"] and len(ds) == 2
