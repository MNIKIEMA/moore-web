"""Tests for moore_web.annotate — IO helpers and annotation functions."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from datasets import Dataset

from moore_web.annotate import (
    _hf_repo,
    _is_hf,
    annotate,
    load_data,
    run_comet_qe,
    run_lang_id,
    run_laser,
    run_quality_warnings,
    save_data,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_ROWS = [
    {"french": "Bonjour tout le monde", "moore": "Yibeogo ne paam"},
    {"french": "La santé est importante", "moore": "Laafɩ yaa sõama"},
]


@pytest.fixture()
def small_dataset() -> Dataset:
    return Dataset.from_list(_ROWS)


@pytest.fixture()
def jsonl_file(tmp_path: Path) -> Path:
    p = tmp_path / "data.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in _ROWS), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# HF URI helpers
# ---------------------------------------------------------------------------


class TestHfHelpers:
    def test_is_hf_true(self):
        assert _is_hf("hf://owner/repo") is True

    def test_is_hf_false_local(self):
        assert _is_hf("/path/to/file.jsonl") is False

    def test_is_hf_false_relative(self):
        assert _is_hf("data.jsonl") is False

    def test_hf_repo_strips_prefix(self):
        assert _hf_repo("hf://owner/repo") == "owner/repo"


# ---------------------------------------------------------------------------
# load_data
# ---------------------------------------------------------------------------


class TestLoadData:
    def test_local_jsonl_row_count(self, jsonl_file: Path):
        ds = load_data(str(jsonl_file))
        assert len(ds) == 2

    def test_local_jsonl_columns(self, jsonl_file: Path):
        ds = load_data(str(jsonl_file))
        assert "french" in ds.column_names
        assert "moore" in ds.column_names

    def test_local_jsonl_values(self, jsonl_file: Path):
        ds = load_data(str(jsonl_file))
        assert ds["french"][0] == _ROWS[0]["french"]

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_data(str(tmp_path / "nonexistent.jsonl"))

    def test_hf_uri_dispatches_to_hub(self):
        """Verify hf:// URIs reach load_dataset — the DatasetNotFoundError proves it."""
        from datasets.exceptions import DatasetNotFoundError

        with pytest.raises(DatasetNotFoundError):
            load_data("hf://this-owner-does-not-exist/this-repo-does-not-exist")


# ---------------------------------------------------------------------------
# save_data
# ---------------------------------------------------------------------------


class TestSaveData:
    def test_local_jsonl_creates_file(self, small_dataset: Dataset, tmp_path: Path):
        out = tmp_path / "out.jsonl"
        save_data(small_dataset, str(out))
        assert out.exists()

    def test_local_jsonl_row_count(self, small_dataset: Dataset, tmp_path: Path):
        out = tmp_path / "out.jsonl"
        save_data(small_dataset, str(out))
        rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(rows) == 2

    def test_local_jsonl_content(self, small_dataset: Dataset, tmp_path: Path):
        out = tmp_path / "out.jsonl"
        save_data(small_dataset, str(out))
        rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert rows[0]["french"] == _ROWS[0]["french"]

    def test_local_creates_parent_dirs(self, small_dataset: Dataset, tmp_path: Path):
        out = tmp_path / "nested" / "dir" / "out.jsonl"
        save_data(small_dataset, str(out))
        assert out.exists()

    def test_hf_uri_calls_push_to_hub(self, small_dataset: Dataset, monkeypatch):
        pushed: dict = {}

        class MockDatasetDict(dict):
            def push_to_hub(self, repo, private=False):
                pushed["repo"] = repo
                pushed["private"] = private

        monkeypatch.setattr("moore_web.annotate.DatasetDict", MockDatasetDict)
        save_data(small_dataset, "hf://owner/repo", private=True, split="train")
        assert pushed["repo"] == "owner/repo"
        assert pushed["private"] is True


# ---------------------------------------------------------------------------
# run_quality_warnings
# ---------------------------------------------------------------------------


class TestRunQualityWarnings:
    def test_adds_quality_warnings_column(self, small_dataset: Dataset):
        result = run_quality_warnings(small_dataset, load_wordlists=False)
        assert "quality_warnings" in result.column_names

    def test_adds_identification_consistency_column(self, small_dataset: Dataset):
        result = run_quality_warnings(small_dataset, load_wordlists=False)
        assert "identification_consistency" in result.column_names

    def test_adds_len_ratio_column(self, small_dataset: Dataset):
        result = run_quality_warnings(small_dataset, load_wordlists=False)
        assert "len_ratio" in result.column_names

    def test_preserves_original_columns(self, small_dataset: Dataset):
        result = run_quality_warnings(small_dataset, load_wordlists=False)
        assert "french" in result.column_names
        assert "moore" in result.column_names

    def test_quality_warnings_is_list_of_strings(self, small_dataset: Dataset):
        result = run_quality_warnings(small_dataset, load_wordlists=False)
        for warnings in result["quality_warnings"]:
            assert isinstance(warnings, list)
            assert all(isinstance(w, str) for w in warnings)

    def test_identification_consistency_in_range(self, small_dataset: Dataset):
        result = run_quality_warnings(small_dataset, load_wordlists=False)
        for score in result["identification_consistency"]:
            assert 0.0 <= score <= 1.0

    def test_len_ratio_in_range(self, small_dataset: Dataset):
        result = run_quality_warnings(small_dataset, load_wordlists=False)
        for ratio in result["len_ratio"]:
            assert 0.0 <= ratio <= 1.0

    def test_len_ratio_symmetric(self):
        ds = Dataset.from_list(
            [
                {"french": "ab", "moore": "abcd"},
                {"french": "abcd", "moore": "ab"},
            ]
        )
        result = run_quality_warnings(ds, load_wordlists=False)
        assert result["len_ratio"][0] == result["len_ratio"][1]
        assert result["len_ratio"][0] == pytest.approx(0.5)

    def test_custom_field_names(self):
        ds = Dataset.from_list([{"src": "Bonjour", "tgt": "Yibeogo"}])
        result = run_quality_warnings(ds, src_field="src", tgt_field="tgt", load_wordlists=False)
        assert "quality_warnings" in result.column_names
        # Injected mapping columns must not leak into output
        assert "eng_Latn" not in result.column_names
        assert "mos_Latn" not in result.column_names

    def test_no_injected_columns_when_using_default_fields(self, small_dataset: Dataset):
        result = run_quality_warnings(small_dataset, load_wordlists=False)
        assert "eng_Latn" not in result.column_names
        assert "mos_Latn" not in result.column_names

    def test_emoji_detected_in_target(self):
        ds = Dataset.from_list([{"french": "Hello", "moore": "Hi 😀"}])
        result = run_quality_warnings(ds, load_wordlists=False)
        assert "emoji" in result["quality_warnings"][0]

    def test_row_count_unchanged(self, small_dataset: Dataset):
        result = run_quality_warnings(small_dataset, load_wordlists=False)
        assert len(result) == len(small_dataset)


# ---------------------------------------------------------------------------
# run_lang_id
# ---------------------------------------------------------------------------


class TestRunLangId:
    def _mock_annotate(self, dataset, model, source_col, target_col, batch_size):
        n = len(dataset)
        return (
            dataset.add_column("source_glotlid_lang", ["fra_Latn"] * n)
            .add_column("source_glotlid_prob", [0.99] * n)
            .add_column("target_glotlid_lang", ["mos_Latn"] * n)
            .add_column("target_glotlid_prob", [0.95] * n)
        )

    def test_adds_glotlid_columns(self, small_dataset: Dataset, monkeypatch):
        import moore_web.glotlid as glotlid_mod

        monkeypatch.setattr(glotlid_mod, "load_model", lambda: object())
        monkeypatch.setattr(glotlid_mod, "annotate_dataset", self._mock_annotate)
        result = run_lang_id(small_dataset)
        for col in (
            "source_glotlid_lang",
            "source_glotlid_prob",
            "target_glotlid_lang",
            "target_glotlid_prob",
        ):
            assert col in result.column_names

    def test_passes_field_names_to_annotate_dataset(self, small_dataset: Dataset, monkeypatch):
        import moore_web.glotlid as glotlid_mod

        called_with: dict = {}

        def capturing_annotate(dataset, model, source_col, target_col, batch_size):
            called_with["source_col"] = source_col
            called_with["target_col"] = target_col
            return self._mock_annotate(dataset, model, source_col, target_col, batch_size)

        monkeypatch.setattr(glotlid_mod, "load_model", lambda: object())
        monkeypatch.setattr(glotlid_mod, "annotate_dataset", capturing_annotate)
        run_lang_id(small_dataset, src_field="french", tgt_field="moore")
        assert called_with["source_col"] == "french"
        assert called_with["target_col"] == "moore"

    def test_accepts_pre_loaded_model(self, small_dataset: Dataset, monkeypatch):
        import moore_web.glotlid as glotlid_mod

        sentinel = object()
        received: list = []

        def capturing_annotate(dataset, model, source_col, target_col, batch_size):
            received.append(model)
            return self._mock_annotate(dataset, model, source_col, target_col, batch_size)

        monkeypatch.setattr(glotlid_mod, "annotate_dataset", capturing_annotate)
        run_lang_id(small_dataset, model=sentinel)
        assert received[0] is sentinel


# ---------------------------------------------------------------------------
# run_laser
# ---------------------------------------------------------------------------


class TestRunLaser:
    def _make_mock_encoder(self, embed_dim: int = 4):
        class _MockEncoder:
            def __init__(self, lang: str):
                self.lang = lang

            def encode_sentences(self, texts, normalize_embeddings=True):
                n = len(texts)
                # Return unit vectors so dot-product == 1.0
                vecs = np.ones((n, embed_dim), dtype=np.float32)
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                return vecs / norms

        return _MockEncoder

    def test_default_column_name(self, small_dataset: Dataset, monkeypatch):
        monkeypatch.setattr("laser_encoders.LaserEncoderPipeline", self._make_mock_encoder())
        result = run_laser(small_dataset)
        assert "laser_fra_mos" in result.column_names

    def test_custom_output_field(self, small_dataset: Dataset, monkeypatch):
        monkeypatch.setattr("laser_encoders.LaserEncoderPipeline", self._make_mock_encoder())
        result = run_laser(small_dataset, output_field="my_laser")
        assert "my_laser" in result.column_names

    def test_laser_score_count_matches_rows(self, small_dataset: Dataset, monkeypatch):
        monkeypatch.setattr("laser_encoders.LaserEncoderPipeline", self._make_mock_encoder())
        result = run_laser(small_dataset)
        assert len(result["laser_fra_mos"]) == len(small_dataset)

    def test_laser_score_range(self, small_dataset: Dataset, monkeypatch):
        monkeypatch.setattr("laser_encoders.LaserEncoderPipeline", self._make_mock_encoder())
        result = run_laser(small_dataset)
        for score in result["laser_fra_mos"]:
            assert -1.0 <= score <= 1.0

    def test_unit_vectors_give_score_one(self, small_dataset: Dataset, monkeypatch):
        monkeypatch.setattr("laser_encoders.LaserEncoderPipeline", self._make_mock_encoder())
        result = run_laser(small_dataset)
        for score in result["laser_fra_mos"]:
            assert score == pytest.approx(1.0, abs=1e-3)

    def test_preserves_original_columns(self, small_dataset: Dataset, monkeypatch):
        monkeypatch.setattr("laser_encoders.LaserEncoderPipeline", self._make_mock_encoder())
        result = run_laser(small_dataset)
        assert "french" in result.column_names
        assert "moore" in result.column_names


# ---------------------------------------------------------------------------
# run_comet_qe
# ---------------------------------------------------------------------------


class TestRunCometQe:
    def _make_mock_model(self, scores: list[float]):
        mock_output = MagicMock()
        mock_output.scores = scores
        mock_model = MagicMock()
        mock_model.predict.return_value = mock_output
        return mock_model

    def test_default_column_name(self, small_dataset: Dataset):
        model = self._make_mock_model([0.85, 0.73])
        result = run_comet_qe(small_dataset, model=model)
        assert "comet_qe_french_moore" in result.column_names

    def test_custom_output_field(self, small_dataset: Dataset):
        model = self._make_mock_model([0.85, 0.73])
        result = run_comet_qe(small_dataset, output_field="my_score", model=model)
        assert "my_score" in result.column_names

    def test_comet_qe_values(self, small_dataset: Dataset):
        model = self._make_mock_model([0.85, 0.73])
        result = run_comet_qe(small_dataset, model=model)
        col = "comet_qe_french_moore"
        assert result[col][0] == pytest.approx(0.85)
        assert result[col][1] == pytest.approx(0.73)

    def test_comet_qe_row_count(self, small_dataset: Dataset):
        model = self._make_mock_model([0.85, 0.73])
        result = run_comet_qe(small_dataset, model=model)
        assert len(result["comet_qe_french_moore"]) == len(small_dataset)

    def test_passes_correct_fields_to_model(self, small_dataset: Dataset):
        model = self._make_mock_model([0.0, 0.0])
        run_comet_qe(small_dataset, src_field="french", tgt_field="moore", model=model)
        call_args = model.predict.call_args
        data = call_args[0][0]
        assert data[0]["src"] == _ROWS[0]["french"]
        assert data[0]["mt"] == _ROWS[0]["moore"]

    def test_preserves_original_columns(self, small_dataset: Dataset):
        model = self._make_mock_model([0.85, 0.73])
        result = run_comet_qe(small_dataset, model=model)
        assert "french" in result.column_names
        assert "moore" in result.column_names


# ---------------------------------------------------------------------------
# annotate (composer)
# ---------------------------------------------------------------------------


class TestAnnotate:
    def test_no_flags_returns_same_columns(self, small_dataset: Dataset):
        result = annotate(small_dataset)
        assert set(result.column_names) == set(small_dataset.column_names)

    def test_quality_warn_adds_column(self, small_dataset: Dataset):
        result = annotate(small_dataset, quality_warn=True, load_wordlists=False)
        assert "quality_warnings" in result.column_names

    def test_consistency_adds_column(self, small_dataset: Dataset):
        result = annotate(small_dataset, consistency=True, load_wordlists=False)
        assert "identification_consistency" in result.column_names

    def test_quality_warn_and_consistency_single_pass(self, small_dataset: Dataset, monkeypatch):
        """Both flags should trigger run_quality_warnings exactly once."""
        call_count: list[int] = [0]
        original = run_quality_warnings

        def counting_wrapper(*args, **kwargs):
            call_count[0] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr("moore_web.annotate.run_quality_warnings", counting_wrapper)
        annotate(small_dataset, quality_warn=True, consistency=True, load_wordlists=False)
        assert call_count[0] == 1

    def test_laser_flag(self, small_dataset: Dataset, monkeypatch):
        monkeypatch.setattr(
            "moore_web.annotate.run_laser",
            lambda ds, **kw: ds.add_column("laser_fra_mos", [0.9] * len(ds)),
        )
        result = annotate(small_dataset, laser=True)
        assert "laser_fra_mos" in result.column_names

    def test_comet_qe_flag(self, small_dataset: Dataset, monkeypatch):
        monkeypatch.setattr(
            "moore_web.annotate.run_comet_qe",
            lambda ds, **kw: ds.add_column("comet_qe_french_moore", [0.8] * len(ds)),
        )
        result = annotate(small_dataset, comet_qe=True)
        assert "comet_qe_french_moore" in result.column_names

    def test_multiple_flags_stack(self, small_dataset: Dataset, monkeypatch):
        monkeypatch.setattr(
            "moore_web.annotate.run_laser",
            lambda ds, **kw: ds.add_column("laser_fra_mos", [0.9] * len(ds)),
        )
        result = annotate(small_dataset, quality_warn=True, laser=True, load_wordlists=False)
        assert "quality_warnings" in result.column_names
        assert "laser_fra_mos" in result.column_names
