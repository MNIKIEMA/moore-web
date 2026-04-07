"""Annotation module for aligned bilingual datasets.

Each ``run_*`` function takes a HuggingFace ``Dataset`` and returns an annotated
``Dataset`` with one or more new columns added.  Functions are composable and
independent — call only what you need.

IO helpers ``load_data`` / ``save_data`` handle both local JSONL files and
HuggingFace Hub datasets via ``hf://owner/repo`` URIs.

New columns per function
------------------------
- :func:`run_lang_id`          → ``source_glotlid_lang``, ``source_glotlid_prob``,
                                  ``target_glotlid_lang``, ``target_glotlid_prob``
- :func:`run_quality_warnings` → ``quality_warnings`` (list[str]), ``identification_consistency`` (float),
                                  ``len_ratio`` (float)
- :func:`run_laser`            → ``laser_score`` (float)
- :func:`run_comet_qe`         → ``comet_qe`` (float)

Examples
--------
    # Local JSONL
    from moore_web.annotate import load_data, save_data, run_quality_warnings
    ds = load_data("data.jsonl")
    ds = run_quality_warnings(ds, src_field="french", tgt_field="moore")
    save_data(ds, "annotated.jsonl")

    # HuggingFace Hub
    ds = load_data("hf://owner/repo")
    ds = run_quality_warnings(ds)
    save_data(ds, "hf://owner/repo-annotated")
"""

from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset  # noqa: F401 — re-exported for monkeypatching

# ---------------------------------------------------------------------------
# HF URI helpers
# ---------------------------------------------------------------------------

_HF_PREFIX = "hf://"


def _is_hf(path: str) -> bool:
    return path.startswith(_HF_PREFIX)


def _hf_repo(path: str) -> str:
    return path[len(_HF_PREFIX) :]


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def load_data(path: str, split: str = "train"):
    """Load an aligned dataset from a local JSONL file or a HuggingFace Hub repo.

    Args:
        path:  Local file path **or** ``hf://owner/repo`` URI.
        split: Dataset split to load (HF mode only, default: ``"train"``).

    Returns:
        A ``datasets.Dataset``.
    """

    if _is_hf(path):
        repo = _hf_repo(path)
        print(f"Loading '{repo}' (split={split}) from HuggingFace Hub…")
        return load_dataset(repo, split=split)

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    rows: list[dict] = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    print(f"Loaded {len(rows):,} rows from {p.name}.")
    return Dataset.from_list(rows)


def save_data(dataset, path: str, private: bool = False, split: str = "train") -> None:
    """Write an annotated dataset to a local JSONL file or push to HuggingFace Hub.

    Args:
        dataset: A ``datasets.Dataset``.
        path:    Local file path **or** ``hf://owner/repo`` URI.
        private: Push as a private dataset (HF mode only).
        split:   Split name used when wrapping in a ``DatasetDict`` (HF mode only).
    """
    if _is_hf(path):
        repo = _hf_repo(path)
        print(f"Pushing {len(dataset):,} rows → '{repo}' …")
        DatasetDict({split: dataset}).push_to_hub(repo, private=private)
        print(f"Done. https://huggingface.co/datasets/{repo}")
        return

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {len(dataset):,} rows → {out} …")
    with out.open("w", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("Done.")


# ---------------------------------------------------------------------------
# Annotation: GlotLID language identification
# ---------------------------------------------------------------------------


def run_lang_id(
    dataset,
    src_field: str = "french",
    tgt_field: str = "moore",
    batch_size: int = 1000,
    model=None,
):
    """Add GlotLID language-ID predictions for source and target columns.

    Adds four columns: ``source_glotlid_lang``, ``source_glotlid_prob``,
    ``target_glotlid_lang``, ``target_glotlid_prob``.

    Args:
        dataset:    Input ``datasets.Dataset``.
        src_field:  Source column name (default: ``"french"``).
        tgt_field:  Target column name (default: ``"moore"``).
        batch_size: Rows per batch for model inference.
        model:      Pre-loaded GlotLID fasttext model; loaded automatically if ``None``.

    Returns:
        Annotated ``datasets.Dataset``.
    """
    from moore_web import glotlid

    if model is None:
        model = glotlid.load_model()

    print(f"Running GlotLID on '{src_field}' and '{tgt_field}' ({len(dataset):,} rows)…")
    return glotlid.annotate_dataset(
        dataset,
        model=model,
        source_col=src_field,
        target_col=tgt_field,
        batch_size=batch_size,
    )


# ---------------------------------------------------------------------------
# Annotation: quality warnings + identification consistency
# ---------------------------------------------------------------------------


def _build_foreign_wordlist(load_wordlists: bool) -> set[str]:
    """Build the foreign-word exclusion set used by quality-warning checks."""
    if not load_wordlists:
        return set()

    from moore_web.wordlists import build_foreign_wordlist

    return build_foreign_wordlist()


def run_quality_warnings(
    dataset,
    src_field: str = "french",
    tgt_field: str = "moore",
    batch_size: int = 1000,
    load_wordlists: bool = True,
):
    """Add quality-warning annotations to each row.

    Adds three columns:

    - ``quality_warnings`` — list of active warning labels
      (``"emoji"``, ``"dots_asymmetry"``, ``"number_mismatch"``,
      ``"parenthesis_asymmetry"``, ``"bullet_asymmetry"``, ``"foreign_words"``).
    - ``identification_consistency`` — float in [0, 1]: fraction of target tokens
      absent from the foreign word list (higher = more Mooré-consistent).
    - ``len_ratio`` — float in [0, 1]: ``min(len(src), len(tgt)) / max(len(src), len(tgt))``.

    Args:
        dataset:        Input ``datasets.Dataset``.
        src_field:      Source column name (default: ``"french"``).
        tgt_field:      Target column name (default: ``"moore"``).
        batch_size:     Rows per batch for ``dataset.map``.
        load_wordlists: Load GlotLID + spellchecker foreign-word lists for richer
                        detection.  Set to ``False`` to skip (faster, no HF download).

    Returns:
        Annotated ``datasets.Dataset``.
    """
    from moore_web.filter_nllb import annotate_warnings

    foreign_wordlist = _build_foreign_wordlist(load_wordlists)

    # annotate_warnings uses hardcoded column names "eng_Latn" / "mos_Latn".
    # Inject temporary mappings and remove them from the output when the caller
    # uses different field names.
    _SRC_KEY = "eng_Latn"
    _TGT_KEY = "mos_Latn"
    src_injected = src_field != _SRC_KEY
    tgt_injected = tgt_field != _TGT_KEY

    def _batch_fn(batch: dict) -> dict:
        batch[_SRC_KEY] = batch[src_field]
        batch[_TGT_KEY] = batch[tgt_field]
        result = annotate_warnings(batch, foreign_wordlist)
        if src_injected:
            result.pop(_SRC_KEY, None)
        if tgt_injected:
            result.pop(_TGT_KEY, None)
        return result

    print(f"Annotating quality warnings ({len(dataset):,} rows)…")
    return dataset.map(
        _batch_fn,
        batched=True,
        batch_size=batch_size,
        desc="quality warnings",
    )


# ---------------------------------------------------------------------------
# Annotation: LASER cosine similarity
# ---------------------------------------------------------------------------


def run_laser(
    dataset,
    src_field: str = "french",
    tgt_field: str = "moore",
    src_lang: str = "fra",
    tgt_lang: str = "mos",
    output_field: str | None = None,
    encoder_src=None,
    encoder_tgt=None,
):
    """Add LASER cosine-similarity scores between source and target sentences.

    Adds one column named ``output_field`` (default: ``"laser_{src_lang}_{tgt_lang}"``).

    Unlike :func:`~moore_web.score_mt_datasets.score_aligned_pairs`, this function
    does **not** drop rows — it annotates every row unconditionally.

    Args:
        dataset:      Input ``datasets.Dataset``.
        src_field:    Source column name (default: ``"french"``).
        tgt_field:    Target column name (default: ``"moore"``).
        src_lang:     LASER language code for the source encoder (default: ``"fra"``).
        tgt_lang:     LASER language code for the target encoder (default: ``"mos"``).
        output_field: Name for the new score column. Defaults to
                      ``"laser_{src_lang}_{tgt_lang}"`` when ``None``.
        encoder_src:  Pre-loaded source encoder; loaded automatically if ``None``.
        encoder_tgt:  Pre-loaded target encoder; loaded automatically if ``None``.

    Returns:
        Annotated ``datasets.Dataset``.
    """
    from moore_web.score_laser import score_dataset

    return score_dataset(
        dataset,
        src_field=src_field,
        tgt_field=tgt_field,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        output_field=output_field,
        encoder_src=encoder_src,
        encoder_tgt=encoder_tgt,
    )


# ---------------------------------------------------------------------------
# Annotation: COMET-QE translation quality
# ---------------------------------------------------------------------------


def run_comet_qe(
    dataset,
    src_field: str = "french",
    tgt_field: str = "moore",
    output_field: str | None = None,
    batch_size: int = 8,
    gpus: int = 1,
    model=None,
):
    """Add COMET-QE reference-free translation quality scores.

    Adds one column named ``output_field`` (default: ``"comet_qe_{src_field}_{tgt_field}"``).

    Uses ``McGill-NLP/ssa-comet-qe`` (~1.5 GB download on first run).

    Args:
        dataset:      Input ``datasets.Dataset``.
        src_field:    Source column name (default: ``"french"``).
        tgt_field:    Target column name (default: ``"moore"``).
        output_field: Name for the new score column. Defaults to
                      ``"comet_qe_{src_field}_{tgt_field}"`` when ``None``.
        batch_size:   Rows per inference batch.
        gpus:         Number of GPUs to use (0 = CPU).
        model:        Pre-loaded COMET model; loaded automatically if ``None``.

    Returns:
        Annotated ``datasets.Dataset``.
    """
    from moore_web.score_comet_qe import score_dataset

    return score_dataset(
        dataset,
        src_field=src_field,
        tgt_field=tgt_field,
        output_field=output_field,
        batch_size=batch_size,
        gpus=gpus,
        model=model,
    )


# ---------------------------------------------------------------------------
# Composer
# ---------------------------------------------------------------------------


def annotate(
    dataset,
    src_field: str = "french",
    tgt_field: str = "moore",
    *,
    lang_id: bool = False,
    quality_warn: bool = False,
    consistency: bool = False,
    laser: bool = False,
    comet_qe: bool = False,
    load_wordlists: bool = True,
    batch_size: int = 1000,
    laser_batch_size: int = 512,
    comet_batch_size: int = 8,
    gpus: int = 1,
):
    """Run any combination of annotation steps on a dataset.

    ``quality_warn`` and ``consistency`` both call :func:`run_quality_warnings` in a
    single pass (the foreign wordlist is loaded only once).

    Args:
        dataset:           Input ``datasets.Dataset``.
        src_field:         Source column name.
        tgt_field:         Target column name.
        lang_id:           Add GlotLID language-ID columns.
        quality_warn:      Add ``quality_warnings`` column.
        consistency:       Add ``identification_consistency`` column.
        laser:             Add ``laser_score`` column.
        comet_qe:          Add ``comet_qe`` column.
        load_wordlists:    Load foreign-word lists for quality-warning checks.
        batch_size:        Rows per batch for lang-ID and warning annotation.
        laser_batch_size:  Rows per batch for LASER encoding.
        comet_batch_size:  Rows per inference batch for COMET-QE.
        gpus:              Number of GPUs for COMET-QE (0 = CPU).

    Returns:
        Annotated ``datasets.Dataset``.
    """
    if lang_id:
        dataset = run_lang_id(dataset, src_field=src_field, tgt_field=tgt_field, batch_size=batch_size)

    if quality_warn or consistency:
        dataset = run_quality_warnings(
            dataset,
            src_field=src_field,
            tgt_field=tgt_field,
            batch_size=batch_size,
            load_wordlists=load_wordlists,
        )

    if laser:
        dataset = run_laser(dataset, src_field=src_field, tgt_field=tgt_field)

    if comet_qe:
        dataset = run_comet_qe(
            dataset,
            src_field=src_field,
            tgt_field=tgt_field,
            batch_size=comet_batch_size,
            gpus=gpus,
        )

    return dataset
