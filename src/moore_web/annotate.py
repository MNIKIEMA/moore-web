"""Annotation module for aligned bilingual datasets.

Each ``run_*`` function takes a HuggingFace ``Dataset`` and returns an annotated
``Dataset`` with one or more new columns added.  Functions are composable and
independent — call only what you need.

IO helpers ``load_data`` / ``save_data`` handle both local JSONL files and
HuggingFace Hub datasets via ``hf://owner/repo`` URIs.

New columns per function
------------------------
- :func:`run_lang_id`          → ``{src}_glotlid_lang``, ``{src}_glotlid_prob``,
                                  ``{tgt}_glotlid_lang``, ``{tgt}_glotlid_prob``
                                  (column names derived from ``src_field`` / ``tgt_field``)
- :func:`run_quality_warnings` → ``quality_warnings`` (list[str]), ``identification_consistency`` (float)
- :func:`run_len_ratio`        → ``len_ratio`` (float)
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


def load_data(path: str, split: str = "train", config_name: str | None = None):
    """Load an aligned dataset from a local JSONL file or a HuggingFace Hub repo.

    Args:
        path:        Local file path **or** ``hf://owner/repo`` URI.
        split:       Dataset split to load (HF mode only, default: ``"train"``).
        config_name: Config to load (HF mode only) -- required when the repo has
                      more than one, e.g. a language pair like ``"mos-fra"`` from
                      a repo ``save_data`` split by pair (see its docstring).
                      ``None`` loads the repo's default config.

    Returns:
        A ``datasets.Dataset``.
    """

    if _is_hf(path):
        repo = _hf_repo(path)
        if config_name is not None:
            print(f"Loading '{repo}' (config={config_name}, split={split}) from HuggingFace Hub…")
            return load_dataset(repo, config_name, split=split)
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


def _split_by_lang_pair(dataset) -> dict[tuple[str, str], "Dataset"] | None:
    """Split a dataset into one sub-dataset per distinct (src_lang, tgt_lang) pair.

    Returns ``None`` (nothing to split) when the dataset has no
    ``src_lang``/``tgt_lang`` columns, or only one distinct pair is present.
    """
    if "src_lang" not in dataset.column_names or "tgt_lang" not in dataset.column_names:
        return None
    pairs = sorted(set(zip(dataset["src_lang"], dataset["tgt_lang"])))
    if len(pairs) <= 1:
        return None
    return {
        pair: dataset.filter(lambda r, sl=pair[0], tl=pair[1]: r["src_lang"] == sl and r["tgt_lang"] == tl)
        for pair in pairs
    }


def _write_jsonl_file(dataset, out: Path) -> None:
    with out.open("w", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_data(dataset, path: str, private: bool = False, split: str = "train") -> None:
    """Write an annotated dataset to local JSONL file(s) or push to HuggingFace Hub.

    A dataset with more than one distinct (src_lang, tgt_lang) pair (e.g. a
    trilingual dictionary's mos-fra rows mixed with its mos-eng rows) is split
    into one clean bitext per pair instead of one file/config a consumer has
    to filter first -- the convention most HF parallel-data consumers expect
    (e.g. opus100 ships separate en-fr, en-de, ... configs). Locally, each
    pair gets its own file (``out.stem + ".{src}-{tgt}" + out.suffix``); on
    the Hub, each pair is pushed as its own config within the same repo
    (``load_dataset(repo, "mos-fra")`` vs. ``load_dataset(repo, "mos-eng")``).
    A dataset with zero or one pair (or without src_lang/tgt_lang columns at
    all -- e.g. a legacy flat french/moore dataset) is written as a single
    file/config, unchanged from before.

    Args:
        dataset: A ``datasets.Dataset``.
        path:    Local file path **or** ``hf://owner/repo`` URI.
        private: Push as a private dataset (HF mode only).
        split:   Split name used when wrapping in a ``DatasetDict`` (HF mode only).
    """
    by_pair = _split_by_lang_pair(dataset)

    if _is_hf(path):
        repo = _hf_repo(path)
        if by_pair is None:
            print(f"Pushing {len(dataset):,} rows → '{repo}' …")
            DatasetDict({split: dataset}).push_to_hub(repo, private=private)
        else:
            for (src_lang, tgt_lang), sub in by_pair.items():
                config_name = f"{src_lang}-{tgt_lang}"
                print(f"Pushing {len(sub):,} rows ({config_name}) → '{repo}' …")
                DatasetDict({split: sub}).push_to_hub(repo, config_name=config_name, private=private)
        print(f"Done. https://huggingface.co/datasets/{repo}")
        return

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if by_pair is None:
        print(f"Writing {len(dataset):,} rows → {out} …")
        _write_jsonl_file(dataset, out)
    else:
        for (src_lang, tgt_lang), sub in by_pair.items():
            sub_path = out.with_name(f"{out.stem}.{src_lang}-{tgt_lang}{out.suffix}")
            print(f"Writing {len(sub):,} rows ({src_lang}-{tgt_lang}) → {sub_path} …")
            _write_jsonl_file(sub, sub_path)
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

    Adds four columns derived from the field names:
    ``{src_field}_glotlid_lang``, ``{src_field}_glotlid_prob``,
    ``{tgt_field}_glotlid_lang``, ``{tgt_field}_glotlid_prob``.

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
    load_wordlists: bool = True,
):
    """Add quality-warning annotations to each row.

    Adds two columns:

    - ``quality_warnings`` — list of active warning labels
      (``"emoji"``, ``"dots_asymmetry"``, ``"number_mismatch"``,
      ``"parenthesis_asymmetry"``, ``"bullet_asymmetry"``, ``"foreign_words"``).
    - ``identification_consistency`` — float in [0, 1]: fraction of target tokens
      absent from the foreign word list (higher = more Mooré-consistent).

    Args:
        dataset:        Input ``datasets.Dataset``.
        src_field:      Source column name (default: ``"french"``).
        tgt_field:      Target column name (default: ``"moore"``).
        load_wordlists: Load GlotLID + spellchecker foreign-word lists for richer
                        detection.  Set to ``False`` to skip (faster, no HF download).

    Returns:
        Annotated ``datasets.Dataset``.
    """
    from moore_web.filter_nllb import annotate_warnings

    foreign_wordlist = _build_foreign_wordlist(load_wordlists)

    print(f"Annotating quality warnings ({len(dataset):,} rows)…")
    return dataset.map(
        lambda batch: annotate_warnings(batch, foreign_wordlist, src_col=src_field, tgt_col=tgt_field),
        batched=True,
        desc="quality warnings",
    )


# ---------------------------------------------------------------------------
# Annotation: length ratio
# ---------------------------------------------------------------------------


def run_len_ratio(
    dataset,
    src_field: str = "french",
    tgt_field: str = "moore",
):
    """Add character-length ratio between source and target sentences.

    Adds one column:

    - ``len_ratio`` — float in [0, 1]: ``min(len(src), len(tgt)) / max(len(src), len(tgt))``.
      A value close to 1.0 means both sentences are similar in length; 0.0 when either is empty.

    Args:
        dataset:    Input ``datasets.Dataset``.
        src_field:  Source column name (default: ``"french"``).
        tgt_field:  Target column name (default: ``"moore"``).

    Returns:
        Annotated ``datasets.Dataset``.
    """
    from moore_web.filter_nllb import annotate_len_ratio

    print(f"Annotating len_ratio ({len(dataset):,} rows)…")
    return dataset.map(
        lambda batch: annotate_len_ratio(batch, src_col=src_field, tgt_col=tgt_field),
        batched=True,
        desc="len_ratio",
    )


# ---------------------------------------------------------------------------
# Annotation: LASER cosine similarity
# ---------------------------------------------------------------------------


def run_laser(
    dataset,
    src_field: str = "french",
    tgt_field: str = "moore",
    src_lang: str | None = None,
    tgt_lang: str | None = None,
    output_field: str | None = None,
    encoder_src=None,
    encoder_tgt=None,
):
    """Add LASER cosine-similarity scores between source and target sentences.

    Adds one column named ``output_field`` (default: ``"laser_score"``).

    Unlike :func:`~moore_web.score_mt_datasets.score_aligned_pairs`, this function
    does **not** drop rows — it annotates every row unconditionally.

    Args:
        dataset:      Input ``datasets.Dataset``.
        src_field:    Source column name (default: ``"french"``).
        tgt_field:    Target column name (default: ``"moore"``).
        src_lang:     LASER language code for the source encoder. Left ``None`` to
                      infer from ``src_field`` (or, if the dataset has ``src_lang``/
                      ``tgt_lang`` columns, to score each row with its own pair).
        tgt_lang:     LASER language code for the target encoder. Same fallback as
                      ``src_lang``.
        output_field: Name for the new score column. Defaults to
                      ``"laser_score"`` when ``None``.
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

    Adds one column named ``output_field`` (default: ``"comet_qe"``).

    Uses ``McGill-NLP/ssa-comet-qe`` (~1.5 GB download on first run).

    Args:
        dataset:      Input ``datasets.Dataset``.
        src_field:    Source column name (default: ``"french"``).
        tgt_field:    Target column name (default: ``"moore"``).
        output_field: Name for the new score column. Defaults to
                      ``"comet_qe"`` when ``None``.
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
    len_ratio: bool = False,
    laser: bool = False,
    comet_qe: bool = False,
    load_wordlists: bool = True,
    batch_size: int = 1000,
    comet_batch_size: int = 8,
    gpus: int = 1,
    src_lang: str | None = None,
    tgt_lang: str | None = None,
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
        len_ratio:         Add ``len_ratio`` column.
        laser:             Add ``laser_score`` column.
        comet_qe:          Add ``comet_qe`` column.
        load_wordlists:    Load foreign-word lists for quality-warning checks.
        batch_size:        Rows per batch for lang-ID and warning annotation.
        comet_batch_size:  Rows per inference batch for COMET-QE.
        gpus:              Number of GPUs for COMET-QE (0 = CPU).
        src_lang:          LASER language code for the source encoder. Falls back to
                           ``FIELD_TO_LANG`` then the ``run_laser`` default.
        tgt_lang:          LASER language code for the target encoder. Falls back to
                           ``FIELD_TO_LANG`` then the ``run_laser`` default.

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
            load_wordlists=load_wordlists,
        )

    if len_ratio:
        dataset = run_len_ratio(dataset, src_field=src_field, tgt_field=tgt_field)

    if laser:
        laser_kwargs = {}
        if src_lang is not None:
            laser_kwargs["src_lang"] = src_lang
        if tgt_lang is not None:
            laser_kwargs["tgt_lang"] = tgt_lang
        dataset = run_laser(dataset, src_field=src_field, tgt_field=tgt_field, **laser_kwargs)

    if comet_qe:
        dataset = run_comet_qe(
            dataset,
            src_field=src_field,
            tgt_field=tgt_field,
            batch_size=comet_batch_size,
            gpus=gpus,
        )

    return dataset
