"""Compute LASER cosine-similarity scores for aligned bilingual datasets.

Provides a dataset-level API (``score_dataset``) that annotates every row
without dropping any pairs, and a lower-level ``load_encoders`` helper for
reusing already-loaded models.

See also ``score_mt_datasets.score_aligned_pairs`` for a list-based API that
filters pairs below a minimum score and returns an ``AlignedCorpus``.

Usage
-----
    from moore_web.score_laser import score_dataset
    ds = score_dataset(dataset, src_field="french", tgt_field="moore")
"""

from __future__ import annotations

# Known field-name → LASER language code mappings for this project.
# Only covers our use-case columns; for any other field the caller must
# pass the correct LASER code explicitly via src_lang / tgt_lang.
_FIELD_TO_LANG: dict[str, str] = {
    # French
    "french": "fra",
    "fr": "fra",
    "fra": "fra",
    "fra_Latn": "fra_Latn",
    # English
    "english": "eng",
    "en": "eng",
    "eng": "eng",
    "eng_Latn": "eng_Latn",
    # Mooré
    "moore": "mos",
    "mooré": "mos",
    "mo": "mos",
    "mos": "mos",
    "mos_Latn": "mos_Latn",
}


def _encode_pair(encoder_src, encoder_tgt, src_texts: list[str], tgt_texts: list[str]) -> list[float]:
    """Cosine similarity between two already-loaded encoders' embeddings."""
    print(f"Encoding {len(src_texts):,} source sentences…")
    src_embs = encoder_src.encode_sentences(src_texts, normalize_embeddings=True)
    print(f"Encoding {len(tgt_texts):,} target sentences…")
    tgt_embs = encoder_tgt.encode_sentences(tgt_texts, normalize_embeddings=True)
    return [round(float(s), 4) for s in (src_embs * tgt_embs).sum(axis=1).tolist()]


def load_encoders(src_lang: str = "fra", tgt_lang: str = "mos"):
    """Load and return a (src_encoder, tgt_encoder) pair.

    Args:
        src_lang: LASER language code for the source side (default: ``"fra"``).
        tgt_lang: LASER language code for the target side (default: ``"mos"``).

    Returns:
        A ``(laser_src, laser_tgt)`` tuple of ``LaserEncoderPipeline`` instances.
    """
    from laser_encoders import LaserEncoderPipeline

    print(f"Loading LASER {src_lang} model…")
    laser_src = LaserEncoderPipeline(lang=src_lang)
    print(f"Loading LASER {tgt_lang} model…")
    laser_tgt = LaserEncoderPipeline(lang=tgt_lang)
    return laser_src, laser_tgt


def score_dataset(
    dataset,
    src_field: str = "french",
    tgt_field: str = "moore",
    src_lang: str | None = None,
    tgt_lang: str | None = None,
    output_field: str | None = None,
    encoder_src=None,
    encoder_tgt=None,
):
    """Add LASER cosine-similarity scores to every row of a HuggingFace ``Dataset``.

    Annotates all rows unconditionally — no rows are dropped.  For a
    filtering variant see ``score_mt_datasets.score_aligned_pairs``.

    If the dataset has ``src_lang``/``tgt_lang`` columns (the long-format HF
    schema -- see ``moore_web.flatten.flat_rows_to_long``) *and* neither
    ``src_lang`` nor ``tgt_lang`` was passed explicitly, language codes are
    read per row instead of assumed uniform for the whole dataset: rows are
    grouped by their distinct (src_lang, tgt_lang) pair, each group is scored
    with its own encoder pair, and the results are reassembled in original
    row order. This is what makes a dataset with mixed pairs (e.g. a
    trilingual dictionary entry's interspersed mos-fra and mos-eng rows)
    score correctly instead of encoding every row with one fixed pair.
    Passing ``src_lang``/``tgt_lang`` explicitly always forces the single-pair
    path below, even if those columns are present.

    Otherwise, language codes are resolved in this order:
    1. Explicit ``src_lang`` / ``tgt_lang`` arguments (highest priority).
    2. ``_FIELD_TO_LANG`` lookup on the field name (covers our known columns).
    3. ``ValueError`` if neither resolves — the caller must pass the lang code.

    Args:
        dataset:      Input ``datasets.Dataset``.
        src_field:    Source column name (default: ``"french"``).
        tgt_field:    Target column name (default: ``"moore"``).
        src_lang:     LASER language code for the source encoder. Inferred from
                      ``src_field`` when ``None``.
        tgt_lang:     LASER language code for the target encoder. Inferred from
                      ``tgt_field`` when ``None``.
        output_field: Name of the new score column. Defaults to
                      ``"laser_{src_lang}_{tgt_lang}"`` when ``None`` (single-pair
                      path), or ``"laser_score"`` in the per-row-language path,
                      since the pair itself already varies by row.
        encoder_src:  Pre-loaded source ``LaserEncoderPipeline``; loaded
                      automatically if ``None``. Ignored in the per-row-language
                      path when more than one distinct language pair is present.
        encoder_tgt:  Pre-loaded target ``LaserEncoderPipeline``; loaded
                      automatically if ``None``. Same caveat as ``encoder_src``.

    Returns:
        Annotated ``datasets.Dataset`` with an added score column.
    """
    if (
        src_lang is None
        and tgt_lang is None
        and "src_lang" in dataset.column_names
        and "tgt_lang" in dataset.column_names
    ):
        return _score_dataset_per_row_lang(dataset, src_field, tgt_field, output_field, encoder_src, encoder_tgt)

    src_lang = src_lang or _FIELD_TO_LANG.get(src_field)
    tgt_lang = tgt_lang or _FIELD_TO_LANG.get(tgt_field)

    if src_lang is None:
        raise ValueError(f"Cannot infer LASER lang for src_field={src_field!r}. Pass src_lang explicitly.")
    if tgt_lang is None:
        raise ValueError(f"Cannot infer LASER lang for tgt_field={tgt_field!r}. Pass tgt_lang explicitly.")

    if output_field is None:
        output_field = f"laser_{src_lang}_{tgt_lang}"

    if encoder_src is None or encoder_tgt is None:
        encoder_src, encoder_tgt = load_encoders(src_lang, tgt_lang)

    scores = _encode_pair(encoder_src, encoder_tgt, dataset[src_field], dataset[tgt_field])
    return dataset.add_column(output_field, scores)


def _score_dataset_per_row_lang(
    dataset,
    src_field: str,
    tgt_field: str,
    output_field: str | None,
    encoder_src,
    encoder_tgt,
):
    """Score a dataset whose (src_lang, tgt_lang) varies per row.

    Groups rows by their distinct language pair, scores each group with its
    own encoder pair (caching encoders by language so e.g. a shared "mos"
    source encoder is loaded once even across two different target-language
    groups), then reassembles in original row order.
    """
    from datasets import concatenate_datasets

    output_field = output_field or "laser_score"
    pairs = sorted(set(zip(dataset["src_lang"], dataset["tgt_lang"])))
    print(f"Scoring {len(dataset):,} rows across {len(pairs)} language pair(s): {pairs}")

    encoders: dict[str, object] = {}

    def _get_encoder(lang: str):
        if lang not in encoders:
            from laser_encoders import LaserEncoderPipeline

            print(f"Loading LASER {lang} model…")
            encoders[lang] = LaserEncoderPipeline(lang=lang)
        return encoders[lang]

    indexed = dataset.add_column("__row_idx__", list(range(len(dataset))))

    scored_parts = []
    for pair_src_lang, pair_tgt_lang in pairs:
        subset = indexed.filter(
            lambda r, sl=pair_src_lang, tl=pair_tgt_lang: r["src_lang"] == sl and r["tgt_lang"] == tl
        )
        if len(subset) == 0:
            continue
        # Reuse caller-provided encoders only when there's exactly one pair
        # (otherwise they'd be wrong for every group but the first).
        enc_src = encoder_src if encoder_src is not None and len(pairs) == 1 else _get_encoder(pair_src_lang)
        enc_tgt = encoder_tgt if encoder_tgt is not None and len(pairs) == 1 else _get_encoder(pair_tgt_lang)
        scores = _encode_pair(enc_src, enc_tgt, subset[src_field], subset[tgt_field])
        scored_parts.append(subset.add_column(output_field, scores))

    merged = concatenate_datasets(scored_parts)
    return merged.sort("__row_idx__").remove_columns("__row_idx__")
