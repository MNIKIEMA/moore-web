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

import numpy as np

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

    Language codes are resolved in this order:
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
                      ``"laser_{src_lang}_{tgt_lang}"`` when ``None``.
        encoder_src:  Pre-loaded source ``LaserEncoderPipeline``; loaded
                      automatically if ``None``.
        encoder_tgt:  Pre-loaded target ``LaserEncoderPipeline``; loaded
                      automatically if ``None``.

    Returns:
        Annotated ``datasets.Dataset`` with an added score column.
    """
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

    src_texts: list[str] = dataset[src_field]
    tgt_texts: list[str] = dataset[tgt_field]

    print(f"Encoding {len(src_texts):,} source sentences…")
    src_embs: np.ndarray = encoder_src.encode_sentences(src_texts, normalize_embeddings=True)
    print(f"Encoding {len(tgt_texts):,} target sentences…")
    tgt_embs: np.ndarray = encoder_tgt.encode_sentences(tgt_texts, normalize_embeddings=True)

    # Dot product on unit vectors == cosine similarity
    scores = [round(float(s), 4) for s in (src_embs * tgt_embs).sum(axis=1).tolist()]
    return dataset.add_column(output_field, scores)
