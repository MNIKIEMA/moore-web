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
    src_lang: str = "fra",
    tgt_lang: str = "mos",
    output_field: str | None = None,
    encoder_src=None,
    encoder_tgt=None,
):
    """Add LASER cosine-similarity scores to every row of a HuggingFace ``Dataset``.

    Annotates all rows unconditionally — no rows are dropped.  For a
    filtering variant see ``score_mt_datasets.score_aligned_pairs``.

    Args:
        dataset:      Input ``datasets.Dataset``.
        src_field:    Source column name (default: ``"french"``).
        tgt_field:    Target column name (default: ``"moore"``).
        src_lang:     LASER language code for the source encoder
                      (default: ``"fra"``).
        tgt_lang:     LASER language code for the target encoder
                      (default: ``"mos"``).
        output_field: Name of the new score column. Defaults to
                      ``"laser_{src_lang}_{tgt_lang}"`` when ``None``.
        encoder_src:  Pre-loaded source ``LaserEncoderPipeline``; loaded
                      automatically if ``None``.
        encoder_tgt:  Pre-loaded target ``LaserEncoderPipeline``; loaded
                      automatically if ``None``.

    Returns:
        Annotated ``datasets.Dataset`` with an added score column.
    """
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
