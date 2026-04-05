"""Align a ParallelText using LASER embeddings + FastDTW.

Replaces ``notebooks/laser_alignment.ipynb``.

Pipeline
--------
1. Encode French and Mooré sentences with LASER (``fra`` / ``mos``)
2. Run FastDTW on the embeddings to find the monotonic alignment path
3. Score each aligned pair by cosine similarity
4. Optionally filter by ``--min-score``

FastDTW naturally handles different lengths (many-to-many), and the
LASER ``mos_Latn`` model gives good Mooré sentence representations.

Usage
-----
    uv run python -m moore_web.align_corpus -i parallel.json -o aligned.json
    uv run python -m moore_web.align_corpus -i parallel.json -o aligned.json --min-score 0.7

Input JSON  (ParallelText)
--------------------------
    {"french": ["sent1", "sent2", ...], "moore": ["sent1", ...], "source": "sida"}

Output JSON
-----------
    [{"fr": "...", "mo": "...", "laser_score": 0.85}, ...]
"""

from __future__ import annotations

import statistics

import msgspec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from moore_web.flatten import AlignedCorpus, ParallelText


def dtw_align(src_embeddings: np.ndarray | list, tgt_embeddings: np.ndarray | list):
    from fastdtw import fastdtw

    alignments = []

    distance, path = fastdtw(src_embeddings, tgt_embeddings, dist=cosine)
    alignments.append(path)

    return alignments


def align_from_embeddings(
    parallel: ParallelText,
    fr_embs: np.ndarray | list,
    mo_embs: np.ndarray | list,
    min_score: float = 0.0,
) -> AlignedCorpus:
    """Align using pre-computed LASER embeddings + FastDTW.

    Useful when encoding many batches: encode all sentences once externally,
    then call this per-batch with the corresponding embedding slices.
    """
    path = dtw_align(src_embeddings=fr_embs, tgt_embeddings=mo_embs)[0]

    fr_out, mo_out, scores_out = [], [], []
    for fr_idx, mo_idx in path:
        fr = parallel.french[fr_idx].strip()
        mo = parallel.moore[mo_idx].strip()
        if not fr or not mo:
            continue
        score = float(cosine_similarity(fr_embs[fr_idx].reshape(1, -1), mo_embs[mo_idx].reshape(1, -1))[0][0])
        if score >= min_score:
            fr_out.append(fr)
            mo_out.append(mo)
            scores_out.append(score)

    if scores_out:
        print(
            f"Aligned {len(scores_out)} pairs — "
            f"mean: {statistics.mean(scores_out):.3f}  "
            f"median: {statistics.median(scores_out):.3f}  "
            f"min: {min(scores_out):.3f}  max: {max(scores_out):.3f}"
        )

    return AlignedCorpus(french=fr_out, moore=mo_out, scores=scores_out, source=parallel.source)


def align(
    parallel: ParallelText,
    min_score: float = 0.0,
    laser_fr=None,
    laser_mo=None,
) -> AlignedCorpus:
    """Align French and Mooré sentences using LASER embeddings + FastDTW.

    Args:
        parallel:  Parallel sentence lists (``ParallelText``).
        min_score: Drop pairs with cosine similarity below this value.
        laser_fr:  Pre-loaded LASER encoder for French. Loaded if not provided.
        laser_mo:  Pre-loaded LASER encoder for Mooré. Loaded if not provided.

    Returns:
        :class:`~moore_web.flatten.AlignedCorpus` with equal-length lists.
    """
    from laser_encoders import LaserEncoderPipeline

    if laser_fr is None:
        print("Loading LASER French model…")
        laser_fr = LaserEncoderPipeline(lang="fra")
    if laser_mo is None:
        print("Loading LASER Mooré model…")
        laser_mo = LaserEncoderPipeline(lang="mos")

    print(f"Encoding {len(parallel.french)} French sentences…")
    fr_embs = laser_fr.encode_sentences(parallel.french, normalize_embeddings=True)

    print(f"Encoding {len(parallel.moore)} Mooré sentences…")
    mo_embs = laser_mo.encode_sentences(parallel.moore, normalize_embeddings=True)

    print("Running FastDTW alignment…")
    return align_from_embeddings(parallel, fr_embs, mo_embs, min_score=min_score)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Align a ParallelText JSON using LASER + FastDTW.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", "-i", required=True, help="Path to ParallelText JSON.")
    parser.add_argument(
        "--output",
        "-o",
        default="aligned.json",
        help="Output aligned pairs JSON (default: %(default)s).",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Keep only pairs with cosine similarity >= this value (default: keep all).",
    )
    args = parser.parse_args()

    raw = open(args.input, "rb").read()
    parallel = ParallelText.from_json(raw)
    print(f"Input: {len(parallel.french)} FR sentences, {len(parallel.moore)} MO sentences")

    pairs = align(parallel, min_score=args.min_score)

    with open(args.output, "wb") as f:
        f.write(msgspec.json.encode(pairs))
    print(f"Wrote {len(pairs.french)} pairs → {args.output}")
