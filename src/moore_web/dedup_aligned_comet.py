"""Deduplicate DTW-aligned parallel pairs using COMET-QE scoring.

After DTW alignment the same source or target sentence can appear in multiple
pairs.  This module detects such repetitions on **both** sides and keeps only
the highest-quality pair per connected duplicate group, as judged by
``McGill-NLP/ssa-comet-qe`` (reference-free quality estimation).

Typical usage
-------------
>>> from moore_web.dedup_aligned_comet import deduplicate_by_comet
>>> clean = deduplicate_by_comet(aligned_pairs)

When every pair already has a score (e.g. e2e with both ``--drop-duplicate``
and ``--add-comet-qe`` scores everything first), use
:func:`deduplicate_by_score`, which loads no model.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Hashable


def duplicate_groups(
    pairs: list[dict],
    src_key: str = "fr",
    mt_key: str = "mo",
    group_key: Callable[[dict], Hashable] | None = None,
) -> list[list[int]]:
    """Return the indices of each duplicate group (components of 2+ pairs).

    Two pairs are in the same group when they share the same ``src_key`` text
    **or** the same ``mt_key`` text. Connected components are built with
    union-find so that transitive duplicates (A shares src with B, B shares mt
    with C) end up together. ``group_key`` restricts matching to pairs with the
    same key, e.g. the language pair when rows mix mos-fra and mos-eng.
    """
    group_key = group_key or (lambda pair: None)
    by_text: dict[tuple, list[int]] = defaultdict(list)
    for idx, pair in enumerate(pairs):
        by_text[(group_key(pair), "src", pair[src_key])].append(idx)
        by_text[(group_key(pair), "mt", pair[mt_key])].append(idx)

    parent = list(range(len(pairs)))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for indices in by_text.values():
        for i in indices[1:]:
            parent[_find(i)] = _find(indices[0])

    components: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(pairs)):
        components[_find(idx)].append(idx)
    return [members for members in components.values() if len(members) > 1]


def deduplicate_by_score(
    pairs: list[dict],
    src_key: str = "fr",
    mt_key: str = "mo",
    score_key: str = "comet_qe",
    group_key: Callable[[dict], Hashable] | None = None,
) -> list[dict]:
    """Keep the highest ``score_key`` pair of each duplicate group; loads no model.

    Every pair in a duplicate group must already have ``score_key``. See
    :func:`duplicate_groups` for what counts as a duplicate.
    """
    indices_to_drop: set[int] = set()
    for members in duplicate_groups(pairs, src_key, mt_key, group_key):
        best = max(members, key=lambda i: pairs[i][score_key])
        indices_to_drop.update(i for i in members if i != best)

    result = [p for i, p in enumerate(pairs) if i not in indices_to_drop]
    print(f"Removed {len(indices_to_drop)} duplicate pairs. {len(result)} pairs remaining.")
    # TODO: Save dropped duplicate groups to a JSONL file for manual inspection.
    #       Each line should contain the full group (all members with their scores)
    #       so it's easy to audit whether the right pair was kept and to
    #       understand the origin of the duplicates (e.g. boilerplate, scraper
    #       re-fetches, segmentation boundary errors).
    return result


def deduplicate_by_comet(
    pairs: list[dict],
    src_key: str = "fr",
    mt_key: str = "mo",
    batch_size: int = 8,
    gpus: int | None = None,
) -> list[dict]:
    """Remove duplicate aligned pairs, keeping the highest COMET-QE score.

    Only pairs that belong to a duplicate group are scored (see
    :func:`duplicate_groups`); the rest are returned untouched.

    Args:
        pairs:      List of dicts, each with at least ``src_key`` and
                    ``mt_key`` string fields.
        src_key:    Key for the source text (default ``"fr"``).
        mt_key:     Key for the MT/target text (default ``"mo"``).
        batch_size: COMET inference batch size.
        gpus:       Number of GPUs to use (0 = CPU); ``None`` uses the GPU when
                    CUDA is available.

    Returns:
        Deduplicated list of pair dicts.  Scored pairs gain a ``"comet_qe"``
        field with their raw model score.
    """
    # TODO: Can we vectorize this to be faster?
    # is this better than google/metricx-24-hybrid-xl-v2p6 mentionned in Omnilingual MT?
    from moore_web.score_comet_qe import load_model

    groups = duplicate_groups(pairs, src_key, mt_key)
    if not groups:
        print("No duplicates found — returning original list unchanged.")
        return pairs

    dup_indices_list = sorted(i for members in groups for i in members)
    print(f"Found {len(dup_indices_list)} pairs involved in duplications. Loading COMET-QE model...")

    # Shared with --add-comet-qe (load_model is cached): keep it loaded.
    model = load_model()
    if gpus is None:
        import torch

        gpus = 1 if torch.cuda.is_available() else 0

    comet_data = [{"src": pairs[i][src_key], "mt": pairs[i][mt_key]} for i in dup_indices_list]
    output = model.predict(comet_data, batch_size=batch_size, gpus=gpus, num_workers=0)
    for rank, idx in enumerate(dup_indices_list):
        pairs[idx]["comet_qe"] = float(output.scores[rank])

    return deduplicate_by_score(pairs, src_key, mt_key, score_key="comet_qe")
