"""Deduplicate DTW-aligned parallel pairs using COMET-QE scoring.

After DTW alignment the same source or target sentence can appear in multiple
pairs.  This module detects such repetitions on **both** sides and keeps only
the highest-quality pair per connected duplicate group, as judged by
``McGill-NLP/ssa-comet-qe`` (reference-free quality estimation).

Typical usage
-------------
>>> from moore_web.dedup_aligned_comet import deduplicate_by_comet
>>> clean = deduplicate_by_comet(aligned_pairs)
"""

from __future__ import annotations

from collections import defaultdict


def deduplicate_by_comet(
    pairs: list[dict],
    src_key: str = "fr",
    mt_key: str = "mo",
    batch_size: int = 8,
    gpus: int = 0,
) -> list[dict]:
    """Remove duplicate aligned pairs, keeping the highest COMET-QE score.

    Two pairs are considered part of the same duplicate *group* when they share
    the same ``src_key`` text **or** the same ``mt_key`` text (checked on both
    sides).  Connected components are built with union-find so that transitive
    duplicates (A shares src with B, B shares mt with C) are handled correctly.
    Within each component only the pair with the highest COMET-QE score is kept.

    Args:
        pairs:      List of dicts, each with at least ``src_key`` and
                    ``mt_key`` string fields.
        src_key:    Key for the source text (default ``"fr"``).
        mt_key:     Key for the MT/target text (default ``"mo"``).
        batch_size: COMET inference batch size.
        gpus:       Number of GPUs to use (0 = CPU).

    Returns:
        Deduplicated list of pair dicts.  Scored pairs gain a ``"comet_qe"``
        field with their raw model score.
    """
    # TODO: Can we vectorize this to be faster?
    from comet import download_model, load_from_checkpoint

    src_to_indices: dict[str, list[int]] = defaultdict(list)
    mt_to_indices: dict[str, list[int]] = defaultdict(list)

    for idx, pair in enumerate(pairs):
        src_to_indices[pair[src_key]].append(idx)
        mt_to_indices[pair[mt_key]].append(idx)

    duplicate_indices: set[int] = set()
    for indices in src_to_indices.values():
        if len(indices) > 1:
            duplicate_indices.update(indices)
    for indices in mt_to_indices.values():
        if len(indices) > 1:
            duplicate_indices.update(indices)

    if not duplicate_indices:
        print("No duplicates found — returning original list unchanged.")
        return pairs

    print(f"Found {len(duplicate_indices)} pairs involved in duplications. Loading COMET-QE model...")

    model_path = download_model("McGill-NLP/ssa-comet-qe")
    model = load_from_checkpoint(model_path)

    dup_indices_list = sorted(duplicate_indices)
    comet_data = [{"src": pairs[i][src_key], "mt": pairs[i][mt_key]} for i in dup_indices_list]

    output = model.predict(comet_data, batch_size=batch_size, gpus=gpus)
    for rank, idx in enumerate(dup_indices_list):
        pairs[idx]["comet_qe"] = float(output.scores[rank])

    parent = list(range(len(pairs)))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: int, b: int) -> None:
        parent[_find(a)] = _find(b)

    for indices in src_to_indices.values():
        for i in range(1, len(indices)):
            _union(indices[0], indices[i])

    for indices in mt_to_indices.values():
        for i in range(1, len(indices)):
            _union(indices[0], indices[i])

    component: dict[int, list[int]] = defaultdict(list)
    for idx in duplicate_indices:
        component[_find(idx)].append(idx)

    indices_to_drop: set[int] = set()
    for members in component.values():
        if len(members) < 2:
            continue
        best = max(members, key=lambda i: pairs[i].get("comet_qe", -1.0))
        indices_to_drop.update(i for i in members if i != best)

    result = [p for i, p in enumerate(pairs) if i not in indices_to_drop]
    print(f"Removed {len(indices_to_drop)} duplicate pairs. {len(result)} pairs remaining.")
    return result
