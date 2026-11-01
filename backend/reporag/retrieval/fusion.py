"""Reciprocal Rank Fusion (Cormack et al., 2009)."""

from __future__ import annotations

from collections.abc import Hashable, Sequence

RRF_K = 60


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Hashable]], k: int = RRF_K
) -> list[tuple[Hashable, float]]:
    """Merge several ranked lists into one.

    Each item scores sum(1 / (k + rank)) over the lists it appears in (rank is
    1-based). Only ranks matter, not raw scores, so BM25 scores and cosine
    similarities, which live on different scales, can be combined without
    normalization. `k` damps the advantage of the very top ranks.
    Ties are broken by first appearance, which keeps the output deterministic.
    """
    scores: dict[Hashable, float] = {}
    first_seen: dict[Hashable, int] = {}
    order = 0
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
            if item not in first_seen:
                first_seen[item] = order
                order += 1
    return sorted(scores.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))
