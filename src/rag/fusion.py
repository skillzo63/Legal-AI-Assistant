"""Reciprocal Rank Fusion (RRF) of multiple ranked result lists.

Dense and BM25 scores live on different scales (cosine vs. BM25's
unbounded term weighting), so they can't be added directly. RRF sidesteps
this by scoring each document from its *rank position* in each list:

    score(doc) = sum over lists of 1 / (k + rank)   # rank is 0-based

A document ranked highly by both methods beats one that a single method
loves and the other ignores. ``k`` damps the influence of the very top
ranks; 60 is the value from the original RRF paper (Cormack et al., 2009).
"""

RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: list[list[int]], k: int = RRF_K
) -> list[tuple[int, float]]:
    """Fuse ranked document-id lists into one ranking.

    Args:
        ranked_lists: Each inner list is document ids ordered best-first.
            Ids may appear in some lists and not others.
        k: RRF damping constant. Larger flattens the contribution of top ranks.

    Returns:
        ``(doc_id, fused_score)`` pairs sorted by descending score.
    """
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
