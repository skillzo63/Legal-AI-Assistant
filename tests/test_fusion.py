"""Tests for Reciprocal Rank Fusion."""

from rag.fusion import reciprocal_rank_fusion


def test_doc_in_both_lists_beats_doc_in_one() -> None:
    """A doc ranked mid in both lists should beat a doc that tops only one."""
    dense = [10, 20, 30]  # doc 20 is 2nd here
    bm25 = [40, 20, 50]  # doc 20 is 2nd here; doc 40 tops this list only
    fused = dict(reciprocal_rank_fusion([dense, bm25]))
    assert fused[20] > fused[40]


def test_agreement_ranks_first() -> None:
    """A doc ranked #1 by both lists is the overall winner."""
    ranking = reciprocal_rank_fusion([[7, 1, 2], [7, 3, 4]])
    assert ranking[0][0] == 7


def test_empty_lists_yield_empty_ranking() -> None:
    """No inputs → no results, no crash."""
    assert reciprocal_rank_fusion([[], []]) == []


def test_k_damps_top_rank_contribution() -> None:
    """Larger k shrinks the score gap between rank 0 and rank 1."""
    small_k = dict(reciprocal_rank_fusion([[1, 2]], k=1))
    large_k = dict(reciprocal_rank_fusion([[1, 2]], k=1000))
    assert (small_k[1] - small_k[2]) > (large_k[1] - large_k[2])
