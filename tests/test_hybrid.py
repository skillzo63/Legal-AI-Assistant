"""Tests for the hybrid retriever, with all collaborators faked (no network)."""

from typing import Any

import numpy as np
import pytest

from rag import hybrid
from rag.bm25 import BM25Index

_METADATA = [
    {
        "id": "a",
        "question": "What is the Trade Marks Act 1995?",
        "answer": "It governs trade marks.",
        "source_name": "TMA",
        "source_url": "http://x/1",
    },
    {
        "id": "b",
        "question": "How do I register a company?",
        "answer": "File with ASIC.",
        "source_name": "ASIC",
        "source_url": "http://x/2",
    },
    {
        "id": "c",
        "question": "What are penalties under section 34?",
        "answer": "Fines apply.",
        "source_name": "S34",
        "source_url": "http://x/3",
    },
]


class _FakeDenseIndex:
    """Returns a fixed neighbor order regardless of the query vector."""

    def __init__(self, order: list[int]) -> None:
        self._order = order

    def search(self, _vec: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        idx = self._order[:k]
        scores = [1.0] * len(idx)
        return np.array([scores]), np.array([idx])


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub embedding and rerank so no API is called."""
    monkeypatch.setattr(hybrid, "get_embedding_cached", lambda _t: (0.0, 0.0))

    def _fake_rerank(
        _query: str, candidates: list[dict[str, Any]], top_n: int | None = None
    ) -> list[dict[str, Any]]:
        # Preserve fused order; assign descending scores so the threshold bites.
        out = []
        for rank, c in enumerate(candidates[: (top_n or len(candidates))]):
            e = dict(c)
            e["rerank_score"] = round(1.0 - rank * 0.1, 4)
            out.append(e)
        return out

    monkeypatch.setattr(hybrid, "rerank", _fake_rerank)


def _make(order: list[int]) -> hybrid.HybridRetriever:
    return hybrid.HybridRetriever(
        _FakeDenseIndex(order), _METADATA, BM25Index.from_metadata(_METADATA)
    )


def test_returns_reranked_entries() -> None:
    """A relevant query yields entries carrying a rerank_score."""
    retriever = _make([0, 1, 2])
    results = retriever.search("Trade Marks Act", top_n=3, threshold=0.0)
    assert results is not None
    assert all("rerank_score" in r for r in results)


def test_threshold_filters_weak_matches() -> None:
    """A threshold above every rerank score returns None (casual-mode signal)."""
    retriever = _make([0, 1, 2])
    assert retriever.search("anything", top_n=3, threshold=1.5) is None


def test_top_n_caps_results() -> None:
    """No more than top_n entries come back."""
    retriever = _make([0, 1, 2])
    results = retriever.search("company", top_n=1, threshold=0.0)
    assert results is not None and len(results) == 1
