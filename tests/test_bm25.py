"""Tests for BM25 keyword search."""

from rag.bm25 import BM25Index

_METADATA = [
    {"question": "What is the Trade Marks Act 1995?"},
    {"question": "How do I register a company in Australia?"},
    {"question": "What are the penalties under section 34?"},
]


def test_exact_term_ranks_first() -> None:
    """An exact statute-name match should rank its document first."""
    idx = BM25Index.from_metadata(_METADATA)
    results = idx.search("Trade Marks Act", k=3)
    assert results[0][0] == 0


def test_section_number_match() -> None:
    """BM25 catches literal section numbers dense retrieval blurs."""
    idx = BM25Index.from_metadata(_METADATA)
    results = idx.search("section 34 penalties", k=3)
    assert results[0][0] == 2


def test_k_limits_result_count() -> None:
    """search returns at most k documents."""
    idx = BM25Index.from_metadata(_METADATA)
    assert len(idx.search("company", k=1)) == 1
