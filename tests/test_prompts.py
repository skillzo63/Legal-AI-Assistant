"""Tests for prompt routing between legal and casual modes."""

from rag.prompts import (
    LLM_UNAVAILABLE_MESSAGE,
    RETRIEVAL_UNAVAILABLE_MESSAGE,
    build_legal_injection,
    route_mode,
)

RESULTS = [
    {
        "id": "abc123",
        "rerank_score": 0.9,
        "question": "What is a trademark?",
        "answer": "A sign used to distinguish goods and services.",
        "source": "Trade Marks Act 1995",
        "url": "https://example.com/tma",
    }
]


def test_legal_injection_contains_entries() -> None:
    msg = build_legal_injection(RESULTS)
    assert msg["role"] == "system"
    assert "Relevant knowledge retrieved" in msg["content"]
    assert "What is a trademark?" in msg["content"]
    assert "0.9" in msg["content"]


def test_route_mode_with_results_is_legal() -> None:
    msg, temperature = route_mode(RESULTS)
    assert "Relevant knowledge retrieved" in msg["content"]
    assert temperature == 0.0  # grounded


def test_route_mode_without_results_is_casual() -> None:
    msg, temperature = route_mode(None)
    assert "NO_LEGAL_CONTEXT" in msg["content"]
    assert temperature == 0.6  # loose


def test_route_mode_empty_list_is_casual() -> None:
    """Empty results must behave exactly like None — same fallback mode."""
    msg, _ = route_mode([])
    assert "NO_LEGAL_CONTEXT" in msg["content"]


def test_degradation_messages_exist_and_are_user_facing() -> None:
    """Degradation copy must be non-empty, actionable, non-technical."""
    for message in (LLM_UNAVAILABLE_MESSAGE, RETRIEVAL_UNAVAILABLE_MESSAGE):
        assert message
        assert "try again" in message.lower()