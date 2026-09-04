"""Tests for nested pydantic-settings config."""

import pytest
from pydantic import ValidationError

from rag.config import (
    EmbeddingSettings,
    LLMSettings,
    RetrievalSettings,
    Settings,
)


def test_defaults_match_original_constants() -> None:
    """Section defaults must mirror the constants the original code hardcoded."""
    s = Settings()
    assert s.retrieval.threshold == 0.5
    assert s.retrieval.top_k == 3
    assert s.llm.model == "qwen/qwen3.8-27b"
    assert s.llm.temperature_legal == 0.0
    assert s.llm.temperature_casual == 0.6
    assert s.embedding.dim == 3072
    assert s.index.max_records == 500


def test_env_override_via_section_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """RETRIEVAL_THRESHOLD env var overrides the nested retrieval section."""
    monkeypatch.setenv("RETRIEVAL_THRESHOLD", "0.9")
    monkeypatch.setenv("LLM_MODEL", "llama-3.1-8b-instant")
    s = Settings()
    assert s.retrieval.threshold == 0.9
    assert s.llm.model == "llama-3.1-8b-instant"


def test_invalid_type_fails_at_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-numeric EMBEDDING_DIM must fail fast, not at request time."""
    monkeypatch.setenv("EMBEDDING_DIM", "not-a-number")
    with pytest.raises(ValidationError):
        EmbeddingSettings()


def test_sections_are_independent(monkeypatch: pytest.MonkeyPatch) -> None:
    """An LLM_* var must not leak into the retrieval section."""
    monkeypatch.setenv("LLM_THRESHOLD", "0.99")  # wrong prefix on purpose
    r = RetrievalSettings()
    assert r.threshold == 0.5


def test_llm_temperature_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Out-of-range temperatures must be rejected — they're silent quality bugs."""
    monkeypatch.setenv("LLM_TEMPERATURE_LEGAL", "1.7")
    with pytest.raises(ValidationError):
        LLMSettings()