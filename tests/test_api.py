"""Tests for the FastAPI service: routes, SSE stream, failure mapping.

All collaborators are faked; no network, no index files. The app is built
through create_app() with app.state.state swapped for a fake AppState.
"""

import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from rag.errors import EmbeddingError, LLMError
from src.api.main import create_app
from src.api.state import AppState


def _fake_retriever(mode: str = "legal") -> MagicMock:
    """Retriever fake: 'legal' returns one hit, 'casual' returns None."""
    retriever = MagicMock()
    if mode == "legal":
        retriever.search.return_value = [
            {
                "citation": "Nasr v NRMA [2006]",
                "text": "The insurer must justify the delay.",
                "rerank_score": 0.99,
                "doc_id": 4,
            }
        ]
    else:
        retriever.search.return_value = None
    return retriever


def _client(mode: str = "legal") -> TestClient:
    """TestClient over an app whose pipeline state is fully faked."""
    app = create_app()
    app.state.state = AppState(retriever=_fake_retriever(mode), llm=MagicMock())
    return TestClient(app)


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    """Split an SSE body into (event, data) pairs."""
    frames = []
    for block in text.strip().split("\n\n"):
        lines = block.splitlines()
        if not lines:
            continue
        event = lines[0].removeprefix("event: ").strip()
        data = json.loads(lines[1].removeprefix("data: "))
        frames.append((event, data))
    return frames


# ── Health ──────────────────────────────────────────────────────────────────


def test_health_ok() -> None:
    """Loaded index answers 200 with status ok."""
    with _client() as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_health_degraded_without_index() -> None:
    """A failed startup leaves the app up but /health answers 503."""
    app = create_app()
    app.state.state = AppState(retriever=None, llm=None)
    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 503


# ── Search ──────────────────────────────────────────────────────────────────


def test_search_legal_mode() -> None:
    """Hits above threshold map to mode=legal and SearchHit shapes."""
    with _client("legal") as client:
        resp = client.post("/search", json={"query": "insurance delay", "top_n": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "legal"
    assert body["hits"][0]["citation"] == "Nasr v NRMA [2006]"
    assert body["hits"][0]["rerank_score"] == pytest.approx(0.99)


def test_search_casual_when_nothing_clears_threshold() -> None:
    """search() returning None maps to mode=casual with empty hits, not a crash."""
    with _client("casual") as client:
        resp = client.post("/search", json={"query": "hello there"})
    assert resp.status_code == 200
    assert resp.json()["mode"] == "casual"
    assert resp.json()["hits"] == []


def test_search_provider_outage_is_503() -> None:
    """Embedding/rerank outage surfaces as 503, not a 500 crash."""
    retriever = MagicMock()
    retriever.search.side_effect = EmbeddingError("gemini down")
    app = create_app()
    app.state.state = AppState(retriever=retriever, llm=MagicMock())
    with TestClient(app) as client:
        resp = client.post("/search", json={"query": "anything"})
    assert resp.status_code == 503


def test_search_validation_rejects_bad_body() -> None:
    """Empty query or out-of-range top_n gets 422 at the boundary."""
    with _client() as client:
        assert client.post("/search", json={"query": ""}).status_code == 422
        assert client.post("/search", json={"query": "q", "top_n": 99}).status_code == 422
        assert client.post("/search", json={"query": "q", "top_n": 0}).status_code == 422


# ── Chat SSE ────────────────────────────────────────────────────────────────


def _chat(client: TestClient, query: str = "q", history: list | None = None) -> list:
    resp = client.post("/chat", json={"query": query, "history": history or []})
    assert resp.status_code == 200
    return _parse_sse(resp.text)


def test_chat_legal_stream_shape() -> None:
    """Legal turn emits retrieving, mode, token frames, then done with the answer."""
    import src.api.routes as routes

    async def gen(llm, payload, temperature):
        for delta in ("The insurer ", "must justify."):
            yield delta

    original = routes._generate
    routes._generate = gen
    try:
        with _client("legal") as client:
            frames = _chat(client)
    finally:
        routes._generate = original

    events = [e for e, _ in frames]
    assert events[0] == "retrieving"
    assert "mode" in events
    done = [d for e, d in frames if e == "done"][0]
    assert done["answer"] == "The insurer must justify."
    assert done["mode"] == "legal"
    assert "error" not in events


def test_chat_casual_mode_uses_casual_directive() -> None:
    """No retrieval hits routes to casual mode; stream still completes."""
    import src.api.routes as routes

    async def gen(llm, payload, temperature):
        yield "Nope, outside my case files."

    original = routes._generate
    routes._generate = gen
    try:
        with _client("casual") as client:
            frames = _chat(client)
    finally:
        routes._generate = original

    mode_frame = [d for e, d in frames if e == "mode"][0]
    assert mode_frame["mode"] == "casual"
    done = [d for e, d in frames if e == "done"][0]
    assert done["answer"] == "Nope, outside my case files."
    assert done["mode"] == "casual"


def test_chat_retrieval_outage_emits_error_event() -> None:
    """Embedding outage mid-turn becomes an error event with degradation copy."""
    import src.api.routes as routes
    from rag.prompts import RETRIEVAL_UNAVAILABLE_MESSAGE

    retriever = MagicMock()
    retriever.search.side_effect = EmbeddingError("gemini down")
    app = create_app()
    app.state.state = AppState(retriever=retriever, llm=MagicMock())

    async def gen(llm, payload, temperature):
        yield "should not be reached"
        raise AssertionError("generation must not run after retrieval failure")

    original = routes._generate
    routes._generate = gen
    try:
        with TestClient(app) as client:
            frames = _chat(client)
    finally:
        routes._generate = original

    error_frames = [d for e, d in frames if e == "error"]
    assert len(error_frames) == 1
    assert error_frames[0]["error_type"] == "embedding_outage"
    assert error_frames[0]["message"] == RETRIEVAL_UNAVAILABLE_MESSAGE
    assert "done" not in [e for e, _ in frames]


def test_chat_generation_failure_emits_error_event() -> None:
    """LLM outage mid-stream becomes an error event; partial tokens were sent."""
    import src.api.routes as routes
    from rag.prompts import LLM_UNAVAILABLE_MESSAGE

    async def gen(llm, payload, temperature):
        yield "partial "
        raise LLMError("groq down")

    original = routes._generate
    routes._generate = gen
    try:
        with _client("legal") as client:
            frames = _chat(client)
    finally:
        routes._generate = original

    error_frames = [d for e, d in frames if e == "error"]
    assert error_frames[0]["error_type"] == "llm_outage"
    assert error_frames[0]["message"] == LLM_UNAVAILABLE_MESSAGE


def test_chat_validation_rejects_bad_history_role() -> None:
    """A junk role in history is rejected at the boundary (422)."""
    with _client() as client:
        resp = client.post(
            "/chat",
            json={"query": "q", "history": [{"role": "banana", "content": "x"}]},
        )
    assert resp.status_code == 422


# ── Metrics ─────────────────────────────────────────────────────────────────


def test_metrics_endpoint_exposes_counters() -> None:
    """/metrics serves Prometheus text format and counts requests."""
    with _client() as client:
        client.get("/health")
        resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "legal_rag_http_requests_total" in resp.text
    assert '/health' in resp.text
