"""Request and response models for the API service."""

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """One turn of conversation history, in OpenAI chat format."""

    role: str = Field(pattern="^(system|user|assistant)$")
    content: str = Field(min_length=1, max_length=20_000)


class ChatRequest(BaseModel):
    """A single chat turn: the new query plus the caller-owned history."""

    query: str = Field(min_length=1, max_length=2000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=50)


class SearchRequest(BaseModel):
    """Retrieval-only query; no LLM call happens for these."""

    query: str = Field(min_length=1, max_length=2000)
    top_n: int = Field(default=1, ge=1, le=20)


class SearchHit(BaseModel):
    """One retrieved chunk as returned to the caller."""

    citation: str
    text: str
    rerank_score: float
    doc_id: int | None = None


class SearchResponse(BaseModel):
    """Result of a /search call."""

    query: str
    mode: str  # "legal" | "casual" (casual = nothing cleared the threshold)
    hits: list[SearchHit]


class HealthResponse(BaseModel):
    """Shallow liveness: process up, index loaded."""

    status: str  # "ok" | "degraded"


class ErrorPayload(BaseModel):
    """Body of an SSE error event, and of non-streaming error responses."""

    error_type: str  # e.g. "embedding_outage", "llm_outage"
    message: str  # user-facing degradation copy from rag.prompts
