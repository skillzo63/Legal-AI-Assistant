"""HTTP routes: chat SSE streaming, retrieval, health, metrics."""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from rag.config import settings
from rag.errors import EmbeddingError, LLMError
from rag.retry import retry_on_exception

from .schemas import (
    ChatRequest,
    HealthResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
)
from .sse import stream_answer
from .state import AppState

router = APIRouter()


def _state(request: Request) -> AppState:
    """Shared pipeline state; 503s before any work if startup failed."""
    state = request.app.state.state
    if state.retriever is None or state.llm is None:
        raise HTTPException(status_code=503, detail="Index not loaded")
    return state


@retry_on_exception()
def _open_groq_stream(
    llm: object, payload: list[dict[str, str]], temperature: float
) -> object:
    """Open the Groq streaming completion, retrying connection failures."""
    return llm.chat.completions.create(  # type: ignore[attr-defined]
        model=settings.llm.model,
        messages=payload,  # type: ignore[arg-type]
        stream=True,
        temperature=temperature,
        max_tokens=settings.llm.max_tokens,
    )


async def _generate(
    llm: object, payload: list[dict[str, str]], temperature: float
) -> object:
    """Async iterator of token deltas from a Groq streaming completion."""
    stream = _open_groq_stream(llm, payload, temperature)
    for chunk in stream:  # type: ignore[union-attr]
        delta = chunk.choices[0].delta.content or ""  # type: ignore[union-attr]
        yield delta


@router.post("/chat")
async def chat(body: ChatRequest, request: Request) -> StreamingResponse:
    """One chat turn as an SSE stream: retrieving, mode, tokens, done."""
    state = _state(request)
    history = [m.model_dump() for m in body.history]

    async def event_stream() -> object:
        async for frame in stream_answer(
            query=body.query,
            history=history,
            retriever=state.retriever,
            llm=state.llm,
            generate=_generate,
        ):
            yield frame

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/search", response_model=SearchResponse)
async def search(body: SearchRequest, request: Request) -> SearchResponse:
    """Retrieval-only: hybrid search + rerank, no LLM call."""
    state = _state(request)
    try:
        results = state.retriever.search(body.query, top_n=body.top_n)
    except (EmbeddingError, LLMError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    results = results or []
    hits = [
        SearchHit(
            citation=r.get("citation", ""),
            text=r.get("text", ""),
            rerank_score=r.get("rerank_score", 0.0),
            doc_id=r.get("doc_id"),
        )
        for r in results
    ]
    return SearchResponse(
        query=body.query,
        mode="legal" if hits else "casual",
        hits=hits,
    )


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Shallow liveness: process up, index loaded."""
    state: AppState = request.app.state.state
    if state.retriever is None:
        raise HTTPException(status_code=503, detail="Index not loaded")
    return HealthResponse(status="ok")
