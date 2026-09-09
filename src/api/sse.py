"""SSE event formatting and the chat stream generator.

Event protocol spoken by POST /chat:

    event: retrieving           -> retrieval phase started
    event: mode                 -> {"mode": "legal"|"casual"}
    event: token                -> {"text": "..."} per generated chunk
    event: done                 -> {"answer": "...", "mode": "..."}
    event: error                -> {"error_type": "...", "message": "..."}
"""

import json
import logging
import time
from collections.abc import AsyncIterator

from rag.errors import EmbeddingError, LLMError
from rag.prompts import LLM_UNAVAILABLE_MESSAGE, RETRIEVAL_UNAVAILABLE_MESSAGE
from rag.prompts import route_mode as _route_mode
from rag.rewrite import rewrite_query

from .metrics import record_chat_duration, record_error, record_mode

logger = logging.getLogger(__name__)


def format_sse_event(event: str, data: dict[str, str] | str) -> str:
    """Render one SSE frame: event line + JSON data line + blank separator."""
    payload = json.dumps(data) if isinstance(data, dict) else json.dumps({"text": data})
    return f"event: {event}\ndata: {payload}\n\n"


def _messages_for_api(history: list[dict[str, str]]) -> list[dict[str, str]]:
    """Convert ChatRequest history entries into rewrite-compatible messages."""
    return [{"role": m["role"], "content": m["content"]} for m in history]


async def stream_answer(
    query: str,
    history: list[dict[str, str]],
    retriever: object,
    llm: object,
    generate: object,
) -> AsyncIterator[str]:
    """Yield the SSE frames for one chat turn, start to finish.

    Phase order mirrors the pipeline: rewrite (best-effort), retrieve,
    route, generate. Provider failures become ``error`` events carrying the
    same user-facing degradation copy the Streamlit app shows, never a bare
    500 mid-stream or a silent empty answer.

    Args:
        query: The raw user message for this turn.
        history: Caller-owned prior turns (role/content dicts).
        retriever: Loaded HybridRetriever.
        llm: Groq client, used for the rewrite call.
        generate: Callable(llm, payload, temperature) -> async iterator of
            token strings; injected so tests can fake token streaming.
    """
    start = time.perf_counter()
    try:
        yield format_sse_event("retrieving", {"phase": "rewrite"})
        try:
            standalone = rewrite_query(llm, query, _messages_for_api(history))
        except LLMError:
            standalone = query  # best-effort: raw query beats a failed turn
        # mypy: retriever/generate are object-typed; the call is duck-typed here
        results = retriever.search(standalone)  # type: ignore[attr-defined]
    except (EmbeddingError, LLMError) as exc:
        logger.warning("retrieval outage: %s", exc)
        record_error("embedding_outage")
        yield format_sse_event(
            "error", {"error_type": "embedding_outage", "message": RETRIEVAL_UNAVAILABLE_MESSAGE}
        )
        return

    mode_msg, temperature = _route_mode(results)
    mode = "legal" if results else "casual"
    record_mode(mode)
    yield format_sse_event("mode", {"mode": mode})

    payload = _messages_for_api(history)
    payload.append({"role": "system", "content": mode_msg["content"]})
    payload.append({"role": "user", "content": query})

    answer_parts: list[str] = []
    try:
        async for delta in generate(llm, payload, temperature):  # type: ignore[call-arg]
            if delta:
                answer_parts.append(delta)
                yield format_sse_event("token", {"text": delta})
    except Exception as exc:
        logger.warning("generation failure: %s", exc)
        record_error("llm_outage")
        yield format_sse_event(
            "error", {"error_type": "llm_outage", "message": LLM_UNAVAILABLE_MESSAGE}
        )
        return

    record_chat_duration(time.perf_counter() - start)
    yield format_sse_event("done", {"answer": "".join(answer_parts), "mode": mode})
