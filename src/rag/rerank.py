"""Cross-encoder reranking via a hosted API (Cohere Rerank).

Hybrid search (dense + BM25 + RRF) casts a wide, cheap net — good recall,
rough ordering. A cross-encoder reads the query and each candidate *together*
and scores the actual relationship, which the separately-embedded dense
vectors never see. Too slow to run over the whole corpus, so it only sorts
the hybrid top-N down to the final few. Hosted (not local) to keep the
deploy image free of PyTorch; the API key comes from ``RERANK_API_KEY``.
"""

from typing import Any

import cohere

from rag.config import settings
from rag.errors import LLMError
from rag.retry import retry_on_exception

_client: cohere.Client | None = None


def _get_client() -> cohere.Client:
    """Create the Cohere client lazily so import needs no API key."""
    global _client
    if _client is None:
        _client = cohere.Client(api_key=settings.rerank.api_key)
    return _client


@retry_on_exception(exceptions=(Exception,))
def _rerank(query_text: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
    response = _get_client().rerank(
        model=settings.rerank.model,
        query=query_text,
        documents=documents,
        top_n=top_n,
    )
    return [(r.index, r.relevance_score) for r in response.results]


def rerank(
    query_text: str, candidates: list[dict[str, Any]], top_n: int | None = None
) -> list[dict[str, Any]]:
    """Reorder candidates by cross-encoder relevance, keeping the best ``top_n``.

    Args:
        query_text: The (rewritten) user query.
        candidates: Hybrid-retrieved entries; each must have a ``question`` and
            ``answer`` used to build the text scored against the query.
        top_n: How many to keep; defaults to ``RETRIEVAL_TOP_K``.

    Returns:
        The ``top_n`` candidates ordered best-first, each with a ``rerank_score``
        field added. Returns ``[]`` when given no candidates.

    Raises:
        LLMError: The rerank call failed after retries.
    """
    if not candidates:
        return []
    top_n = top_n if top_n is not None else settings.retrieval.top_k

    documents = [f"{c['question']}\n{c['answer']}" for c in candidates]
    try:
        ranked = _rerank(query_text, documents, min(top_n, len(candidates)))
    except Exception as exc:
        raise LLMError(f"Rerank failed after retries: {exc}") from exc

    reranked: list[dict[str, Any]] = []
    for idx, score in ranked:
        entry = dict(candidates[idx])
        entry["rerank_score"] = round(float(score), 4)
        reranked.append(entry)
    return reranked
