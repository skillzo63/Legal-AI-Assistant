"""Gemini embedding access with retries and memoization."""

import functools

from google import genai
from google.genai import types as genai_types

from rag.config import settings
from rag.errors import EmbeddingError
from rag.retry import retry_on_exception

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    """Create the Gemini client lazily so importing this module needs no API key."""
    global _client
    if _client is None:
        _client = genai.Client()
    return _client


@retry_on_exception()
def _embed(text: str, task_type: str) -> list[float]:
    result = _get_client().models.embed_content(
        model=settings.embedding.model,
        contents=text,
        config=genai_types.EmbedContentConfig(task_type=task_type),
    )
    assert result.embeddings is not None, "Gemini returned no embeddings"
    assert result.embeddings[0].values is not None, "Gemini returned empty values"
    return list(result.embeddings[0].values)


def get_embedding(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
    """Embed text with Gemini, retrying transient failures.

    Args:
        text: Text to embed.
        task_type: Gemini task hint. Documents use RETRIEVAL_DOCUMENT;
            queries should use RETRIEVAL_QUERY for better retrieval quality.

    Returns:
        Embedding vector of length ``settings.embedding.dim``.

    Raises:
        EmbeddingError: All retry attempts failed.
    """
    try:
        return _embed(text, task_type)
    except Exception as exc:
        raise EmbeddingError(
            f"Gemini embedding failed for model {settings.embedding.model!r} "
            f"after retries: {exc}"
        ) from exc


@functools.lru_cache(maxsize=256)
def get_embedding_cached(text: str) -> tuple[float, ...]:
    """Memoized embedding as a hashable tuple.

    Raises:
        EmbeddingError: All retry attempts failed (errors are not cached).
    """
    return tuple(get_embedding(text))