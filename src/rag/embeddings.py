"""Gemini embedding access with batching, memoization, and retries."""

import functools
import logging
import time
from typing import Any

from google import genai
from google.genai import types as genai_types

from rag.config import settings
from rag.errors import EmbeddingError
from rag.retry import retry_on_exception

logger = logging.getLogger(__name__)

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
    return list(result.embeddings[0].values or [])


def get_embedding(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
    """Embed a single text with Gemini, retrying transient failures.

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
            f"Gemini embedding failed for model {settings.embedding.model!r}: {exc}"
        ) from exc


@retry_on_exception()
def _embed_batch(contents: list[Any], task_type: str) -> list[list[float]]:
    result = _get_client().models.embed_content(
        model=settings.embedding.model,
        contents=contents,
        config=genai_types.EmbedContentConfig(task_type=task_type),
    )
    assert result.embeddings is not None, "Gemini returned no embeddings in batch"
    vectors: list[list[float]] = []
    for emb in result.embeddings:
        assert emb.values is not None, "Gemini returned an empty vector"
        vectors.append(list(emb.values))
    return vectors


def get_embeddings_batch(
    texts: list[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
    batch_size: int = 50,
    min_interval: float = 0.5,
) -> list[list[float]]:
    """Embed multiple texts in batches with pacing.

    Args:
        texts: Texts to embed, in order.
        task_type: Gemini task hint for every text.
        batch_size: Texts per API call.
        min_interval: Minimum seconds between API calls.

    Returns:
        One embedding per input text, same order.

    Raises:
        EmbeddingError: Any batch failed after retries.
    """
    vectors: list[list[float]] = []
    total = len(texts)

    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        contents = [
            genai_types.Content(parts=[genai_types.Part.from_text(text=t)]) for t in batch
        ]
        t0 = time.perf_counter()
        try:
            vectors.extend(_embed_batch(contents, task_type))
        except Exception as exc:
            raise EmbeddingError(
                f"Gemini batch embedding failed at batch {i // batch_size + 1}: {exc}"
            ) from exc

        # Pacing keeps free-tier request rates happy.
        elapsed = time.perf_counter() - t0
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        processed = i + len(batch)
        if processed % 100 < batch_size or processed == total:
            logger.info(
                "  Progress: embedded %d/%d chunks (%.1f%%)...",
                processed,
                total,
                (processed / total) * 100 if total else 0.0,
            )

    return vectors


@functools.lru_cache(maxsize=256)
def get_embedding_cached(text: str, task_type: str = "RETRIEVAL_QUERY") -> tuple[float, ...]:
    """Memoized embedding as a hashable tuple.

    Defaults to RETRIEVAL_QUERY because its callers embed queries;
    indexing goes through ``get_embeddings_batch`` instead.
    """
    return tuple(get_embedding(text, task_type))
