"""Dense retrieval over the TurboVec quantized index."""

import json
from pathlib import Path
from typing import Any

import numpy as np
from turbovec import TurboQuantIndex

from rag.config import settings
from rag.embeddings import get_embedding_cached


class LegalRetriever:
    """Retrieves the most relevant legal Q&A entries for a query."""

    def __init__(self, index: TurboQuantIndex, metadata: list[dict[str, Any]]) -> None:
        """Store an already-loaded index and its metadata.

        Use :meth:`load` for the normal path; direct construction exists so
        tests (and later the hybrid retriever) can inject fakes.
        """
        self.index = index
        self.metadata = metadata

    @classmethod
    def load(
        cls, index_path: str | Path | None = None, metadata_path: str | Path | None = None
    ) -> "LegalRetriever":
        """Load a retriever from the files written by ``rag.indexer``.

        Args:
            index_path: Path to the TurboVec index; defaults to ``INDEX_INDEX_PATH``.
            metadata_path: Path to the JSON metadata; defaults to ``INDEX_METADATA_PATH``.

        Returns:
            A ready-to-search retriever.
        """
        cfg = settings.index
        index = TurboQuantIndex.load(path=str(index_path or cfg.index_path))
        metadata = json.loads(
            Path(metadata_path or cfg.metadata_path).read_text(encoding="utf-8")
        )
        return cls(index, metadata)

    def search(
        self,
        query_text: str,
        k: int | None = None,
        threshold: float | None = None,
    ) -> list[dict[str, Any]] | None:
        """Search the index for entries similar to ``query_text``.

        Args:
            query_text: Raw user query (embedded via the cached Gemini client).
            k: Number of neighbors to fetch; defaults to ``RETRIEVAL_TOP_K``.
            threshold: Minimum cosine score to keep; defaults to ``RETRIEVAL_THRESHOLD``.

        Returns:
            Scored entries sorted by the index, or ``None`` when nothing
            clears the threshold (the caller's signal to switch to casual mode).
        """
        cfg = settings.retrieval
        k = k if k is not None else cfg.top_k
        threshold = threshold if threshold is not None else cfg.threshold

        query_vec = get_embedding_cached(query_text)
        scores, indices = self.index.search(np.array([query_vec], dtype=np.float32), k=k)

        results: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if round(float(score), 4) >= threshold:
                entry = self.metadata[int(idx)]
                results.append(
                    {
                        "id": entry["id"],
                        "score": round(float(score), 4),
                        "question": entry["question"],
                        "answer": entry["answer"],
                        "source": entry["source_name"],
                        "url": entry["source_url"],
                    }
                )
        return results or None