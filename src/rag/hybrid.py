"""Hybrid retrieval: dense + BM25 → RRF fusion → cross-encoder rerank.

The two-stage production RAG shape. Stage one casts a wide, cheap net:
dense retrieval (meaning) and BM25 (exact terms) each rank the corpus, and
RRF fuses their rankings into a candidate pool. Stage two is the expensive,
accurate pass: a cross-encoder reranks that pool down to the final few.
The rerank score is authoritative, so the threshold applies to it: nothing
weakly relevant reaches the LLM, preserving the hard-grounding guarantee.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from turbovec import TurboQuantIndex

from rag.bm25 import BM25Index
from rag.config import settings
from rag.embeddings import get_embedding_cached
from rag.fusion import reciprocal_rank_fusion
from rag.rerank import rerank
from rag.vector_store import VectorStore, load_vector_store


class HybridRetriever:
    """Retrieves legal Q&A entries or chunks via hybrid search plus reranking."""

    def __init__(
        self,
        index: VectorStore | TurboQuantIndex | Any,
        metadata: list[dict[str, Any]],
        bm25: BM25Index,
    ) -> None:
        """Store a loaded dense index, its metadata, and a BM25 index.

        Use :meth:`load` for the normal path; direct construction lets tests
        inject fakes for all three collaborators.
        """
        self.index = index
        self.metadata = metadata
        self.bm25 = bm25

    @classmethod
    def load(
        cls, index_path: str | Path | None = None, metadata_path: str | Path | None = None
    ) -> HybridRetriever:
        """Load from the files written by ``rag.indexer``, building BM25 in memory.

        Supports both TurboVec and NumPy vector stores based on configuration.
        """
        cfg = settings.index
        backend = settings.retrieval.vector_backend
        metadata_file = Path(metadata_path or cfg.metadata_path)
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))

        if backend == "faiss":
            target_path = Path(index_path or cfg.faiss_index_path)
            index = load_vector_store("faiss", path=target_path, dim=settings.embedding.dim)
        else:
            target_path = Path(index_path or cfg.index_path)
            index = load_vector_store("turbovec", path=target_path, dim=settings.embedding.dim)

        return cls(index, metadata, BM25Index.from_metadata(metadata))

    def _dense_ranking(self, query_text: str, k: int) -> list[int]:
        """Dense top-``k`` metadata indices, best-first (no threshold here)."""
        # RETRIEVAL_QUERY: queries must be embedded with the query task type,
        # not the document one, or dense similarity is quietly degraded.
        query_vec = np.array(
            get_embedding_cached(query_text, "RETRIEVAL_QUERY"), dtype=np.float32
        )

        if isinstance(self.index, VectorStore):
            _, indices = self.index.search(query_vec, top_k=k)
            return [int(i) for i in indices]

        # Raw TurboQuantIndex or test fake
        if query_vec.ndim == 1:
            q_arr = query_vec.reshape(1, -1)
        else:
            q_arr = query_vec
        _, indices = self.index.search(q_arr, k=k)
        if isinstance(indices, np.ndarray) and indices.ndim == 2:
            return [int(i) for i in indices[0]]
        return [int(i) for i in indices]

    def search(
        self,
        query_text: str,
        top_n: int | None = None,
        threshold: float | None = None,
    ) -> list[dict[str, Any]] | None:
        """Hybrid-search then rerank, returning the best entries above threshold.

        Args:
            query_text: The (already rewritten) standalone query.
            top_n: Final entries to return; defaults to ``RETRIEVAL_TOP_K``.
            threshold: Minimum rerank score to keep; defaults to
                ``RETRIEVAL_THRESHOLD``. Applied to the cross-encoder score,
                not the dense cosine score.

        Returns:
            Reranked entries clearing the threshold, or ``None`` when none do
            (the caller's signal to switch to casual mode).
        """
        cfg = settings.retrieval
        top_n = top_n if top_n is not None else cfg.top_k
        threshold = threshold if threshold is not None else cfg.threshold
        pool = settings.rerank.candidate_pool

        dense_ranked = self._dense_ranking(query_text, pool)
        bm25_ranked = [idx for idx, _ in self.bm25.search(query_text, pool)]
        fused = reciprocal_rank_fusion([dense_ranked, bm25_ranked])

        candidates = [self._entry(doc_id) for doc_id, _ in fused[:pool]]
        reranked = rerank(query_text, candidates, top_n=top_n)

        kept = [e for e in reranked if e["rerank_score"] >= threshold]
        return kept or None

    def _entry(self, doc_id: int) -> dict[str, Any]:
        """Project a metadata record into the public result shape."""
        entry = self.metadata[doc_id]
        clean_text = entry.get("clean_text") or entry.get("text") or ""
        citation = entry.get("citation") or entry.get("source_name") or ""
        url = entry.get("url") or entry.get("source_url") or ""

        return {
            "id": entry.get("chunk_id", entry.get("id")),
            "chunk_id": entry.get("chunk_id"),
            "doc_id": entry.get("doc_id"),
            "question": entry.get("question", ""),
            "answer": entry.get("answer", ""),
            "text": clean_text,
            "injected_text": entry.get("injected_text", ""),
            "citation": citation,
            "source": citation,
            "url": url,
            "jurisdiction": entry.get("jurisdiction", ""),
            "para_markers": entry.get("para_markers", []),
        }
