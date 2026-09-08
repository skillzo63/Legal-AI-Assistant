"""Build legal-structural chunks and index with TurboVec and NumPy."""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any

import numpy as np
from datasets import load_dataset

from rag.chunker import LegalChunk, LegalStructuralChunker
from rag.config import settings
from rag.embeddings import get_embeddings_batch
from rag.vector_store import FAISSVectorStore, TurboVecVectorStore

logger = logging.getLogger(__name__)


def build_index(
    max_records: int | None = None,
    use_chunking: bool = True,
) -> tuple[int, int]:
    """Extract legal chunks from source documents, embed, and build dual vector indices.

    Args:
        max_records: Number of dataset records to process. If None, uses settings.index.max_records.
                     Pass 0 or 2124 to process the full dataset.
        use_chunking: When True, chunk the raw source.text with legal boundaries.
                      When False, fall back to embedding the question text directly.

    Returns:
        Tuple of (num_records_processed, num_chunks_indexed).
    """
    cfg = settings.index
    limit: int | None = max_records if max_records is not None else cfg.max_records
    if limit is not None and limit <= 0:
        limit = None

    logger.info(
        "Loading dataset 'isaacus/open-australian-legal-qa' (target records: %s)...",
        limit or "ALL (2,124)",
    )
    dataset = load_dataset("isaacus/open-australian-legal-qa", split="train")
    total_available = len(dataset)
    process_count = min(limit, total_available) if limit else total_available

    chunker = LegalStructuralChunker(
        max_chars=settings.chunk.max_chars,
        min_chars=settings.chunk.min_chars,
        overlap_chars=settings.chunk.overlap_chars,
    )

    metadata: list[dict[str, Any]] = []
    chunks: list[LegalChunk] = []

    logger.info("Extracting chunks from %d source records...", process_count)
    for i in range(process_count):
        record = dataset[i]
        src_dict = record.get("source", {}) or {}
        q = record.get("question", "") or ""
        a = record.get("answer", "") or ""

        if use_chunking and src_dict.get("text"):
            doc_chunks = chunker.chunk_document(
                doc_id=i,
                source_dict=src_dict,
                start_chunk_id=len(chunks),
            )
        else:
            # Fallback for question-only mode or missing source text
            doc_chunks = [
                LegalChunk(
                    chunk_id=len(chunks),
                    doc_id=i,
                    citation=src_dict.get("citation", "") or "",
                    url=src_dict.get("url", "") or "",
                    jurisdiction=src_dict.get("jurisdiction", "") or "",
                    doc_type=src_dict.get("type", "") or "",
                    clean_text=q,
                    injected_text=q,
                )
            ]

        for c in doc_chunks:
            chunks.append(c)
            metadata.append(
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "citation": c.citation,
                    "source_name": c.citation,
                    "url": c.url,
                    "source_url": c.url,
                    "jurisdiction": c.jurisdiction,
                    "type": c.doc_type,
                    "para_markers": c.para_markers,
                    "text": c.clean_text,
                    "clean_text": c.clean_text,
                    "injected_text": c.injected_text,
                    "question": q,
                    "answer": a,
                }
            )

    logger.info(
        "Produced %d chunks from %d documents (avg %.2f chunks/doc).",
        len(chunks),
        process_count,
        len(chunks) / max(1, process_count),
    )

    # Embedding chunks
    logger.info("Generating embeddings for %d chunks via batched Gemini API...", len(chunks))
    texts_to_embed = [c.injected_text if c.injected_text else c.clean_text for c in chunks]
    vectors = get_embeddings_batch(texts_to_embed, task_type="RETRIEVAL_DOCUMENT", batch_size=50)

    vec_array = np.array(vectors, dtype=np.float32)

    # 1. Build and persist TurboVec index (4-bit TurboQuant)
    logger.info("Building TurboVec 4-bit quantized index...")
    tv_store = TurboVecVectorStore(dim=settings.embedding.dim, bit_width=4)
    tv_store.add(vec_array)
    tv_store.save(cfg.index_path)

    # 2. Build and persist FAISS index (FP32 exact)
    logger.info("Building FAISS FP32 index...")
    faiss_store = FAISSVectorStore(dim=settings.embedding.dim)
    faiss_store.add(vec_array)
    faiss_store.save(cfg.faiss_index_path)

    # 3. Persist metadata
    logger.info("Writing metadata to %s...", cfg.metadata_path)
    with open(cfg.metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Compare sizes on disk
    tv_size = os.path.getsize(cfg.index_path) if os.path.exists(cfg.index_path) else 0
    faiss_size = (
        os.path.getsize(cfg.faiss_index_path) if os.path.exists(cfg.faiss_index_path) else 0
    )
    ratio = (faiss_size / tv_size) if tv_size > 0 else 0

    logger.info("=== Index Build Complete ===")
    logger.info("  Documents processed: %d", process_count)
    logger.info("  Chunks indexed:     %d", len(chunks))
    logger.info("  TurboVec index:     %s (%.2f KB)", cfg.index_path, tv_size / 1024)
    logger.info("  FAISS index:        %s (%.2f KB)", cfg.faiss_index_path, faiss_size / 1024)
    logger.info("  Compression factor: %.1fx smaller via TurboQuant vs FAISS FP32", ratio)

    return process_count, len(chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index legal documents into TurboVec & FAISS.")
    parser.add_argument(
        "--max-records", type=int, default=None, help="Number of records to process (default: all)"
    )
    parser.add_argument(
        "--no-chunking", action="store_true", help="Disable source document chunking"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    build_index(max_records=args.max_records, use_chunking=not args.no_chunking)