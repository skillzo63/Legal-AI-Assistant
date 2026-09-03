"""Build the TurboVec index and metadata from the HuggingFace dataset."""

import hashlib
import json
import logging
from typing import Any

import numpy as np
from datasets import load_dataset
from turbovec import TurboQuantIndex

from rag.config import settings
from rag.embeddings import get_embedding

logger = logging.getLogger(__name__)


def _hash(text: str) -> str:
    """Stable content id used for dedup and citation."""
    return hashlib.md5(text.encode()).hexdigest()


def build_index() -> None:
    """Embed the first N dataset questions and write index + metadata files.

    Embeds the question text only (matches how queries are embedded at search
    time) and stores the full record as JSON metadata alongside the quantized
    TurboVec index. Output paths come from ``INDEX_INDEX_PATH`` /
    ``INDEX_METADATA_PATH``.
    """
    cfg = settings.index
    logger.info("Loading open-australian-legal-qa (first %d records)", cfg.max_records)
    dataset = load_dataset("isaacus/open-australian-legal-qa", split="train")

    metadata: list[dict[str, Any]] = []
    vectors: list[list[float]] = []

    for i, record in enumerate(dataset):
        if i >= cfg.max_records:
            break
        vectors.append(get_embedding(record.get("question")))
        metadata.append(
            {
                "id": _hash(record.get("question")),
                "question": record.get("question"),
                "answer": record.get("answer"),
                "source_name": record.get("source", {})["citation"],
                "source_url": record.get("source", {})["url"],
            }
        )
        if (i + 1) % 10 == 0:
            logger.info("  Embedded %d/%d records...", i + 1, cfg.max_records)

    index = TurboQuantIndex(dim=settings.embedding.dim, bit_width=4)
    index.add(np.array(vectors, dtype=np.float32))
    index.write(cfg.index_path)

    with open(cfg.metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    logger.info("Index saved: %s", cfg.index_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_index()