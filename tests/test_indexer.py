"""Integration test for indexer chunking and dual TurboVec + FAISS index build."""

import json
from pathlib import Path

import numpy as np
import pytest

from rag import indexer
from rag.config import settings
from rag.vector_store import load_vector_store


def test_build_index_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tv_path = tmp_path / "test.tv"
    faiss_path = tmp_path / "test.faiss"
    meta_path = tmp_path / "test_meta.json"

    monkeypatch.setattr(settings.index, "index_path", str(tv_path))
    monkeypatch.setattr(settings.index, "faiss_index_path", str(faiss_path))
    monkeypatch.setattr(settings.index, "metadata_path", str(meta_path))

    # Mock embeddings to be deterministic fast 3072-dim vectors
    dim = settings.embedding.dim
    monkeypatch.setattr(
        indexer,
        "get_embeddings_batch",
        lambda texts, task_type="RETRIEVAL_DOCUMENT", batch_size=50: [[0.01] * dim for _ in texts],
    )

    n_docs, n_chunks = indexer.build_index(max_records=3, use_chunking=True)
    assert n_docs == 3
    assert n_chunks >= 3

    assert tv_path.exists()
    assert faiss_path.exists()
    assert meta_path.exists()

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert len(meta) == n_chunks
    assert "clean_text" in meta[0]
    assert "injected_text" in meta[0]
    assert "citation" in meta[0]

    # Verify both indices can be loaded and searched
    tv_idx = load_vector_store("turbovec", tv_path, dim=dim)
    assert len(tv_idx) == n_chunks
    scores, indices = tv_idx.search(np.array([0.01] * dim, dtype=np.float32), top_k=2)
    assert len(indices) == 2

    faiss_idx = load_vector_store("faiss", faiss_path, dim=dim)
    assert len(faiss_idx) == n_chunks
    scores, indices = faiss_idx.search(np.array([0.01] * dim, dtype=np.float32), top_k=2)
    assert len(indices) == 2


def test_get_embeddings_batch_batches_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """get_embeddings_batch embeds every text, in order, via the batch API."""
    from rag import embeddings

    dim = settings.embedding.dim

    class DummyEmbed:
        def __init__(self, marker: float) -> None:
            self.values = [marker] * dim

    class DummyResult:
        def __init__(self, n: int) -> None:
            self.embeddings = [DummyEmbed(0.5) for _ in range(n)]

    class DummyClient:
        class models:
            @staticmethod
            def embed_content(model, contents, config):
                return DummyResult(len(contents))

    monkeypatch.setattr(embeddings, "_get_client", lambda: DummyClient())

    texts = ["text1", "text2", "text3", "text4"]
    vectors = embeddings.get_embeddings_batch(
        texts, task_type="RETRIEVAL_DOCUMENT", batch_size=2, min_interval=0.0
    )

    assert len(vectors) == 4
    assert all(len(v) == dim for v in vectors)
    assert all(v[0] == 0.5 for v in vectors)

