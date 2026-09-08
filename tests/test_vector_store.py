"""Tests for TurboVec and FAISS vector stores."""

import os
import tempfile

import numpy as np
import pytest

from rag.vector_store import (
    create_vector_store,
    load_vector_store,
)


@pytest.mark.parametrize("backend", ["turbovec", "faiss"])
def test_vector_store_add_search(backend):
    dim = 64
    n = 20
    store = create_vector_store(backend=backend, dim=dim)
    assert store.name == backend
    assert store.dim == dim
    assert len(store) == 0

    vecs = np.random.randn(n, dim).astype(np.float32)
    store.add(vecs)
    assert len(store) == n

    # Search with first vector
    scores, indices = store.search(vecs[0], top_k=3)
    assert len(scores) == 3
    assert len(indices) == 3
    # Top match should be the query vector itself (index 0)
    assert indices[0] == 0
    assert scores[0] > 0.95


@pytest.mark.parametrize("backend,ext", [("turbovec", ".tv"), ("faiss", ".faiss")])
def test_vector_store_save_load(backend, ext):
    dim = 32
    n = 10
    store = create_vector_store(backend=backend, dim=dim)
    vecs = np.random.randn(n, dim).astype(np.float32)
    store.add(vecs)

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
        tmp_path = f.name

    try:
        store.save(tmp_path)
        loaded = load_vector_store(backend=backend, path=tmp_path, dim=dim)
        assert loaded.name == backend
        assert len(loaded) == n
        scores, indices = loaded.search(vecs[2], top_k=1)
        assert indices[0] == 2
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_memory_compression():
    dim = 1024
    n = 100
    faiss_store = create_vector_store("faiss", dim=dim)
    turbovec_store = create_vector_store("turbovec", dim=dim, bit_width=4)

    vecs = np.random.randn(n, dim).astype(np.float32)
    faiss_store.add(vecs)
    turbovec_store.add(vecs)

    # TurboVec 4-bit should be 8x smaller in memory than FAISS FP32
    assert turbovec_store.size_bytes() * 8 == faiss_store.size_bytes()
