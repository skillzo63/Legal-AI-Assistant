"""Vector store abstraction supporting TurboVec (4-bit TurboQuant) and FAISS (FP32 baseline)."""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Literal

import faiss
import numpy as np
import turbovec

from rag.errors import RetrievalError

VectorBackend = Literal["turbovec", "faiss"]


class VectorStore(abc.ABC):
    """Abstract interface for dense vector indexing and similarity search."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Name of the vector backend."""

    @property
    @abc.abstractmethod
    def dim(self) -> int:
        """Vector dimensionality."""

    @abc.abstractmethod
    def __len__(self) -> int:
        """Number of vectors currently indexed."""

    @abc.abstractmethod
    def add(self, vectors: np.ndarray) -> None:
        """Add float32 vectors to the index. Vectors are normalized to unit length."""

    @abc.abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """Search nearest vectors.

        Args:
            query_vector: 1D or 2D array of shape (dim,) or (1, dim).
            top_k: Number of nearest neighbors to return.

        Returns:
            Tuple of (scores, indices), each 1D array of length top_k.
        """

    @abc.abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist index to disk."""

    @classmethod
    @abc.abstractmethod
    def load(cls, path: str | Path, dim: int) -> VectorStore:
        """Load index from disk."""

    @abc.abstractmethod
    def size_bytes(self) -> int:
        """Approximate in-memory footprint of vector index."""


class FAISSVectorStore(VectorStore):
    """Exact inner product search via FAISS IndexFlatIP (32-bit floating point)."""

    def __init__(self, dim: int, index: faiss.Index | None = None) -> None:
        self._dim = dim
        self._index = index or faiss.IndexFlatIP(dim)

    @property
    def name(self) -> str:
        return "faiss"

    @property
    def dim(self) -> int:
        return self._dim

    def __len__(self) -> int:
        return self._index.ntotal

    def add(self, vectors: np.ndarray) -> None:
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        faiss.normalize_L2(vecs)
        self._index.add(vecs)

    def search(self, query_vector: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        q = np.ascontiguousarray(query_vector, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        faiss.normalize_L2(q)
        actual_k = min(top_k, max(1, self._index.ntotal))
        scores, indices = self._index.search(q, actual_k)
        return scores[0], indices[0]

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(p))

    @classmethod
    def load(cls, path: str | Path, dim: int) -> FAISSVectorStore:
        p = Path(path)
        if not p.exists():
            raise RetrievalError(f"FAISS index file not found: {p}")
        index = faiss.read_index(str(p))
        return cls(dim=dim, index=index)

    def size_bytes(self) -> int:
        return self._index.ntotal * self._dim * 4


class TurboVecVectorStore(VectorStore):
    """Memory-efficient similarity search using Google TurboQuant 4-bit quantization."""

    def __init__(
        self,
        dim: int,
        bit_width: int = 4,
        index: turbovec.TurboQuantIndex | None = None,
    ) -> None:
        self._dim = dim
        self._bit_width = bit_width
        self._index = index or turbovec.TurboQuantIndex(dim=dim, bit_width=bit_width)

    @property
    def name(self) -> str:
        return "turbovec"

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def bit_width(self) -> int:
        return self._bit_width

    def __len__(self) -> int:
        return len(self._index)

    def add(self, vectors: np.ndarray) -> None:
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms
        self._index.add(vecs)
        self._index.prepare()

    def search(self, query_vector: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        q = np.ascontiguousarray(query_vector, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        norms = np.linalg.norm(q, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        q = q / norms
        actual_k = min(top_k, max(1, len(self._index)))
        scores, indices = self._index.search(q, actual_k)
        return scores[0], indices[0]

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._index.write(str(p))

    @classmethod
    def load(cls, path: str | Path, dim: int) -> TurboVecVectorStore:
        p = Path(path)
        if not p.exists():
            raise RetrievalError(f"TurboVec index file not found: {p}")
        index = turbovec.TurboQuantIndex.load(str(p))
        index.prepare()
        bit_w = getattr(index, "bit_width", 4)
        return cls(dim=dim, bit_width=bit_w, index=index)

    def size_bytes(self) -> int:
        return int(len(self._index) * self._dim * (self._bit_width / 8.0))


def create_vector_store(
    backend: VectorBackend = "turbovec",
    dim: int = 3072,
    bit_width: int = 4,
) -> VectorStore:
    """Instantiate a new empty vector store."""
    if backend == "turbovec":
        return TurboVecVectorStore(dim=dim, bit_width=bit_width)
    if backend == "faiss":
        return FAISSVectorStore(dim=dim)
    raise ValueError(f"Unknown vector backend: {backend!r}. Choose 'turbovec' or 'faiss'.")


def load_vector_store(
    backend: VectorBackend,
    path: str | Path,
    dim: int = 3072,
) -> VectorStore:
    """Load a vector store from disk."""
    if backend == "turbovec":
        return TurboVecVectorStore.load(path=path, dim=dim)
    if backend == "faiss":
        return FAISSVectorStore.load(path=path, dim=dim)
    raise ValueError(f"Unknown vector backend: {backend!r}. Choose 'turbovec' or 'faiss'.")
