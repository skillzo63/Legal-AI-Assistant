"""BM25 keyword search over the Q&A corpus.

Complements dense retrieval: BM25 matches exact terms (statute names,
section numbers) that embeddings blur into general meaning. Pure Python
via ``rank-bm25``, so there are no index files; the index is rebuilt from
metadata at load time.
"""

import re
from typing import Any

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokens. No stemming or stopwords on purpose.

    BM25's own term weighting already down-weights common words, so a
    plain word split is enough to catch the exact-term matches dense
    retrieval misses.
    """
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """In-memory BM25 index over question text."""

    def __init__(self, corpus_tokens: list[list[str]]) -> None:
        """Build the index from pre-tokenized documents.

        Args:
            corpus_tokens: One token list per document, positionally aligned
                with the retriever's metadata.
        """
        self._bm25 = BM25Okapi(corpus_tokens)

    @classmethod
    def from_metadata(cls, metadata: list[dict[str, Any]]) -> "BM25Index":
        """Build from retriever metadata, indexing injected_text, text, or question."""
        tokens = [
            _tokenize(
                entry.get("injected_text")
                or entry.get("text")
                or entry.get("question")
                or ""
            )
            for entry in metadata
        ]
        return cls(tokens)

    def search(self, query_text: str, k: int) -> list[tuple[int, float]]:
        """Return the top-``k`` documents as ``(metadata_index, score)`` pairs.

        Args:
            query_text: Raw user query.
            k: Number of documents to return.

        Returns:
            ``(index, score)`` pairs sorted by descending BM25 score. Scores
            are BM25's own scale (unbounded, not comparable to cosine); RRF
            fusion uses only the resulting rank order, never these values.
        """
        scores = self._bm25.get_scores(_tokenize(query_text))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(i, float(scores[i])) for i in top]
