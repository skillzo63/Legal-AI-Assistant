"""Retrieval quality metrics: recall@{1,3,5} and MRR.

Uses ``threshold=0.0`` and ``top_n=5`` so all results the reranker returns
are visible regardless of the production threshold. This measures retrieval
quality independent of the grounding gate; the gate is a separate concern.

Hit criterion: the sampled record's ``chunk_id`` appears in the top-k
results (chunk-level). Doc-level recall, where any chunk of the target
document counts, is reported alongside as a secondary, friendlier number;
the chunk-level metric is the regression gate because it has headroom to fail.

Only auto samples (with a ground-truth chunk) are used; hand-written
queries have no ground-truth document in the index so recall can't be
measured for them.
"""

import logging
import time
from dataclasses import dataclass, field

from evals.golden_set import EvalSample
from rag.hybrid import HybridRetriever

logger = logging.getLogger(__name__)


@dataclass
class RetrievalReport:
    """Retrieval quality metrics over the auto eval set."""

    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    n_samples: int  # auto samples evaluated
    doc_recall_at_3: float = 0.0  # secondary: any-chunk-of-target-doc in top-3
    n_errors: int = 0  # provider failures during search, not a quality signal
    latency_p50_ms: float = field(default=0.0)
    latency_p95_ms: float = field(default=0.0)


def _rank_of(result_ids: list[object], target: object) -> int | None:
    """1-based rank of ``target`` in ``result_ids``, or None if absent."""
    for pos, rid in enumerate(result_ids, start=1):
        if rid == target:
            return pos
    return None


def eval_retrieval(retriever: HybridRetriever, samples: list[EvalSample]) -> RetrievalReport:
    """Compute chunk-level recall@{1,3,5}, MRR, and doc-level recall over auto queries.

    Args:
        retriever: Loaded HybridRetriever (dense + BM25 + rerank).
        samples: Full golden set; only those with ``expected_id`` are used.

    Returns:
        RetrievalReport. ``n_errors > 0`` means the run itself is unreliable:
        provider failures were excluded from scoring, not counted as misses.
    """
    auto_samples = [s for s in samples if s.expected_id is not None]

    hits_at: dict[int, int] = {1: 0, 3: 0, 5: 0}
    doc_hits_at_3 = 0
    reciprocal_ranks: list[float] = []
    scored = 0
    n_errors = 0
    latencies_ms: list[float] = []

    for i, sample in enumerate(auto_samples):
        logger.info("[%d/%d] %s", i + 1, len(auto_samples), sample.query[:70])
        t0 = time.perf_counter()
        try:
            results = retriever.search(sample.query, top_n=5, threshold=0.0) or []
        except Exception as exc:
            logger.warning("  retrieval error (excluded from scoring): %s", exc)
            n_errors += 1
            continue
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        scored += 1
        result_ids = [r.get("id") for r in results]
        rank = _rank_of(result_ids, sample.expected_id)
        if rank is not None:
            for k in (1, 3, 5):
                if rank <= k:
                    hits_at[k] += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

        if any(r.get("doc_id") == sample.expected_doc_id for r in results):
            doc_hits_at_3 += 1

    n = scored or 1
    latencies_ms.sort()
    p50 = latencies_ms[len(latencies_ms) // 2] if latencies_ms else 0.0
    p95 = latencies_ms[min(int(len(latencies_ms) * 0.95), len(latencies_ms) - 1)] if latencies_ms else 0.0

    return RetrievalReport(
        recall_at_1=hits_at[1] / n,
        recall_at_3=hits_at[3] / n,
        recall_at_5=hits_at[5] / n,
        mrr=sum(reciprocal_ranks) / n,
        n_samples=scored,
        doc_recall_at_3=doc_hits_at_3 / n,
        n_errors=n_errors,
        latency_p50_ms=round(p50, 1),
        latency_p95_ms=round(p95, 1),
    )
