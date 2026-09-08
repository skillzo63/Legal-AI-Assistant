#!/usr/bin/env python3
"""Evaluation harness orchestrator.

Runs retrieval eval (chunk-level recall@{1,3,5}, MRR), casual-mode eval,
context relevance, and faithfulness eval (claim-level judge). Writes
``evals/report.json``.

Exit codes:
    0 : all thresholds passed
    1 : threshold failure (quality regression)
    2 : harness error (provider failures made the run unreliable)

Usage
-----
    python -m evals.run_eval --retrieval-only
    python -m evals.run_eval --compare-engines
    python -m evals.run_eval --samples 40
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Windows console encoding safe-guard
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dotenv import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from groq import Groq  # noqa: E402
from openai import OpenAI  # noqa: E402

from evals.config import eval_settings  # noqa: E402
from evals.faithfulness_eval import (  # noqa: E402
    CasualModeReport,
    ContextRelevanceReport,
    FaithfulnessReport,
    eval_casual_mode,
    eval_context_relevance,
    eval_faithfulness,
)
from evals.golden_set import load_golden_set  # noqa: E402
from evals.retrieval_eval import RetrievalReport, eval_retrieval  # noqa: E402
from rag.config import settings  # noqa: E402
from rag.hybrid import HybridRetriever  # noqa: E402

logger = logging.getLogger("evals")

_EVALS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EVALS_DIR.parent


# ── Helpers ─────────────────────────────────────────────────────────────────


def _print_report(
    ret: RetrievalReport | None,
    faith: FaithfulnessReport | None,
    relevance: ContextRelevanceReport | None = None,
    casual: CasualModeReport | None = None,
) -> None:
    divider = "-" * 44
    print(f"\n{'Metric':<34} {'Value':>8}")
    print(divider)
    if ret:
        print(f"{'recall@1 (chunk-level)':<34} {ret.recall_at_1:>8.3f}")
        print(f"{'recall@3 (chunk-level)':<34} {ret.recall_at_3:>8.3f}")
        print(f"{'recall@5 (chunk-level)':<34} {ret.recall_at_5:>8.3f}")
        print(f"{'MRR (chunk-level)':<34} {ret.mrr:>8.3f}")
        print(f"{'recall@3 (doc-level, ref only)':<34} {ret.doc_recall_at_3:>8.3f}")
        print(f"{'retrieval p50/p95 ms':<34} {ret.latency_p50_ms:>8.1f}/{ret.latency_p95_ms:.1f}")
    if casual and casual.n_samples:
        print(divider)
        print(f"{'casual-mode decline rate':<34} {casual.decline_rate:>8.3f}")
    if relevance and relevance.n_passages:
        print(divider)
        print(f"{'context_relevance (0-1)':<34} {relevance.relevance_score:>8.3f}")
        print(f"{'mean_rating (1-3)':<34} {relevance.mean_rating:>8.2f}")
    if faith:
        print(divider)
        print(f"{'faithfulness':<34} {faith.faithfulness:>8.3f}")
        print(f"{'min per-sample faithfulness':<34} {faith.min_sample_faithfulness:>8.3f}")
        print(f"{'citation precision':<34} {faith.citation_precision:>8.3f}")
        print(f"{'supported claims':<34} {faith.supported_claims:>8d}/{faith.n_claims}")
    print(divider)
    if ret:
        print(f"Retrieval samples   : {ret.n_samples}  |  errors: {ret.n_errors}")
    if casual and casual.n_samples:
        print(f"Casual samples      : {casual.n_samples}  |  errors: {casual.n_errors}")
    if relevance:
        print(
            f"Relevance samples   : {relevance.n_samples}"
            f"  |  passages: {relevance.n_passages}  |  errors: {relevance.n_errors}"
        )
    if faith:
        print(
            f"Faithfulness samples: {faith.n_samples}  |  casual: {faith.n_casual}"
            f"  |  no-claims: {faith.n_no_claims}  |  errors: {faith.n_errors}"
        )


def _measure_latency(index: object, query_vecs: list[np.ndarray], top_k: int = 5) -> dict[str, float]:
    """Measure query latency (mean, p50, p95, QPS) against a vector index."""
    from rag.vector_store import VectorStore

    if not query_vecs or not isinstance(index, VectorStore):
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "qps": 0.0}

    # Warm-up passes
    for _ in range(min(5, len(query_vecs))):
        index.search(query_vecs[0], top_k=top_k)

    latencies_ms: list[float] = []
    for q in query_vecs:
        t0 = time.perf_counter_ns()
        index.search(q, top_k=top_k)
        elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
        latencies_ms.append(elapsed_ms)

    latencies_ms.sort()
    n = len(latencies_ms)
    mean_val = sum(latencies_ms) / n
    p50_val = latencies_ms[int(n * 0.50)]
    p95_val = latencies_ms[min(int(n * 0.95), n - 1)]
    qps_val = (1000.0 / mean_val) if mean_val > 0 else 0.0

    return {
        "mean_ms": mean_val,
        "p50_ms": p50_val,
        "p95_ms": p95_val,
        "qps": qps_val,
    }


def _benchmark_engines(samples: list) -> None:
    """Benchmark TurboVec (4-bit TurboQuant) against FAISS (FP32 exact) side-by-side."""
    from rag.vector_store import load_vector_store

    cfg = settings.index
    tv_path = _REPO_ROOT / cfg.index_path
    faiss_path = _REPO_ROOT / cfg.faiss_index_path

    if not tv_path.exists() or not faiss_path.exists():
        logger.info("Both indices (%s and %s) required for dual-engine benchmark.", tv_path, faiss_path)
        return

    print("\n" + "=" * 72)
    print("  DUAL-ENGINE VECTOR BENCHMARK: TurboVec (4-bit) vs FAISS (FP32)")
    print("=" * 72)

    metadata: list[dict] = json.loads(
        (_REPO_ROOT / cfg.metadata_path).read_text(encoding="utf-8")
    )
    bm25 = HybridRetriever.load().bm25

    print("\n[1/2] Benchmarking TurboVec (4-bit TurboQuant)...")
    tv_index = load_vector_store("turbovec", tv_path, dim=settings.embedding.dim)
    retriever_tv = HybridRetriever(tv_index, metadata, bm25)
    rep_tv = eval_retrieval(retriever_tv, samples)

    print("\n[2/2] Benchmarking FAISS (FP32)...")
    faiss_index = load_vector_store("faiss", faiss_path, dim=settings.embedding.dim)
    retriever_faiss = HybridRetriever(faiss_index, metadata, bm25)
    rep_faiss = eval_retrieval(retriever_faiss, samples)

    from rag.embeddings import get_embedding_cached

    auto_queries = [s.query for s in samples if s.expected_id is not None][:30]
    query_vecs = [
        np.array(get_embedding_cached(q), dtype=np.float32) for q in auto_queries
    ]
    tv_lat = _measure_latency(tv_index, query_vecs, top_k=5)
    faiss_lat = _measure_latency(faiss_index, query_vecs, top_k=5)

    tv_size_kb = os.path.getsize(tv_path) / 1024.0
    faiss_size_kb = os.path.getsize(faiss_path) / 1024.0
    comp_factor = (faiss_size_kb / tv_size_kb) if tv_size_kb > 0 else 1.0

    print("\n" + "-" * 72)
    print(f"{'Metric':<30} {'FAISS (FP32)':<20} {'TurboVec (4-bit)':<20}")
    print("-" * 72)
    print(
        f"{'Index Size on Disk':<30} {faiss_size_kb / 1024:.1f} MB"
        f"{'':<6} {tv_size_kb / 1024:.1f} MB  ({comp_factor:.1f}x smaller)"
    )
    print(f"{'Recall@1':<30} {rep_faiss.recall_at_1:<20.3f} {rep_tv.recall_at_1:<20.3f}")
    print(f"{'Recall@3':<30} {rep_faiss.recall_at_3:<20.3f} {rep_tv.recall_at_3:<20.3f}")
    print(f"{'Recall@5':<30} {rep_faiss.recall_at_5:<20.3f} {rep_tv.recall_at_5:<20.3f}")
    print(f"{'MRR':<30} {rep_faiss.mrr:<20.3f} {rep_tv.mrr:<20.3f}")
    print("-" * 72)
    print("--- Latency & Throughput Benchmark ---")
    print(
        f"{'Mean Search Latency':<30} {faiss_lat['mean_ms']:.3f} ms"
        f"{'':<12} {tv_lat['mean_ms']:.3f} ms"
    )
    print(
        f"{'p50 (Median) Latency':<30} {faiss_lat['p50_ms']:.3f} ms"
        f"{'':<12} {tv_lat['p50_ms']:.3f} ms"
    )
    print(
        f"{'p95 Latency':<30} {faiss_lat['p95_ms']:.3f} ms"
        f"{'':<12} {tv_lat['p95_ms']:.3f} ms"
    )
    print(
        f"{'Throughput (QPS)':<30} {faiss_lat['qps']:.1f} q/s"
        f"{'':<12} {tv_lat['qps']:.1f} q/s"
    )
    print("-" * 72)


def _check_thresholds(
    ret: RetrievalReport | None,
    faith: FaithfulnessReport | None,
    casual: CasualModeReport | None = None,
    thresholds_path: Path | None = None,
) -> list[str]:
    """Return threshold failure messages (empty = all passed)."""
    path = thresholds_path or (_EVALS_DIR / "thresholds.yaml")
    if not path.exists():
        logger.info("No %s; skipping gate check.", path)
        return []

    thresholds: dict = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    failures: list[str] = []

    if ret is not None and ret.n_samples > 0:
        r = thresholds.get("retrieval", {})
        for key, value in (("recall_at_3", ret.recall_at_3), ("mrr", ret.mrr)):
            if key in r and value < r[key]:
                failures.append(f"{key}  {value:.3f} < {r[key]}")
    if faith is not None and faith.n_claims > 0:
        f = thresholds.get("faithfulness", {})
        target = f.get("min_faithfulness", 0.0)
        if target and faith.faithfulness < target:
            failures.append(f"faithfulness  {faith.faithfulness:.3f} < {target}")
        cit_target = f.get("min_citation_precision", 0.0)
        if cit_target and faith.n_citations > 0 and faith.citation_precision < cit_target:
            failures.append(
                f"citation_precision  {faith.citation_precision:.3f} < {cit_target}"
            )
    if casual is not None and casual.n_samples > 0:
        c = thresholds.get("casual_mode", {})
        target = c.get("min_decline_rate", 0.0)
        if target and casual.decline_rate < target:
            failures.append(f"decline_rate  {casual.decline_rate:.3f} < {target}")

    return failures


def _count_errors(
    ret: RetrievalReport | None,
    faith: FaithfulnessReport | None,
    relevance: ContextRelevanceReport | None = None,
    casual: CasualModeReport | None = None,
) -> int:
    """Total harness errors across every eval that ran."""
    return sum(
        r.n_errors for r in (ret, faith, relevance, casual) if r is not None
    )


# ── Main ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="M3 Evaluation Harness")
    parser.add_argument(
        "--samples",
        type=int,
        default=eval_settings.golden_set_size,
        help=f"Number of auto samples from metadata (default: {eval_settings.golden_set_size})",
    )
    parser.add_argument("--retrieval-only", action="store_true", help="Retrieval eval only")
    parser.add_argument("--faithfulness-only", action="store_true", help="Faithfulness eval only")
    parser.add_argument(
        "--compare-engines",
        action="store_true",
        help="Run side-by-side benchmark of TurboVec vs FAISS",
    )
    parser.add_argument(
        "--include-relevance",
        action="store_true",
        help="Also evaluate retrieved context relevance with LLM judge",
    )
    parser.add_argument(
        "--faith-samples",
        type=int,
        default=None,
        help="Max samples for faithfulness (default: all)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("Loading pipeline...")
    retriever = HybridRetriever.load()
    llm_client = Groq()
    judge = OpenAI(
        base_url=eval_settings.judge_base_url,
        api_key=eval_settings.active_judge_api_key,
        timeout=45.0,
    )

    metadata: list[dict] = json.loads(
        (_REPO_ROOT / settings.index.metadata_path).read_text(encoding="utf-8")
    )

    eval_settings.golden_set_size = args.samples
    samples = load_golden_set(metadata)
    auto_n = sum(1 for s in samples if s.expected_id is not None)
    hand_n = sum(1 for s in samples if s.expected_id is None)
    print(f"Golden set: {len(samples)} samples ({auto_n} auto + {hand_n} hand-written)\n")

    ret_report: RetrievalReport | None = None
    faith_report: FaithfulnessReport | None = None
    relevance_report: ContextRelevanceReport | None = None
    casual_report: CasualModeReport | None = None

    if args.compare_engines:
        _benchmark_engines(samples)

    if not args.faithfulness_only:
        print("=== Retrieval Eval ===")
        ret_report = eval_retrieval(retriever, samples)

        print("\n=== Casual-Mode Eval ===")
        casual_report = eval_casual_mode(retriever, llm_client, samples)

    if args.include_relevance:
        print("\n=== Context Relevance Eval ===")
        relevance_report = eval_context_relevance(retriever, judge, samples)

    if not args.retrieval_only:
        print("\n=== Faithfulness Eval ===")
        faith_eval_samples = samples if not args.faith_samples else samples[: args.faith_samples]
        # Casual queries are the casual-mode eval's job, not faithfulness's.
        faith_eval_samples = [s for s in faith_eval_samples if s.category != "casual"]
        faith_report = eval_faithfulness(retriever, llm_client, judge, faith_eval_samples)

    print("\n=== Results ===")
    _print_report(ret_report, faith_report, relevance_report, casual_report)

    report: dict[str, dict] = {}
    if ret_report:
        report["retrieval"] = {
            "recall_at_1": round(ret_report.recall_at_1, 4),
            "recall_at_3": round(ret_report.recall_at_3, 4),
            "recall_at_5": round(ret_report.recall_at_5, 4),
            "mrr": round(ret_report.mrr, 4),
            "doc_recall_at_3": round(ret_report.doc_recall_at_3, 4),
            "n_samples": ret_report.n_samples,
            "n_errors": ret_report.n_errors,
            "latency_p50_ms": ret_report.latency_p50_ms,
            "latency_p95_ms": ret_report.latency_p95_ms,
        }
    if casual_report and casual_report.n_samples:
        report["casual_mode"] = {
            "decline_rate": round(casual_report.decline_rate, 4),
            "n_samples": casual_report.n_samples,
            "n_errors": casual_report.n_errors,
        }
    if relevance_report:
        report["context_relevance"] = {
            "score": round(relevance_report.relevance_score, 4),
            "mean_rating": relevance_report.mean_rating,
            "n_samples": relevance_report.n_samples,
            "n_passages": relevance_report.n_passages,
            "n_errors": relevance_report.n_errors,
        }
    if faith_report:
        report["faithfulness"] = {
            "score": round(faith_report.faithfulness, 4),
            "min_sample": round(faith_report.min_sample_faithfulness, 4),
            "citation_precision": round(faith_report.citation_precision, 4),
            "n_citations": faith_report.n_citations,
            "supported_claims": faith_report.supported_claims,
            "n_claims": faith_report.n_claims,
            "n_samples": faith_report.n_samples,
            "n_casual": faith_report.n_casual,
            "n_no_claims": faith_report.n_no_claims,
            "n_errors": faith_report.n_errors,
        }

    out = _EVALS_DIR / "report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport written -> {out}")

    n_errors = _count_errors(ret_report, faith_report, relevance_report, casual_report)
    if n_errors > 0:
        print(f"\n[HARNESS ERROR] {n_errors} provider failure(s); run is unreliable, exiting 2.")
        return 2

    failures = _check_thresholds(ret_report, faith_report, casual_report)
    if failures:
        print("\n[FAIL] GATE FAILURES:")
        for msg in failures:
            print(f"  - {msg}")
        return 1

    print("\n[PASS] All thresholds passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
