"""Tests for M3 evaluation harness modules.

Targets the live paths: batch verdict parsing (including the unknown-verdict
path), the faithfulness aggregation with failure accounting, chunk-level
retrieval hit logic, citation extraction, and the threshold gate.
"""

from unittest.mock import MagicMock, patch

import pytest

from evals.config import EvalSettings
from evals.faithfulness_eval import (
    CasualModeReport,
    FaithfulnessReport,
    _citation_precision,
    _extract_citations,
    _parse_batch_verdicts,
    eval_casual_mode,
    eval_context_relevance,
    eval_faithfulness,
)
from evals.golden_set import EvalSample, build_auto_set, load_golden_set, load_hand_written
from evals.retrieval_eval import RetrievalReport, _rank_of, eval_retrieval
from evals.run_eval import _check_thresholds, _count_errors


def test_eval_settings_defaults() -> None:
    """EvalSettings reads the EVAL_ prefix and defaults to the single judge."""
    s = EvalSettings()
    assert s.golden_set_size >= 1
    assert s.random_seed == 42
    assert s.judge_base_url.startswith("https://")


def test_build_auto_set_reproducible() -> None:
    """Seeded sampling from metadata produces identical sets."""
    meta = [
        {"chunk_id": i, "doc_id": i, "question": f"Q{i}", "answer": f"A{i}"}
        for i in range(50)
    ]
    set1 = build_auto_set(meta, n=5, seed=42)
    set2 = build_auto_set(meta, n=5, seed=42)
    assert len(set1) == 5
    assert [s.expected_id for s in set1] == [s.expected_id for s in set2]
    # expected_doc_id is populated; the doc-level secondary metric needs it
    assert all(s.expected_doc_id is not None for s in set1)


def test_load_golden_set_dedups_by_query() -> None:
    """A hand-written query identical to a sampled auto query appears once."""
    meta = [
        {"chunk_id": 0, "doc_id": 0, "question": "Q0", "answer": "A0"},
        {"chunk_id": 1, "doc_id": 1, "question": "duplicated", "answer": "A1"},
    ]
    hand = [EvalSample(query="duplicated", expected_id=None, category="casual")]
    with patch("evals.golden_set.load_hand_written", return_value=hand):
        combined = load_golden_set(meta)
    # The hand-written duplicate was dropped
    assert len(combined) == 2
    assert all(s.query != "duplicated" for s in combined if s.expected_id is None)


def test_load_hand_written() -> None:
    """Hand-written queries load with expected_id=None and a category each."""
    samples = load_hand_written()
    assert len(samples) >= 10
    assert all(s.expected_id is None for s in samples)
    categories = {s.category for s in samples}
    assert "casual" in categories  # the casual-mode eval needs these


# ── Retrieval eval ──────────────────────────────────────────────────────────


def test_rank_of_finds_first_position() -> None:
    assert _rank_of(["a", "b", "c"], "b") == 2
    assert _rank_of(["a", "b"], "z") is None


def test_retrieval_eval_chunk_level_hits_and_errors() -> None:
    """Chunk-level hit = sampled chunk_id in top-k; errors excluded, not missed."""
    samples = [
        EvalSample(query="q1", expected_id=1, expected_doc_id=1),
        EvalSample(query="q2", expected_id=2, expected_doc_id=2),
        EvalSample(query="q3", expected_id=99, expected_doc_id=99),  # miss
    ]
    mock_retriever = MagicMock()

    def fake_search(query: str, top_n: int = 5, threshold: float = 0.0) -> list[dict]:
        if query == "q1":
            return [
                {"id": 1, "doc_id": 1},
                {"id": 7, "doc_id": 7},
            ]  # rank 1 hit
        if query == "q2":
            return [
                {"id": 8, "doc_id": 2},  # wrong chunk, right doc
                {"id": 2, "doc_id": 2},
            ]  # chunk hit at rank 2; doc-level hit at rank 1
        raise RuntimeError("provider down")

    mock_retriever.search.side_effect = fake_search

    report = eval_retrieval(mock_retriever, samples)
    assert report.n_samples == 2  # the errored sample was excluded, not scored
    assert report.n_errors == 1
    assert report.recall_at_1 == pytest.approx(1 / 2)  # q1 only
    assert report.recall_at_3 == pytest.approx(2 / 2)
    assert report.mrr == pytest.approx((1.0 + 0.5) / 2)
    # q2's rank-1 result was a same-doc chunk: doc-level recall is perfect
    assert report.doc_recall_at_3 == pytest.approx(2 / 2)


# ── Verdict parsing ─────────────────────────────────────────────────────────


def test_parse_batch_verdicts_basic() -> None:
    assert _parse_batch_verdicts("1. YES\n2. NO\n3. YES", 3) == [True, False, True]
    assert _parse_batch_verdicts("1) YES\n2) No\n3. YES (supported)", 3) == [True, False, True]


def test_parse_batch_verdicts_missing_is_unknown_not_false() -> None:
    """A claim the judge never answered parses as None, never as a NO."""
    assert _parse_batch_verdicts("1. YES\n3. YES", 3) == [True, None, True]


def test_parse_batch_verdicts_skips_reasoning_lines() -> None:
    """Reasoning-model chatter before the verdict list doesn't fool the parser."""
    raw = (
        "Let me check each claim against the context.\n"
        "1. The first claim mentions the Limitation Act which appears in passage 1.\n"
        "2. YES\n3. NO"
    )
    assert _parse_batch_verdicts(raw, 3) == [None, True, False]


def test_parse_batch_verdicts_all_unknown() -> None:
    assert _parse_batch_verdicts("no verdicts here", 2) == [None, None]


# ── Citation precision ──────────────────────────────────────────────────────


def test_extract_citations() -> None:
    text = (
        "In Nasr v NRMA Insurance [2006] NSWSC 1018 the court held... "
        "See also Smith v Jones and the Limitation Act 1969 (NSW)."
    )
    cited = _extract_citations(text)
    assert any("Nasr v NRMA" in c for c in cited)
    assert any("Smith v Jones" in c for c in cited)
    assert not any("Limitation" in c for c in cited)  # statutes are not cases


def test_citation_precision_fabrication() -> None:
    context = "Citation: Nasr v NRMA Insurance [2006] NSWSC 1018\nPassage: ..."
    answer = "Per Nasr v NRMA Insurance [2006] NSWSC 1018 ... and Madeup v Fake [2020] NSWSC 1"
    precision, n = _citation_precision(answer, context)
    assert n == 2
    assert precision == pytest.approx(0.5)  # one real, one fabricated


def test_citation_precision_no_citations() -> None:
    precision, n = _citation_precision("There is no citation here.", "some context")
    assert n == 0
    assert precision == 0.0


# ── Faithfulness eval ───────────────────────────────────────────────────────


def test_eval_faithfulness_aggregates_and_accounts() -> None:
    """Verdicts aggregate to the score; casual and error samples are counted,
    never scored; unknown verdicts are excluded from the denominator."""
    samples = [
        EvalSample(query="Legal 1", expected_id=1),
        EvalSample(query="Legal 2", expected_id=2),
        EvalSample(query="Legal 3 (casual decline)", expected_id=None),
        EvalSample(query="Legal 4 (provider outage)", expected_id=3),
    ]
    mock_retriever, mock_llm, mock_judge = MagicMock(), MagicMock(), MagicMock()

    with (
        patch("evals.faithfulness_eval._generate_answer") as mock_gen,
        patch("evals.faithfulness_eval._extract_claims") as mock_extract,
        patch("evals.faithfulness_eval._judge_claims_batch") as mock_batch,
        patch("evals.faithfulness_eval._citation_precision", return_value=(0.0, 0)),
    ):
        # 1: judged, 2 claims, 1 supported; 2: judged but all verdicts unknown
        # -> excluded as unreliable; 3: casual decline; 4: pipeline raises
        mock_gen.side_effect = [
            ("Answer 1", "Context 1"),
            ("Answer 2", "Context 2"),
            (None, None),
            RuntimeError("groq down"),
        ]
        mock_extract.return_value = ["Claim 1", "Claim 2"]
        mock_batch.side_effect = [[True, False], [None, None]]

        report = eval_faithfulness(mock_retriever, mock_llm, mock_judge, samples)

    assert report.n_samples == 1
    assert report.n_claims == 2
    assert report.supported_claims == 1
    assert report.faithfulness == pytest.approx(0.5)
    assert report.min_sample_faithfulness == pytest.approx(0.5)
    assert report.n_casual == 1
    assert report.n_errors == 2  # outage + all-unknown sample


def test_eval_faithfulness_min_sample_exposes_outlier() -> None:
    """One garbage answer drags min_sample_faithfulness to 0 while mean stays high."""
    samples = [EvalSample(query=f"q{i}", expected_id=i) for i in range(3)]
    mock_retriever, mock_llm, mock_judge = MagicMock(), MagicMock(), MagicMock()

    with (
        patch("evals.faithfulness_eval._generate_answer") as mock_gen,
        patch("evals.faithfulness_eval._extract_claims") as mock_extract,
        patch("evals.faithfulness_eval._judge_claims_batch") as mock_batch,
        patch("evals.faithfulness_eval._citation_precision", return_value=(0.0, 0)),
    ):
        mock_gen.return_value = ("A", "C")
        mock_extract.return_value = ["c1"]
        mock_batch.side_effect = [[True], [True], [False]]

        report = eval_faithfulness(mock_retriever, mock_llm, mock_judge, samples)

    assert report.faithfulness == pytest.approx(2 / 3)
    assert report.min_sample_faithfulness == pytest.approx(0.0)


# ── Casual-mode eval ────────────────────────────────────────────────────────


def test_eval_casual_mode_decline_rate() -> None:
    """Casual queries that retrieve nothing count as correct declines;
    a provider error is excluded and reported, never counted as a decline."""
    samples = [
        EvalSample(query="hey", category="casual"),
        EvalSample(query="weather?", category="casual"),
        EvalSample(query="unreachable legal", category="clear_legal"),  # ignored
    ]
    mock_retriever = MagicMock()
    mock_retriever.search.side_effect = [
        [],  # decline (correct)
        RuntimeError("embedding outage"),  # error, excluded
    ]
    report = eval_casual_mode(mock_retriever, MagicMock(), samples)
    assert report.n_samples == 1  # the errored sample was excluded from scoring
    assert report.n_errors == 1
    # 1 non-error sample, 1 decline
    assert report.decline_rate == pytest.approx(1.0)


# ── Thresholds and error counting ───────────────────────────────────────────


def test_check_thresholds(tmp_path: object) -> None:
    """Threshold checker flags metrics below bars and skips absent metrics."""
    from pathlib import Path

    thresh_file = Path(str(tmp_path)) / "thresholds.yaml"
    thresh_file.write_text(
        """
retrieval:
  recall_at_3: 0.50
  mrr: 0.40
faithfulness:
  min_faithfulness: 0.75
  min_citation_precision: 0.90
casual_mode:
  min_decline_rate: 0.80
""",
        encoding="utf-8",
    )

    pass_ret = RetrievalReport(
        recall_at_1=0.4, recall_at_3=0.6, recall_at_5=0.7, mrr=0.5, n_samples=10
    )
    pass_faith = FaithfulnessReport(
        faithfulness=0.85,
        n_samples=10,
        n_claims=20,
        supported_claims=17,
        citation_precision=0.95,
        n_citations=4,
    )
    pass_casual = CasualModeReport(decline_rate=1.0, n_samples=5, n_errors=0)
    assert _check_thresholds(pass_ret, pass_faith, pass_casual, thresh_file) == []

    fail_ret = RetrievalReport(
        recall_at_1=0.1, recall_at_3=0.3, recall_at_5=0.4, mrr=0.2, n_samples=10
    )
    fail_faith = FaithfulnessReport(
        faithfulness=0.60,
        n_samples=10,
        n_claims=20,
        supported_claims=12,
        citation_precision=0.5,
        n_citations=4,
    )
    fail_casual = CasualModeReport(decline_rate=0.5, n_samples=5, n_errors=0)
    failures = _check_thresholds(fail_ret, fail_faith, fail_casual, thresh_file)
    assert len(failures) == 5


def test_check_thresholds_no_citations_skips_citation_gate() -> None:
    """Answers with no citations don't trip the citation-precision gate."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        thresh_file = Path(td) / "thresholds.yaml"
        thresh_file.write_text(
            "faithfulness:\n  min_citation_precision: 0.90\n", encoding="utf-8"
        )
        faith = FaithfulnessReport(
            faithfulness=0.9, n_samples=5, n_claims=10, supported_claims=9, n_citations=0
        )
        assert _check_thresholds(None, faith, None, thresh_file) == []


def test_count_errors_aggregates() -> None:
    ret = RetrievalReport(
        recall_at_1=0.0, recall_at_3=0.0, recall_at_5=0.0, mrr=0.0, n_samples=0, n_errors=2
    )
    faith = FaithfulnessReport(
        faithfulness=0.0, n_samples=0, n_claims=0, supported_claims=0, n_errors=1
    )
    assert _count_errors(ret, faith) == 3
    assert _count_errors(None, None) == 0


# ── Context relevance ───────────────────────────────────────────────────────


def test_eval_context_relevance_filters_casual_and_unknown_ratings() -> None:
    """Casual queries skipped; unparseable ratings excluded, not defaulted to 2."""
    samples = [
        EvalSample(query="What is defamation in NSW?", category="clear_legal"),
        EvalSample(query="Hey Mike, how is Harvey?", category="casual"),
    ]
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = [
        {"text": "Defamation requires publication.", "rerank_score": 0.8}
    ]
    mock_judge = MagicMock()

    with patch("evals.faithfulness_eval._judge_call") as mock_call:
        mock_call.return_value = "Let me think.\n1. 3"
        report = eval_context_relevance(mock_retriever, mock_judge, samples)

    assert report.n_samples == 1
    assert report.n_passages == 1
    assert report.mean_rating == 3.0
    assert report.relevance_score == pytest.approx(1.0)
    assert mock_retriever.search.call_count == 1
