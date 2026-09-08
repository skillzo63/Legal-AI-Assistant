"""Claim-level faithfulness + casual-mode + citation-precision evaluation.

A single LLM judge (default MiniMax-M3 via NVIDIA NIM) evaluates each
factual claim in a generated answer against the retrieved context:

  YES -> claim is directly supported by the retrieved context.
  NO  -> claim is unsupported or contradicted.

  faithfulness = supported_claims / total_claims

Failure accounting: casual-mode declines (legitimate: the grounding
contract doing its job) and provider errors (harness failures) are counted
separately and never enter the score. ``n_errors > 0`` means the run is
unreliable; the orchestrator exits with a distinct code for that.

Citation precision: case citations extracted from the generated answer
are checked against the retrieved context. A made-up citation is the one
failure mode claim-level faithfulness alone does not catch, and in a legal
system it is the most damaging one.
"""

import logging
import re
import time
from dataclasses import dataclass

from groq import Groq
from openai import OpenAI

from evals.config import eval_settings
from evals.golden_set import EvalSample
from rag.config import settings
from rag.hybrid import HybridRetriever
from rag.prompts import SYSTEM_PROMPT, route_mode
from rag.retry import retry_on_exception

logger = logging.getLogger(__name__)

# ── Prompts ────────────────────────────────────────────────────────────────

_CLAIM_EXTRACTION_PROMPT = """\
Read the following answer and extract every factual claim as a numbered list.
A claim is a single, atomic statement that can be independently verified.
Output ONLY the numbered list, one claim per line, no preamble, no explanation.

Answer:
{answer}"""

_JUDGE_BATCH_PROMPT = """\
Context (retrieved legal documents):
{context}

Claims to verify:
{claims}

Evaluation Rubric:
- YES: The claim is directly and explicitly supported by facts, statements, or legal tests in the context. Minor paraphrasing is allowed if the legal meaning is strictly preserved.
- NO: The claim makes assertions NOT stated in the context, extrapolates ungrounded specifics (unsupported dates, party names, citations), or contradicts the context.

For each numbered claim, evaluate strictly according to the rubric above.
Respond with a numbered list indicating YES or NO for each claim, with no extra commentary:
1. YES/NO
2. YES/NO
..."""

# Reasoning models (e.g. MiniMax-M3) can emit chain-of-thought before the
# verdict list; strip lines that talk about reasoning instead of deciding.
_VERDICT_LINE = re.compile(r"^(\d+)[.)]\s*(YES|NO)\b", re.IGNORECASE)


# ── Report dataclasses ──────────────────────────────────────────────────────


@dataclass
class FaithfulnessReport:
    """Claim-level faithfulness with explicit failure accounting."""

    faithfulness: float  # supported_claims / total_claims
    n_samples: int  # samples that reached legal mode and were judged
    n_claims: int  # total claims judged
    supported_claims: int
    min_sample_faithfulness: float = 0.0  # worst per-sample score; hides nothing
    n_casual: int = 0  # legitimate declines: retrieval returned nothing
    n_errors: int = 0  # provider failures: run unreliable if nonzero
    n_no_claims: int = 0  # samples whose answer yielded no extractable claims
    citation_precision: float = 0.0  # fraction of cited cases present in context
    n_citations: int = 0


@dataclass
class CasualModeReport:
    """How often the system correctly declines to answer off-corpus queries.

    The hand-written casual queries exist to test the grounding gate; this
    reports the fraction that fell through to casual mode as designed.
    """

    decline_rate: float
    n_samples: int
    n_errors: int


@dataclass
class ContextRelevanceReport:
    """Whether retrieved passages are relevant to the queries (1-3 judged scale)."""

    relevance_score: float  # 0.0 to 1.0 (mean rating normalized to [0, 1])
    mean_rating: float  # 1.0 to 3.0
    n_samples: int
    n_passages: int
    n_errors: int = 0


# ── Judge plumbing ──────────────────────────────────────────────────────────


@retry_on_exception(attempts=4)
def _judge_call(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float = 0.0,
) -> str:
    """Call the OpenAI-compatible judge endpoint; retry/backoff via the shared helper."""
    resp = client.chat.completions.create(
        model=model,
        messages=messages,  # type: ignore[arg-type]
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


def _extract_claims(answer: str, client: OpenAI, model: str) -> list[str]:
    """Ask the judge to decompose an answer into atomic factual claims."""
    messages = [{"role": "user", "content": _CLAIM_EXTRACTION_PROMPT.format(answer=answer)}]
    text = _judge_call(client, model, messages, max_tokens=512)
    claims: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if re.match(r"^(\d+[.)]|\*|-)\s+", line):
            claim = re.sub(r"^(\d+[.)]|\*|-)\s+", "", line).strip()
            if claim:
                claims.append(claim)
    if not claims and text.strip():
        claims = [line.strip() for line in text.splitlines() if line.strip()]
    return claims


def _parse_batch_verdicts(response_text: str, n_claims: int) -> list[bool | None]:
    """Parse numbered YES/NO verdicts.

    Returns a list of ``True``/``False``/``None``; ``None`` marks claims the
    judge never answered. Unparsed is *unknown*, not unsupported; the caller
    excludes those claims rather than scoring them as failures.
    """
    verdicts: dict[int, bool] = {}
    for line in response_text.splitlines():
        match = _VERDICT_LINE.match(line.strip())
        if match:
            verdicts[int(match.group(1))] = match.group(2).upper() == "YES"
    return [verdicts.get(i) for i in range(1, n_claims + 1)]


def _judge_claims_batch(
    claims: list[str], context: str, client: OpenAI, model: str
) -> list[bool | None]:
    """Ask the judge to evaluate all claims in one batch call.

    Raises on provider failure; the caller records an error for the sample
    instead of inventing NO verdicts.
    """
    if not claims:
        return []
    claims_text = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))
    prompt = _JUDGE_BATCH_PROMPT.format(context=context, claims=claims_text)
    text = _judge_call(
        client,
        model,
        [{"role": "user", "content": prompt}],
        max_tokens=max(512, len(claims) * 80),
    )
    return _parse_batch_verdicts(text, len(claims))


# ── Citation precision ─────────────────────────────────────────────────────

# Case citations: "Nasr v NRMA Insurance [2006] NSWSC 1018", "R v Smith (2004)",
# "Smith v Jones [2010] HCA 5". Requires a "v" between parties, which filters out
# statute names and prose.
_CITATION_RE = re.compile(
    r"\b[A-Z][A-Za-z.'’\-]+(?:\s+[A-Z][A-Za-z.'’\-]+)*\s+v\.?\s+"
    r"[A-Z][A-Za-z.'’\-]+(?:\s+[A-Z][A-Za-z.'’\-]+)*"
    r"(?:\s+\[\d{4}\]\s*\w+\s*\d+)?",
)


def _extract_citations(text: str) -> list[str]:
    """Extract case citations (Party v Party) from text, deduplicated in order."""
    seen: set[str] = set()
    out: list[str] = []
    for match in _CITATION_RE.finditer(text):
        citation = " ".join(match.group(0).split())
        if citation not in seen:
            seen.add(citation)
            out.append(citation)
    return out


def _citation_precision(answer: str, context: str) -> tuple[float, int]:
    """Fraction of case citations in the answer that appear in the context."""
    cited = _extract_citations(answer)
    if not cited:
        return 0.0, 0
    # Substring match on normalized whitespace; citations may be split across
    # passage boundaries, so match on the party names (before the year/brackets)
    # being conservative: full citation first, party pair as fallback.
    hits = 0
    for citation in cited:
        parties = citation.split("[")[0].split("(")[0].strip()
        # Trim leading connective words ("Per Nasr v ...", "In Smith v ...")
        for prefix in ("per ", "in ", "see ", "and "):
            if parties.lower().startswith(prefix):
                parties = parties[len(prefix) :].strip()
        if citation in context or parties in context:
            hits += 1
    return hits / len(cited), len(cited)


# ── Live pipeline (mirrors app.py) ──────────────────────────────────────────


def _generate_answer(
    retriever: HybridRetriever,
    llm_client: Groq,
    query: str,
) -> tuple[str | None, str | None]:
    """Run the live pipeline for one query.

    Returns:
        ``(answer, context_text)`` when retrieval succeeds (legal mode).
        ``(None, None)`` when retrieval returns nothing (casual mode).
        Raises on provider failure; an outage must not look like a decline.
    """
    results = retriever.search(query) or []
    if not results:
        return None, None  # casual mode; faithfulness undefined

    context_text = "\n\n".join(
        f"Citation: {r.get('citation') or r.get('source', '')}"
        f"\nPassage: {r.get('text') or r.get('clean_text') or r.get('answer', '')}"
        for r in results
    )
    mode_msg, temperature = route_mode(results)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        mode_msg,  # type: ignore[arg-type]
        {"role": "user", "content": query},
    ]
    completion = llm_client.chat.completions.create(
        model=settings.llm.model,
        messages=messages,  # type: ignore[arg-type]
        temperature=temperature,
        max_tokens=settings.llm.max_tokens,
    )
    answer = completion.choices[0].message.content or ""
    return answer, context_text


# ── Public API ──────────────────────────────────────────────────────────────


def eval_faithfulness(
    retriever: HybridRetriever,
    llm_client: Groq,
    judge: OpenAI,
    samples: list[EvalSample],
) -> FaithfulnessReport:
    """Claim-level faithfulness of generated answers against retrieved context.

    Per sample: run the live pipeline, extract claims with the judge, judge
    each claim YES/NO against context. Casual declines and provider errors
    are counted, never scored.

    Args:
        retriever: Loaded HybridRetriever.
        llm_client: Groq client for generating the answer under eval.
        judge: OpenAI-compatible client for the judge model.
        samples: Golden set (auto + hand-written), legal queries only
            (pass casual queries to ``eval_casual_mode`` instead).

    Returns:
        FaithfulnessReport; see the dataclass for the accounting fields.
    """
    supported_claims = total_claims = n_samples = 0
    n_casual = n_errors = n_no_claims = 0
    sample_scores: list[float] = []
    citation_scores: list[float] = []
    n_citations = 0
    judge_model = eval_settings.active_judge_model

    for i, sample in enumerate(samples):
        logger.info("[%d/%d] %s", i + 1, len(samples), sample.query[:70])

        try:
            answer, context = _generate_answer(retriever, llm_client, sample.query)
        except Exception as exc:
            logger.warning("  pipeline error (excluded): %s", exc)
            n_errors += 1
            continue
        if answer is None or context is None:
            logger.info("  casual mode, declining (faithfulness undefined)")
            n_casual += 1
            continue

        try:
            claims = _extract_claims(answer, judge, judge_model)
        except Exception as exc:
            logger.warning("  claim extraction failed (excluded): %s", exc)
            n_errors += 1
            continue
        if not claims:
            logger.info("  no claims extracted, skipping sample")
            n_no_claims += 1
            continue

        try:
            verdicts = _judge_claims_batch(claims, context, judge, judge_model)
        except Exception as exc:
            logger.warning("  judge call failed (excluded): %s", exc)
            n_errors += 1
            continue

        known = [v for v in verdicts if v is not None]
        if not known:
            logger.warning("  judge returned no parseable verdicts, skipping sample")
            n_errors += 1
            continue

        n_samples += 1
        total_claims += len(known)
        supported_claims += sum(1 for v in known if v)
        sample_scores.append(sum(1 for v in known if v) / len(known))

        prec, n_cit = _citation_precision(answer, context)
        if n_cit > 0:
            citation_scores.append(prec)
            n_citations += n_cit

        time.sleep(0.5)  # gentle pacing between judge-heavy samples

    return FaithfulnessReport(
        faithfulness=(supported_claims / total_claims) if total_claims else 0.0,
        n_samples=n_samples,
        n_claims=total_claims,
        supported_claims=supported_claims,
        min_sample_faithfulness=min(sample_scores) if sample_scores else 0.0,
        n_casual=n_casual,
        n_errors=n_errors,
        n_no_claims=n_no_claims,
        citation_precision=(
            sum(citation_scores) / len(citation_scores) if citation_scores else 0.0
        ),
        n_citations=n_citations,
    )


def eval_casual_mode(
    retriever: HybridRetriever,
    llm_client: Groq,
    samples: list[EvalSample],
) -> CasualModeReport:
    """Measure how often off-corpus queries correctly decline to answer.

    For casual queries the *correct* behavior is legal-mode refusal: retrieval
    returns nothing and the answer routes to casual mode. Decline rate is the
    fraction of casual samples where that happened: the grounding contract,
    measured.
    """
    casual = [s for s in samples if s.category == "casual"]
    declines = 0
    scored = 0
    n_errors = 0

    for i, sample in enumerate(casual):
        logger.info("[casual %d/%d] %s", i + 1, len(casual), sample.query[:70])
        try:
            results = retriever.search(sample.query)
        except Exception as exc:
            logger.warning("  retrieval error (excluded): %s", exc)
            n_errors += 1
            continue
        scored += 1
        if not results:
            declines += 1
            logger.info("  declined (correct)")
        else:
            logger.info(
                "  answered in legal mode (scores=%s)",
                [round(r.get("rerank_score", 0.0), 3) for r in results],
            )

    return CasualModeReport(
        decline_rate=declines / scored if scored else 0.0,
        n_samples=scored,
        n_errors=n_errors,
    )


# ── Context relevance ────────────────────────────────────────────────────────

_RELEVANCE_BATCH_PROMPT = """\
Query: {query}

Retrieved Legal Passages:
{passages}

Evaluation Rubric for each passage (1-3 scale):
3 - Directly Relevant: Directly discusses the legal rule, facts, parties, or issue asked in the query.
2 - Partially Relevant: Relates to the broader topic, statute, or court, but lacks specific details to answer.
1 - Irrelevant: Off-topic, unrelated proceedings, or boilerplate.

For each numbered passage, respond with its rating on a new line (e.g. '1. 3'):
1. 1, 2, or 3
2. 1, 2, or 3
..."""


def _parse_relevance_batch_ratings(response_text: str, n_passages: int) -> list[int | None]:
    """Parse numbered 1-3 ratings; unparseable entries are excluded, not defaulted."""
    scores: dict[int, int] = {}
    for line in response_text.splitlines():
        match = re.match(r"^(\d+)[.)]\s*([123])", line.strip())
        if match:
            scores[int(match.group(1))] = int(match.group(2))
    return [scores.get(i) for i in range(1, n_passages + 1)]


def eval_context_relevance(
    retriever: HybridRetriever,
    judge: OpenAI,
    samples: list[EvalSample],
    top_k: int = 3,
    threshold: float | None = None,
) -> ContextRelevanceReport:
    """Rate retrieved passages 1-3 against the query (precision signal)."""
    judge_model = eval_settings.active_judge_model
    ratings: list[int] = []
    n_samples = 0
    n_errors = 0

    eval_samples = [s for s in samples if s.category != "casual"]

    for sample in eval_samples:
        try:
            results = retriever.search(sample.query, top_n=top_k, threshold=threshold) or []
        except Exception as exc:
            logger.warning("  retrieval error (excluded): %s", exc)
            n_errors += 1
            continue
        if not results:
            continue

        passages = [
            (r.get("text") or r.get("clean_text") or r.get("answer") or "")[:1200]
            for r in results
        ]
        passages = [p for p in passages if p]
        if not passages:
            continue

        n_samples += 1
        formatted = "\n\n".join(f"[{idx + 1}] {p}" for idx, p in enumerate(passages))
        prompt = _RELEVANCE_BATCH_PROMPT.format(query=sample.query, passages=formatted)

        try:
            content = _judge_call(
                judge,
                judge_model,
                [{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0.0,
            )
            parsed = _parse_relevance_batch_ratings(content, len(passages))
        except Exception as exc:
            logger.warning("  relevance judge failed (excluded): %s", exc)
            n_errors += 1
            continue

        ratings.extend(r for r in parsed if r is not None)
        time.sleep(0.5)

    mean_val = (sum(ratings) / len(ratings)) if ratings else 0.0
    # Normalize 1..3 scale to 0..1 scale: (rating - 1) / 2
    normalized = ((mean_val - 1.0) / 2.0) if ratings else 0.0
    return ContextRelevanceReport(
        relevance_score=max(0.0, min(1.0, normalized)),
        mean_rating=round(mean_val, 2),
        n_samples=n_samples,
        n_passages=len(ratings),
        n_errors=n_errors,
    )
