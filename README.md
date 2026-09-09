# ⚖️ Mike Ross — Legal RAG Assistant

A Retrieval-Augmented Generation assistant over Australian case law, themed as Mike Ross from *Suits*. Answers are grounded in a chunked corpus of case-law source texts, not the model's own training data. When nothing relevant is retrieved for a legal question, the LLM is not called at all.

It runs a full production RAG pipeline: structural legal chunking, hybrid retrieval (dense + keyword), rank fusion, cross-encoder reranking, multi-turn query rewriting, typed error handling, a measured evaluation harness with regression gates, and a fully tested, type-checked codebase.

---

## What the pipeline handles that single-lookup RAG doesn't

A basic setup embeds the query, does one nearest-neighbour lookup, and puts the top-k into a prompt. That misses exact-term matches (case names, statute numbers), can't distinguish "close in vector space" from "actually relevant", and breaks on follow-up questions. Each row below is a gap this pipeline closes:

| Problem | Single-lookup RAG | Here |
|---------|-----------|------|
| Whole judgments vs. prompt-sized context | one doc = one vector | **legal-structural chunking** with citation/jurisdiction headers |
| Semantic *and* keyword relevance | dense only | **dense + BM25**, fused |
| Combining two rankers with different score scales | n/a | **Reciprocal Rank Fusion** (rank-based, scale-free) |
| "Near in vector space" ≠ "relevant" | cosine threshold | **cross-encoder rerank** reads query+doc together |
| Follow-ups ("what about for businesses?") | embedded as-is | **LLM query rewriting** resolves them to standalone queries |
| Provider outage | silent failure / hallucination | **typed exceptions, explicit user-facing degradation** |
| Quality regressions | undetected | **eval harness with threshold gates** (exit code 1) |

---

## Architecture

```
                        User message + conversation history
                                      │
                                      ▼
                    ┌─────────────────────────────────┐
                    │   Query rewrite (Groq LLM)       │  follow-up → standalone query
                    │   first turn → skipped           │  (best-effort; falls back to raw)
                    └──────────────────┬──────────────┘
                                       │ standalone query
                    ┌──────────────────┴──────────────────┐
                    ▼                                      ▼
        ┌───────────────────────┐              ┌───────────────────────┐
        │  Dense retrieval      │              │  BM25 keyword search  │
        │  Gemini embed +       │              │  (rank-bm25, in-mem)  │
        │  TurboVec 4-bit ANN   │              │                       │
        └───────────┬───────────┘              └───────────┬───────────┘
                    │ ranked ids                           │ ranked ids
                    └──────────────────┬───────────────────┘
                                       ▼
                    ┌─────────────────────────────────┐
                    │  Reciprocal Rank Fusion (k=60)   │  → candidate pool (top ~50)
                    └──────────────────┬──────────────┘
                                       ▼
                    ┌─────────────────────────────────┐
                    │  Cross-encoder rerank (Cohere)   │  pool → top-k, authoritative score
                    └──────────────────┬──────────────┘
                                       │ keep rerank_score ≥ 0.5
                          ┌────────────┴────────────┐
                          │  results  → Legal Mode (grounded, temp 0.0)
                          │  none     → Casual Mode (ungrounded, temp 0.6)
                          └────────────┬────────────┘
                                       ▼
                    ┌─────────────────────────────────┐
                    │   LLM generation (Mike Ross)     │
                    └─────────────────────────────────┘
```

Stage one casts a wide, cheap net (dense + BM25 → fused pool). Stage two is the expensive, accurate pass (cross-encoder rerank). The **rerank score is authoritative**, so the relevance threshold applies to it. Nothing weakly relevant reaches the LLM, which is what keeps the hard-grounding guarantee intact.

## Key design decisions

- **Legal-structural chunking.** Judgments run thousands of words; a single vector can't represent one. The chunker splits on paragraph markers and sentence boundaries (max 1000 chars, 100 overlap), and each chunk is embedded *with* a `[Citation | Jurisdiction | Para]` header so the vector carries provenance, not just prose.
- **Dual vector backends.** TurboVec (4-bit TurboQuant, ~8x smaller in memory) is the default; an exact FAISS IndexFlatIP index is built alongside it, and `python -m evals.run_eval --compare-engines` benchmarks recall and latency side by side on the same queries.
- **Hybrid retrieval.** Dense catches meaning ("detaining people" → *Ruddock v Vadarlis*); BM25 catches exact terms (case names, section numbers) that embeddings blur. Neither alone is enough for legal text.
- **RRF over score-averaging.** Cosine and BM25 scores live on incompatible scales, so averaging them is meaningless. RRF fuses by *rank position* (`Σ 1/(k+rank)`), so it never needs the raw scores to be comparable.
- **Cross-encoder rerank as the relevance gate.** A bi-encoder embeds query and doc separately; a cross-encoder reads them *together* and scores true relevance. On this corpus Cohere's scores are near-binary (relevant ≈ 1.0, irrelevant ≈ 0.0), so the `0.5` threshold sits in a wide empty gap. The value was tuned from the observed score distribution, not guessed.
- **Multi-turn query rewriting.** A retriever can't resolve "what about for businesses?" because the subject is in an earlier turn. An LLM rewrites it to a standalone query before embedding. First-turn queries skip the call.
- **Hard grounding.** With no retrieved context for a legal question, the LLM is never invoked, so it can't confidently hallucinate.
- **Typed degradation.** Provider errors (`EmbeddingError`, `LLMError`) surface as explicit user-facing messages rather than silent failures or invented answers. Rewrite failure degrades gracefully to the raw query.
- **Config, not constants.** Every tunable (models, threshold, top-k, candidate pool, temperatures) is env-driven via `pydantic-settings` with validation bounds. Nothing is hardcoded in pipeline code.
- **Classified retries.** All provider calls route through one retry helper that classifies exceptions first: auth errors fail fast, rate limits back off longer, transients retry with jitter.
- **Serving + observability.** A FastAPI service owns the pipeline: stateless `/chat` streaming over SSE, `/search` for debugging retrieval, shallow `/health`, Prometheus metrics on `/metrics` (request counts, latency histograms, legal/casual mode split, pipeline error counts). The UI is a thin client of the service; failures surface as SSE error events with user-facing degradation copy, never a bare 500 mid-stream.
- **CI with a quality gate.** Lint, type-check, and unit tests on every PR; a retrieval eval gate against a cached index fails the build on quality regressions (exit code 1), and provider outages during the run fail it loudly as harness errors (exit code 2).
- **Tested & typed.** `pytest` suite runs with zero network calls (collaborators are faked); `mypy --strict` clean; `ruff` clean.

---

## Setup

### 1. Clone & install
```bash
git clone https://github.com/skillzo63/Legal-AI-Assistant.git
cd Legal-AI-Assistant
python -m venv .venv
.venv\Scripts\activate     # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .
```

### 2. Configure API keys
Create a `.env` file in the project root:
```
GEMINI_API_KEY=your_google_gemini_api_key
GROQ_API_KEY=your_groq_api_key
RERANK_API_KEY=your_cohere_api_key
EVAL_JUDGE_API_KEY=your_nvidia_nim_api_key   # optional, for the eval judge
```

Get keys (all have free tiers):
- Gemini: https://aistudio.google.com/apikey
- Groq: https://console.groq.com/keys
- Cohere (reranker): https://dashboard.cohere.com/api-keys
- NVIDIA NIM (eval judge): https://build.nvidia.com (top-right "Get API Key")

All pipeline settings are configurable via env vars with section prefixes (`EMBEDDING_*`, `RETRIEVAL_*`, `RERANK_*`, `LLM_*`, `INDEX_*`); see [src/rag/config.py](src/rag/config.py).

### 3. Build the vector index
```bash
python -m rag.indexer
```
Downloads the [Open Australian Legal QA](https://huggingface.co/datasets/isaacus/open-australian-legal-qa) dataset, chunks each record's `source.text` on legal structural boundaries, embeds the chunks (batched, Gemini), and writes `aus_legal_qa.tv` + `aus_legal_qa.faiss` + `metadata.json`. Pass `--max-records 0` for the full dataset (2,124 documents -> ~3.6k chunks). BM25 is rebuilt in memory from the metadata at load time.

### 4. Run the API and the UI

```bash
uvicorn src.api.main:app --port 8000    # the service (start this first)
streamlit run app.py                    # the chat UI, a client of the API
```

The Streamlit app is a thin client over the FastAPI service: history lives in the browser session, every turn is a POST to `/chat` whose SSE stream renders token by token. The API itself is stateless.

Endpoints: `POST /chat` (SSE stream: retrieving → mode → tokens → done), `POST /search` (retrieval-only, no LLM call), `GET /health` (shallow liveness), `GET /metrics` (Prometheus text format: request counts, latency histograms, mode split, pipeline errors).

### 5. Run the checks
```bash
pytest && ruff check . && mypy
```

CI runs the same chain on every PR and push to main, plus a retrieval eval gate: the vector index is cached (keyed on the chunker/indexer source), a cache miss rebuilds it, and `python -m evals.run_eval --ci` fails the build on any quality regression. LLM-judge evals (faithfulness, citation precision) run locally against main, where a human decides whether the numbers moved.

### 6. Docker (optional)
```bash
docker build -t legal-rag .
docker run -p 8000:8000 -v "$PWD:/data" legal-rag
```
Multi-stage build on `python:3.12-slim`, non-root user, healthcheck on `/health`. Index artifacts are mounted at `/data`, not baked into the image.

## Project structure

```
Legal-AI-Assistant/
├── src/rag/
│   ├── config.py       # pydantic-settings: all tunables, env-driven, validated
│   ├── chunker.py      # legal-structural chunking with context headers
│   ├── vector_store.py # VectorStore interface; TurboVec (4-bit) + FAISS backends
│   ├── embeddings.py   # Gemini embedding client (batching, cache, retries)
│   ├── indexer.py      # dataset load -> chunk -> embed -> dual index build
│   ├── bm25.py         # BM25 keyword search over chunks
│   ├── fusion.py       # Reciprocal Rank Fusion
│   ├── rerank.py       # Cohere cross-encoder reranker
│   ├── rewrite.py      # multi-turn LLM query rewriting
│   ├── hybrid.py       # HybridRetriever: dense + BM25 -> RRF -> rerank
│   ├── prompts.py      # Mike Ross persona, two-mode routing, degradation copy
│   ├── retry.py        # classified retry helper (fatal / rate-limit / transient)
│   └── errors.py       # typed provider exceptions
├── src/api/
│   ├── main.py         # app factory + lifespan (index loaded once)
│   ├── routes.py       # /chat SSE, /search, /health
│   ├── sse.py          # event protocol + chat stream generator
│   ├── schemas.py      # Pydantic request/response models
│   ├── state.py        # AppState: retriever + LLM client shared by requests
│   └── metrics.py      # Prometheus middleware and counters
├── evals/
│   ├── config.py       # eval settings (judge endpoint, golden-set size)
│   ├── golden_set.py   # seeded auto samples + hand-written queries
│   ├── hand_written.json
│   ├── retrieval_eval.py    # chunk-level recall@{1,3,5}, MRR, latency
│   ├── faithfulness_eval.py # claim-level faithfulness, citation precision, casual mode
│   ├── run_eval.py     # orchestrator; report.json, threshold gates, exit codes
│   └── thresholds.yaml # regression bars, calibrated from baseline runs
├── app.py              # Streamlit chat UI (thin SSE client of the API)
├── tests/              # pytest suite (no network calls)
├── .github/workflows/  # CI: lint -> type-check -> test -> eval gate
├── Dockerfile          # multi-stage, non-root, healthchecked
├── requirements.txt / requirements-dev.txt
└── .env                # API keys (gitignored)
```

---

## Evaluation

The eval harness measures the pipeline the way a reviewer would: does retrieval surface the right chunk, and does the generated answer stay inside the retrieved context?

```bash
python -m evals.run_eval --retrieval-only   # retrieval + casual-mode gate (no LLM cost)
python -m evals.run_eval                    # full run: adds claim-level faithfulness
python -m evals.run_eval --compare-engines  # TurboVec vs FAISS benchmark
```

What it measures (all chunk-level, not doc-level):
- **recall@{1,3,5} and MRR** against a seeded 40-sample golden set drawn from the indexed chunks, plus 15 hand-written queries.
- **Faithfulness**: an LLM judge decomposes each generated answer into atomic claims and verifies each against the retrieved context. Unsupported specifics count NO; paraphrase counts YES.
- **Citation precision**: every `Party v Party [YYYY] COURT N` citation in an answer must appear in the retrieved context. A made-up citation is the costliest failure a legal RAG has.
- **Casual-mode decline rate**: off-corpus queries must come back with a decline, not a hallucinated legal answer.

Honest failure accounting: provider outages are counted (`n_errors`) and excluded from scoring rather than silently turned into misses, and the run exits 2 when it's unreliable. Threshold gates live in [evals/thresholds.yaml](evals/thresholds.yaml); a regression fails the run with exit code 1.

Latest baseline (40 auto samples, 0 errors): recall@3 = 1.000, recall@1 = 0.600, MRR = 0.792, casual decline rate = 1.000; on the last full run, claim-level faithfulness = 0.898 and citation precision = 0.963. Re-run `--retrieval-only` to reproduce; numbers land in `evals/report.json` (gitignored).

## Dataset

[isaacus/open-australian-legal-qa](https://huggingface.co/datasets/isaacus/open-australian-legal-qa): 2,124 Australian legal Q&A pairs with citations, source URLs, and full source text of the underlying judgments and statutes. Predominantly case-law questions ("what were the key issues in *X v Y*?").
