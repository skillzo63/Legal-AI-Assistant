# ⚖️ Mike Ross — Legal RAG Assistant

A Retrieval-Augmented Generation assistant over Australian case law, themed as Mike Ross from *Suits*. Answers are grounded in a curated legal Q&A corpus — never the model's own training data. When nothing relevant is retrieved for a legal question, the LLM is not called at all.

Built to show an **end-to-end, production-shaped RAG pipeline**: hybrid retrieval (dense + keyword), rank fusion, cross-encoder reranking, multi-turn query rewriting, typed error handling, and a fully tested, type-checked codebase.

---

## Why this is more than a vector-search demo

Most RAG demos embed the query, do one nearest-neighbour lookup, and dump the top-k into a prompt. That misses exact-term matches (case names, statute numbers), has no way to tell "close in vector space" from "actually relevant", and breaks on follow-up questions. This pipeline addresses each:

| Problem | Naive demo | Here |
|---------|-----------|------|
| Semantic *and* keyword relevance | dense only | **dense + BM25**, fused |
| Combining two rankers with different score scales | n/a | **Reciprocal Rank Fusion** (rank-based, scale-free) |
| "Near in vector space" ≠ "relevant" | cosine threshold | **cross-encoder rerank** reads query+doc together |
| Follow-ups ("what about for businesses?") | embedded as-is | **LLM query rewriting** resolves them to standalone queries |
| Provider outage | silent failure / hallucination | **typed exceptions → explicit user-facing degradation** |

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

Stage one casts a wide, cheap net (dense + BM25 → fused pool). Stage two is the expensive, accurate pass (cross-encoder rerank). The **rerank score is authoritative**, so the relevance threshold applies to it — nothing weakly relevant reaches the LLM, which is what preserves the hard-grounding guarantee.

## Key design decisions

- **Hybrid retrieval.** Dense catches meaning ("detaining people" → *Ruddock v Vadarlis*); BM25 catches exact terms (case names, section numbers) that embeddings blur. Neither alone is enough for legal text.
- **RRF over score-averaging.** Cosine and BM25 scores live on incompatible scales — averaging them is meaningless. RRF fuses by *rank position* (`Σ 1/(k+rank)`), so it never needs the raw scores to be comparable.
- **Cross-encoder rerank as the relevance gate.** A bi-encoder embeds query and doc separately; a cross-encoder reads them *together* and scores true relevance. Cohere's scores are near-binary (relevant ≈ 1.0, irrelevant ≈ 0.0) on this corpus, so the `0.5` threshold sits in a wide empty gap — maximum margin, tuned from observed score distribution.
- **Multi-turn query rewriting.** A retriever can't resolve "what about for businesses?" — the subject is in an earlier turn. An LLM rewrites it to a standalone query before embedding. First-turn queries skip the call.
- **Hard grounding.** No retrieved context for a legal question → the LLM is never invoked. No confident hallucination.
- **Typed degradation.** Provider errors (`EmbeddingError`, `LLMError`) surface as explicit user-facing messages, never silent failures or fabricated answers. Rewrite failure degrades gracefully to the raw query.
- **Config, not constants.** Every tunable (models, threshold, top-k, candidate pool, temperatures) is env-driven via `pydantic-settings` with validation bounds — nothing hardcoded in pipeline code.
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
```

Get keys (all have free tiers):
- Gemini: https://aistudio.google.com/apikey
- Groq: https://console.groq.com/keys
- Cohere (reranker): https://dashboard.cohere.com/api-keys

All pipeline settings are configurable via env vars with section prefixes (`EMBEDDING_*`, `RETRIEVAL_*`, `RERANK_*`, `LLM_*`, `INDEX_*`) — see [src/rag/config.py](src/rag/config.py).

### 3. Build the vector index
```bash
python -m rag.indexer
```
Downloads the [Open Australian Legal QA](https://huggingface.co/datasets/isaacus/open-australian-legal-qa) dataset, embeds `INDEX_MAX_RECORDS` entries (default 500, predominantly case-law Q&A), and writes `aus_legal_qa.tv` + `metadata.json`. BM25 is rebuilt in memory from the metadata at load time.

### 4. Run the app
```bash
streamlit run app.py
```

### 5. Run the checks
```bash
pytest && ruff check src tests app.py && mypy
```

---

## Project structure

```
Legal-AI-Assistant/
├── src/rag/
│   ├── config.py       # pydantic-settings — all tunables, env-driven, validated
│   ├── embeddings.py   # Gemini embedding client (retries, lru_cache)
│   ├── indexer.py      # dataset load → embed → TurboVec index build
│   ├── bm25.py         # BM25 keyword search over questions
│   ├── fusion.py       # Reciprocal Rank Fusion
│   ├── rerank.py       # Cohere cross-encoder reranker
│   ├── rewrite.py      # multi-turn LLM query rewriting
│   ├── hybrid.py       # HybridRetriever: dense + BM25 → RRF → rerank
│   ├── prompts.py      # Mike Ross persona, two-mode routing, degradation copy
│   ├── retry.py        # exponential-backoff-with-jitter helper
│   └── errors.py       # typed provider exceptions
├── app.py              # Streamlit UI
├── tests/              # pytest suite (no network calls)
├── requirements.txt / requirements-dev.txt
└── .env                # API keys (gitignored)
```

---

## Evaluation

An eval harness (golden held-out Q&A set → recall@k, MRR, LLM-judge faithfulness) is the next milestone. Retrieval-quality numbers and the hybrid-vs-dense comparison will be reported here from measured runs — no placeholder figures until then.

## Dataset

[isaacus/open-australian-legal-qa](https://huggingface.co/datasets/isaacus/open-australian-legal-qa) — 2,124 Australian legal Q&A pairs with citations and source URLs. Predominantly case-law questions ("what were the key issues in *X v Y*?").
