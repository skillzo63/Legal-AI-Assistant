# ⚖️ Mike Ross — Legal RAG Assistant

A Retrieval-Augmented Generation (RAG) chatbot built from scratch, themed as Mike Ross from *Suits*. Ask legal questions and get answers grounded in a curated legal Q&A dataset — not the model's training data.

Built to demonstrate end-to-end RAG system design: custom vector indexing with **TurboVec**, semantic embedding with **Gemini**, and LLM inference.

---

## Architecture

```
        User Query
            │
            ▼
┌─────────────────────────┐
│      Embedding API      │  ← query embedded
│  (RETRIEVAL_QUERY mode) │
└───────────┬─────────────┘
            │  query vector
            ▼
┌─────────────────────────┐
│  TurboVec ANN Index     │  ← quantized 4-bit index, cosine similarity search
│  (aus_legal_qa.tv)      │
└───────────┬─────────────┘
            │  top-k results (score ≥ 0.65)
            ▼
┌─────────────────────────┐
│                         │
│  Results found  → Strict Mode   
│  No results     → Casual Mode  
└───────────┬─────────────┘
            │  
            ▼
┌─────────────────────────┐
│          LLM            │
│  (Mike Ross persona)    │
└─────────────────────────┘
```

## Key Design Decisions

- **TurboVec**: faster search, smaller index files
- **Embedding only the question** for indexing (not the full Q&A text) — improves semantic match accuracy
- **Hashed ID** to avoid duplicates while updating the index
- **Cached query embeddings** via `functools.lru_cache` — avoids redundant API calls for repeated queries
- **Hard grounding**: the LLM is never called when no RAG context is found for legal queries

---

## Setup

### 1. Clone & install
```bash
git clone https://github.com/skillzo63/RAG.git
cd RAG
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### 2. Configure API keys
Create a `.env` file in the project root:
```
GEMINI_API_KEY=your_google_gemini_api_key
GROQ_API_KEY=your_groq_api_key
```

Get keys:
- Gemini: https://aistudio.google.com/apikey
- Groq: https://console.groq.com/keys

### 3. Build the vector index
```bash
python indexer.py
```
This downloads the [Open Australian Legal QA](https://huggingface.co/datasets/isaacus/open-australian-legal-qa) dataset, embeds `MAX_RECORDS` entries, and saves `aus_legal_qa.tv` + `metadata.json`.


### 4. Run the app
```bash
streamlit run app.py
```

---

## Project Structure

```
RAG/
├── app.py          # Streamlit UI, RAG pipeline, two-mode router
├── indexer.py      # Dataset loading, embedding, TurboVec index builder
├── retriever.py    # LegalRetriever class, cached embedding lookup
├── requirements.txt
└── .env            # API keys (not committed)
```

---

## Dataset

[isaacus/open-australian-legal-qa](https://huggingface.co/datasets/isaacus/open-australian-legal-qa) — 2,124 legal Q&A pairs with citations and source URLs.