"""Shared per-process state: the pipeline objects built at startup."""

from dataclasses import dataclass

from groq import Groq

from rag.hybrid import HybridRetriever


@dataclass
class AppState:
    """Pipeline objects owned by the process, shared across requests."""

    retriever: HybridRetriever | None  # None => /health degraded, /chat 503
    llm: Groq | None


def build_state() -> AppState:
    """Build the retriever and LLM client from configured settings.

    The Groq client reads GROQ_API_KEY from the process environment
    (main.py loads .env first), matching how run_eval.py constructs it.
    """
    retriever = HybridRetriever.load()
    llm = Groq()
    return AppState(retriever=retriever, llm=llm)
