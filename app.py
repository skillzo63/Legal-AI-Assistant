"""Streamlit chat UI for the Mike Ross legal assistant."""

import streamlit as st
from groq import Groq

from rag.config import settings
from rag.errors import EmbeddingError, LLMError
from rag.hybrid import HybridRetriever
from rag.prompts import (
    LLM_UNAVAILABLE_MESSAGE,
    RETRIEVAL_UNAVAILABLE_MESSAGE,
    SYSTEM_PROMPT,
    route_mode,
)
from rag.retry import retry_on_exception
from rag.rewrite import rewrite_query

st.set_page_config(page_title="Mike Ross | Legal AI", page_icon="⚖️", layout="centered")
st.title("⚖️ Mike Ross Legal Assistant")


@st.cache_resource
def load_system() -> tuple[HybridRetriever, Groq]:
    """Load the retriever and LLM client once per session."""
    return HybridRetriever.load(), Groq()


try:
    retriever, llm_client = load_system()
except Exception:
    # Index or metadata unreachable → full outage for both modes.
    st.error(RETRIEVAL_UNAVAILABLE_MESSAGE)
    st.stop()


@retry_on_exception()
def _start_stream(payload: list[dict[str, str]], temperature: float):
    """Open the Groq streaming completion, retrying connection failures."""
    return llm_client.chat.completions.create(
        model=settings.llm.model,
        messages=payload,
        stream=True,
        temperature=temperature,
        max_tokens=settings.llm.max_tokens,
    )


if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "assistant",
            "content": "What are we looking at today? I've got my photographic memory ready.",
        },
    ]

for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

if user_query := st.chat_input("Ask a legal question..."):
    with st.chat_message("user"):
        st.markdown(user_query)

    # RAG retrieval — runs for every query, so an embedding outage degrades
    # both legal and casual modes. Multi-turn: rewrite follow-ups into a
    # standalone query first so retrieval isn't fed a dangling pronoun.
    with st.spinner("Searching the archives..."):
        try:
            search_query = rewrite_query(
                llm_client, user_query, st.session_state.messages
            )
        except LLMError:
            # Rewrite is best-effort — fall back to the raw query rather than
            # failing the whole turn.
            search_query = user_query
        try:
            results = retriever.search(search_query)
        except (EmbeddingError, LLMError):
            # Embedding or rerank provider down → no trustworthy grounding.
            results = None
            retrieval_down = True
        else:
            retrieval_down = False

    with st.chat_message("assistant"):
        if retrieval_down:
            st.markdown(RETRIEVAL_UNAVAILABLE_MESSAGE)
            st.session_state.messages.append({"role": "user", "content": user_query})
            st.session_state.messages.append(
                {"role": "assistant", "content": RETRIEVAL_UNAVAILABLE_MESSAGE}
            )
            st.stop()

        mode_msg, temperature = route_mode(results)
        payload = st.session_state.messages.copy()
        payload.append(mode_msg)
        payload.append({"role": "user", "content": user_query})

        response_placeholder = st.empty()
        reply = ""
        try:
            stream = _start_stream(payload, temperature)
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                reply += delta
                response_placeholder.markdown(reply + "▌")
        except Exception:
            # LLM outage → explicit error message.
            st.error(LLM_UNAVAILABLE_MESSAGE)
            st.stop()

        response_placeholder.markdown(reply)

    st.session_state.messages.append({"role": "user", "content": user_query})
    st.session_state.messages.append({"role": "assistant", "content": reply})