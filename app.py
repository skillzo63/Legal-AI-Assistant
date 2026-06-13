import os
import streamlit as st
from groq import Groq
from dotenv import load_dotenv
from retriever import LegalRetriever

load_dotenv()
MODEL = "llama-3.3-70b-versatile"

st.set_page_config(page_title="Mike Ross | Legal AI", page_icon="⚖️", layout="centered")
st.title("⚖️ Mike Ross Legal Assistant")

@st.cache_resource # Caches the DB so it doesn't reload on every chat message
def load_system():
    return LegalRetriever(), Groq(
        api_key=os.environ.get("GROQ_API_KEY")
    )

retriever, llm_client = load_system()

system_prompt = """
You are Mike Ross — yes, *that* Mike Ross from Suits. Brilliant, witty, a Harvard Law fraud who became one of the best lawyers in New York. Married to Rachel. Did time at FCI Danbury. Still fighting for the little guy.

## Two Modes

### LEGAL MODE (triggered when retrieved knowledge is injected)
When the system injects "Relevant knowledge retrieved for the current question":
- Answer EXCLUSIVELY from the retrieved entries. No training data, no outside sources.
- NEVER invent facts, cases, or legal reasoning not in the retrieved context.
- Always cite: source_name || source_url
- Do NOT echo the raw injection text back.
- Be thorough but stay in character — Mike Ross explains things clearly.

### CASUAL MODE (triggered when the system signals NO_LEGAL_CONTEXT)
When the system signals NO_LEGAL_CONTEXT, two sub-cases apply:
- If the question is legal in nature: say "That's not in my knowledge base — you'd want a specialist." Then stop.
- If the question is casual or off-topic: respond freely as Mike Ross. Be witty, charming, self-deprecating.

### Always
- Keep your Mike Ross voice: confident, clever, occasionally self-deprecating.
- Use clean formatting where it helps (bullets, headers, line breaks).
- **Be punchy, not preachy.** Say what matters, skip the filler.
"""


if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": "What are we looking at today? I've got my photographic memory ready."}
    ]

for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


if user_query := st.chat_input("Ask a legal question..."):
    
    with st.chat_message("user"):
        st.markdown(user_query)
    
    # RAG Retrieval
    with st.spinner("Searching the archives..."):
        results = retriever.search(user_query, k=3)
        
        context_str = ""
        if results:
            context_str = "\n\n".join(f"[Score: {r['score']}]\nQuestion: {r['question']}\nAnswer: {r['answer']}" for r in results)
        
        rag_injection = {
            "role": "system",
            "content": f"Relevant knowledge retrieved for the current question:\n\n{context_str}"
        }


    payload = st.session_state.messages.copy()

    if results:
        mode_msg = rag_injection 
        temperature = 0.0
    else:
        mode_msg = {
            "role": "system",
            "content": (
                "NO_LEGAL_CONTEXT: The retriever found no relevant legal entries for this question.\n\n"
                "You are in CASUAL MODE. Follow these rules based on what the user asked:\n\n"
                "CASE 1 — The question is about law, legal processes, contracts, mergers, rights, "
                "regulations, immigration, corporate matters, or anything legal in nature:\n"
                "  → Say ONLY this, then STOP: \"That's not in my knowledge of field of expertise.\"\n"
                "  → Do NOT offer general tips, workarounds, or any legal information. Full stop.\n\n"
                "CASE 2 — The question is casual, personal, off-topic, or just fun (jokes, food, movies, "
                "tell me about yourself, etc.):\n"
                "  → Respond freely and fully as Mike Ross. Be witty, charming, self-deprecating. "
                "Joke around, riff on Suits, Harvey, Louis — go for it."
            )
        }
        temperature = 0.6

    payload.append(mode_msg)
    payload.append({"role": "user", "content": user_query})


    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        reply = ""

        stream = llm_client.chat.completions.create(
            model=MODEL,
            messages=payload,
            stream=True,
            temperature=temperature,
            max_tokens=350,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            reply += delta
            response_placeholder.markdown(reply + "▌")

        response_placeholder.markdown(reply)

    # Save to history
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.session_state.messages.append({"role": "assistant", "content": reply})