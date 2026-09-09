"""Streamlit chat UI for the Mike Ross legal assistant.

Thin client over the FastAPI service: every turn is a POST to /chat whose
SSE stream is rendered token by token. Conversation history lives in the
Streamlit session and is sent with each request; the API itself is
stateless. Run the API first: uvicorn src.api.main:app --port 8000
"""

import json

import requests
import streamlit as st

# 127.0.0.1, not localhost: skips IPv6 (::1) resolution flakes on Windows.
API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Mike Ross | Legal AI", page_icon="⚖️", layout="centered")
st.title("⚖️ Mike Ross Legal Assistant")


def parse_sse_stream(response: requests.Response) -> list[tuple[str, dict]]:
    """Yield (event, data) frames from an SSE response as they arrive.

    The stream is consumed line by line so tokens render live; frames are
    buffered per SSE block (blank line terminated).
    """
    event = ""
    data_lines: list[str] = []
    for line in response.iter_lines(decode_unicode=True):
        if line is None:
            continue
        if line == "":
            if event:
                yield event, json.loads("".join(data_lines))
            event, data_lines = "", []
            continue
        if line.startswith("event: "):
            event = line.removeprefix("event: ").strip()
        elif line.startswith("data: "):
            data_lines.append(line.removeprefix("data: "))
    if event and data_lines:
        yield event, json.loads("".join(data_lines))


if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "What are we looking at today? I've got my photographic memory ready.",
        },
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


def _api_history() -> list[dict[str, str]]:
    """Session history in the API's ChatRequest shape (no system prompt)."""
    return [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
        if m["role"] in ("user", "assistant")
    ]


def render_chat(query: str) -> None:
    """POST the turn to /chat and render the SSE stream as it arrives."""
    with st.chat_message("assistant"):
        placeholder = st.empty()
        reply = ""
        try:
            response = requests.post(
                f"{API_URL}/chat",
                json={"query": query, "history": _api_history()},
                stream=True,
                timeout=120,
            )
        except requests.RequestException:
            placeholder.markdown(
                "The assistant service is not reachable. Is the API running? "
                "Start it with: `uvicorn src.api.main:app --port 8000`"
            )
            st.session_state.messages.append(
                {"role": "assistant", "content": "Service unreachable."}
            )
            return
        if response.status_code != 200:
            # The API answered with an error status; show what it said rather
            # than pretending the connection failed (503 = degraded startup).
            try:
                detail = response.json().get("detail", response.text[:200])
            except ValueError:
                detail = response.text[:200]
            placeholder.markdown(f"The API returned {response.status_code}: {detail}")
            st.session_state.messages.append(
                {"role": "assistant", "content": f"API error ({response.status_code})."}
            )
            return

        for event, data in parse_sse_stream(response):
            if event == "token":
                reply += data["text"]
                placeholder.markdown(reply + "▌")
            elif event == "error":
                placeholder.markdown(data["message"])
                reply = data["message"]
                break
            elif event == "done":
                reply = data["answer"]
        placeholder.markdown(reply)

    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.messages.append({"role": "assistant", "content": reply})


if user_query := st.chat_input("Ask a legal question..."):
    with st.chat_message("user"):
        st.markdown(user_query)
    render_chat(user_query)
