"""Multi-turn query rewriting.

Follow-ups like "what about for businesses?" are meaningless to a retriever
on their own — the subject lives in earlier turns. This rewrites the latest
user message into a standalone question using recent history, so retrieval
sees "Do business torts differ from personal torts?" instead of a dangling
pronoun. First-turn queries need no rewrite and skip the LLM call.
"""

from groq import Groq

from rag.config import settings
from rag.errors import LLMError
from rag.retry import retry_on_exception

_REWRITE_SYSTEM = (
    "You rewrite a user's latest message into a single standalone search "
    "query, resolving pronouns and references using the conversation history. "
    "Output ONLY the rewritten query — no preamble, no quotes, no explanation. "
    "If the message is already self-contained, return it unchanged."
)

# How many prior turns of context the rewriter sees. Enough to resolve a
# reference without paying to re-read the whole thread every turn.
_HISTORY_TURNS = 6


def _recent_turns(history: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return the last few non-system turns as rewrite context."""
    turns = [m for m in history if m["role"] in ("user", "assistant")]
    return turns[-_HISTORY_TURNS:]


@retry_on_exception(exceptions=(Exception,))
def _call(client: Groq, payload: list[dict[str, str]]) -> str:
    completion = client.chat.completions.create(
        model=settings.llm.model,
        messages=payload,  # type: ignore[arg-type]
        temperature=0.0,
        max_tokens=128,
    )
    return (completion.choices[0].message.content or "").strip()


def rewrite_query(
    client: Groq, query_text: str, history: list[dict[str, str]]
) -> str:
    """Rewrite ``query_text`` into a standalone question.

    Args:
        client: Groq client for the rewrite completion.
        query_text: The raw latest user message.
        history: Full conversation so far (system/user/assistant messages).

    Returns:
        The standalone query, or the original text unchanged when there is no
        prior context to resolve against.

    Raises:
        LLMError: The rewrite completion failed after retries.
    """
    context = _recent_turns(history)
    if not context:
        return query_text

    payload: list[dict[str, str]] = [{"role": "system", "content": _REWRITE_SYSTEM}]
    payload.extend(context)
    payload.append(
        {"role": "user", "content": f"Rewrite this into a standalone query: {query_text}"}
    )

    try:
        rewritten = _call(client, payload)
    except Exception as exc:
        raise LLMError(f"Query rewrite failed after retries: {exc}") from exc

    # Empty rewrite (model returned nothing usable) → fall back to the original
    # rather than embedding an empty string.
    return rewritten or query_text
