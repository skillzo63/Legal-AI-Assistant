"""Tests for multi-turn query rewriting, with the Groq client faked."""

from typing import Any

from rag import rewrite


class _FakeGroq:
    """Records the payload it was called with and returns a canned rewrite."""

    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.seen_payload: list[dict[str, str]] | None = None

        class _Completions:
            def create(inner_self, model: str, messages: list[dict[str, str]], **_: Any):  # noqa: N805
                self.seen_payload = messages
                msg = type("M", (), {"content": self._reply})
                choice = type("C", (), {"message": msg})
                return type("R", (), {"choices": [choice]})

        self.chat = type("Chat", (), {"completions": _Completions()})


def test_first_turn_skips_llm() -> None:
    """No history → return the query unchanged without calling the model."""
    client = _FakeGroq(reply="SHOULD NOT BE USED")
    out = rewrite.rewrite_query(client, "What is a tort?", history=[])
    assert out == "What is a tort?"
    assert client.seen_payload is None


def test_followup_is_rewritten() -> None:
    """With history, the model's standalone rewrite is returned."""
    client = _FakeGroq(reply="Do business torts differ from personal torts?")
    history = [
        {"role": "user", "content": "What is a tort?"},
        {"role": "assistant", "content": "A civil wrong causing harm."},
    ]
    out = rewrite.rewrite_query(client, "what about for businesses?", history)
    assert out == "Do business torts differ from personal torts?"
    assert client.seen_payload is not None


def test_empty_rewrite_falls_back_to_original() -> None:
    """A blank model reply must not produce an empty query."""
    client = _FakeGroq(reply="   ")
    history = [{"role": "user", "content": "prior"}]
    out = rewrite.rewrite_query(client, "original query", history)
    assert out == "original query"
