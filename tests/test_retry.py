"""Tests for the retry helper — fast, with zero delay."""

import pytest

from rag.retry import retry_on_exception


def test_succeeds_first_attempt() -> None:
    calls = []

    @retry_on_exception(base_delay=0)
    def fn() -> str:
        calls.append(1)
        return "ok"

    assert fn() == "ok"
    assert len(calls) == 1


def test_retries_then_succeeds() -> None:
    calls = []

    @retry_on_exception(attempts=3, base_delay=0)
    def flaky() -> int:
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("transient")
        return 42

    assert flaky() == 42
    assert len(calls) == 3


def test_raises_after_exhausting_attempts() -> None:
    calls = []

    @retry_on_exception(attempts=3, base_delay=0)
    def always_down() -> None:
        calls.append(1)
        raise TimeoutError("down")

    with pytest.raises(TimeoutError):
        always_down()
    assert len(calls) == 3


def test_only_matching_exceptions_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-matching exceptions propagate immediately without retries."""
    calls = []

    @retry_on_exception(attempts=3, base_delay=0, exceptions=(TimeoutError,))
    def bad_request() -> None:
        calls.append(1)
        raise ValueError("permanent 4xx-style failure")

    with pytest.raises(ValueError):
        bad_request()
    assert len(calls) == 1