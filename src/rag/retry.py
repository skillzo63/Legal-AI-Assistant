"""Retry with exception classification for external-provider calls.

Every provider call in the pipeline routes through ``retry_on_exception``.
Exceptions are classified once, here:

- ``fatal``      : auth/permission/bad-request (401/403/400/422, "invalid
  api key"). Retrying is waste, so raise immediately.
- ``rate_limit`` : 429 / quota / "too many requests". Exponential backoff
  with a longer base.
- ``transient``  : timeouts, connection resets. Standard backoff.
"""

import functools
import logging
import random
import time
from collections.abc import Callable
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

MAX_ATTEMPTS = 3
BASE_DELAY_S = 1.0
RATE_LIMIT_DELAY_S = 10.0

_FATAL_STATUS = {400, 401, 403, 404, 422}
_RATE_STATUS = {429}
_RATE_MARKERS = ("429", "rate limit", "too many requests", "resource_exhausted", "quota")
_FATAL_MARKERS = ("api key", "apikey", "unauthorized", "forbidden", "invalid request")


def classify_exception(exc: BaseException) -> str:
    """Classify a provider exception as ``fatal``, ``rate_limit``, or ``transient``.

    Rate markers are checked before fatal markers because rate-limit errors
    often mention the API key ("quota exceeded for api key ...") and must not
    be mistaken for auth failures.
    """
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        if status in _FATAL_STATUS:
            return "fatal"
        if status in _RATE_STATUS:
            return "rate_limit"
    msg = str(exc).lower()
    if any(marker in msg for marker in _RATE_MARKERS):
        return "rate_limit"
    if any(marker in msg for marker in _FATAL_MARKERS):
        return "fatal"
    return "transient"


def retry_on_exception(
    attempts: int = MAX_ATTEMPTS,
    base_delay: float = BASE_DELAY_S,
    rate_limit_delay: float = RATE_LIMIT_DELAY_S,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator retrying transient failures with exponential backoff and full jitter.

    Args:
        attempts: Total attempts (first call included).
        base_delay: Delay scale in seconds; attempt *n* sleeps up to ``base_delay * 2**(n-1)``.
        rate_limit_delay: Same, but for classified rate-limit errors.
        exceptions: Exception types worth retrying.

    Returns:
        The wrapped function. Fatal errors and the last exception are re-raised.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: object, **kwargs: object) -> T:
            for attempt in range(1, attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    kind = classify_exception(exc)
                    if kind == "fatal" or attempt == attempts:
                        raise
                    delay = rate_limit_delay if kind == "rate_limit" else base_delay
                    sleep_s = random.uniform(0, delay * 2 ** (attempt - 1))
                    logger.warning(
                        "%s failed (attempt %d/%d, %s) - retrying in %.1fs",
                        func.__name__,
                        attempt,
                        attempts,
                        kind,
                        sleep_s,
                    )
                    time.sleep(sleep_s)
            raise AssertionError("unreachable")  # pragma: no cover

        return wrapper

    return decorator
