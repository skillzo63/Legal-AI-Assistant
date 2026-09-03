"""Retry with exponential backoff for network-boundary calls."""

import functools
import random
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

MAX_ATTEMPTS = 3
BASE_DELAY_S = 1.0


def retry_on_exception(
    attempts: int = MAX_ATTEMPTS,
    base_delay: float = BASE_DELAY_S,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator retrying transient failures with exponential backoff and full jitter.

    Args:
        attempts: Total attempts (first call included).
        base_delay: Delay scale in seconds; attempt *n* sleeps up to ``base_delay * 2**(n-1)``.
        exceptions: Exception types worth retrying.

    Returns:
        The wrapped function. Raises the last exception once attempts are exhausted.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: object, **kwargs: object) -> T:
            for attempt in range(1, attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions:
                    if attempt == attempts:
                        raise
                    # Retries every matching exception, including non-transient
                    # 4xx — acceptable at 3 attempts; narrow by provider error
                    # code if it ever wastes real time.
                    time.sleep(random.uniform(0, base_delay * 2 ** (attempt - 1)))
            raise AssertionError("unreachable")  # pragma: no cover

        return wrapper

    return decorator