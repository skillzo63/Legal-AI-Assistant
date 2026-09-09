"""Prometheus instrumentation: one middleware, four metrics, no decorators."""

import time

from fastapi import FastAPI, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

REQUESTS = Counter(
    "legal_rag_http_requests_total",
    "HTTP requests by route and status.",
    ["route", "status"],
)
# Middleware timing covers headers-out, not stream completion; chat
# generation time is the real story, so it gets its own histogram.
CHAT_DURATION = Histogram(
    "legal_rag_chat_duration_seconds",
    "Full chat answer time, first retrieval to final token.",
    buckets=(0.5, 1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 30.0),
)
LATENCY = Histogram(
    "legal_rag_http_request_duration_seconds",
    "Request latency by route (headers-out for streams).",
    ["route"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)
MODE_SPLIT = Counter(
    "legal_rag_chat_mode_total",
    "Chat responses by mode (legal vs casual).",
    ["mode"],
)
ERRORS = Counter(
    "legal_rag_pipeline_errors_total",
    "Pipeline errors by type (embedding, llm, rerank).",
    ["error_type"],
)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Counts every request and observes its latency, labeled by route.

    Known ceiling: for SSE routes ``call_next`` returns at headers-out,
    so the observed latency under-measures streaming time. CHAT_DURATION
    covers the real chat timing from inside the stream.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        route = request.url.path
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        REQUESTS.labels(route=route, status=str(response.status_code)).inc()
        LATENCY.labels(route=route).observe(elapsed)
        return response


def metrics_endpoint(request: Request) -> Response:
    """Render the registry in Prometheus text format."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


def setup_metrics(app: FastAPI) -> None:
    """Attach the metrics middleware and /metrics route to the app."""
    app.add_middleware(MetricsMiddleware)
    app.add_route("/metrics", metrics_endpoint, methods=["GET"])


def record_mode(mode: str) -> None:
    """Called from /chat once the mode is decided: legal or casual."""
    MODE_SPLIT.labels(mode=mode).inc()


def record_chat_duration(seconds: float) -> None:
    """Called from the chat stream when the answer completes (or fails)."""
    CHAT_DURATION.observe(seconds)


def record_error(error_type: str) -> None:
    """Called from /chat error handling: embedding_outage, llm_outage, ..."""
    ERRORS.labels(error_type=error_type).inc()
