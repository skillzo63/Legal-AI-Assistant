"""App assembly: lifespan, middleware, routers."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

from .metrics import setup_metrics
from .routes import router
from .state import AppState, build_state

# The Groq SDK reads its key from the process environment, not pydantic-settings,
# so .env must be loaded before build_state() constructs the client.
load_dotenv()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the retriever once at startup; requests share it via app.state.

    The index load (vectors + BM25 rebuild) takes a few seconds and must not
    happen per-request. If it fails, the app still starts and /health reports
    degraded rather than crashing the container into a restart loop. Tests
    pre-seed app.state.state before startup; only build when absent.
    """
    if getattr(app.state, "state", None) is None:
        try:
            app.state.state = build_state()
        except Exception:  # noqa: BLE001 - startup must not die on a bad index
            logger.exception("pipeline startup failed; serving degraded")
            app.state.state = AppState(retriever=None, llm=None)
    yield


def create_app() -> FastAPI:
    """Application factory. Tests call this with dependencies swapped out."""
    app = FastAPI(title="Legal AI Assistant", lifespan=lifespan)
    setup_metrics(app)
    app.include_router(router)
    return app


app = create_app()
