# Multi-stage: build pinned deps in a venv, copy it onto slim runtime.
FROM python:3.12-slim AS builder

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.12-slim

# Run as a non-root user; the app never needs more.
RUN useradd --create-home appuser
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY pyproject.toml README.md ./
COPY src ./src
COPY app.py ./

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    # Index artifacts live in a volume, not the image.
    LEGAL_RAG_DATA_DIR=/data

RUN pip install --no-cache-dir --no-deps . && mkdir -p /data && chown appuser:appuser /data
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)" || exit 1

# Index files (metadata.json, aus_legal_qa.tv) are expected in /data,
# mounted at run time or built by the entrypoint on first start.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
