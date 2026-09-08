"""Application configuration via pydantic-settings.

All tunables live here; nothing is hardcoded in pipeline code. Values load
from environment variables / .env with per-section prefixes
(EMBEDDING_*, RETRIEVAL_*, LLM_*, INDEX_*, CHUNK_*). Flat env keys, nested
objects in code: ``settings.retrieval.threshold``.
"""

from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingSettings(BaseSettings):
    """Gemini embedding provider settings."""

    model_config = SettingsConfigDict(
        env_prefix="EMBEDDING_", env_file=".env", extra="ignore"
    )

    model: str = "gemini-embedding-2"
    dim: int = Field(3072, ge=1)


class RetrievalSettings(BaseSettings):
    """Dense retrieval settings."""

    model_config = SettingsConfigDict(
        env_prefix="RETRIEVAL_", env_file=".env", extra="ignore"
    )

    vector_backend: Literal["turbovec", "faiss"] = "turbovec"
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    top_k: int = Field(3, ge=1)


class LLMSettings(BaseSettings):
    """Groq generation settings."""

    model_config = SettingsConfigDict(env_prefix="LLM_", env_file=".env", extra="ignore")

    model: str = "qwen/qwen3.8-27b"
    temperature_legal: float = Field(0.0, ge=0.0, le=1.0)
    temperature_casual: float = Field(0.6, ge=0.0, le=1.0)
    max_tokens: int = Field(350, ge=1)


class RerankSettings(BaseSettings):
    """Hosted cross-encoder reranker (Cohere) settings."""

    model_config = SettingsConfigDict(
        env_prefix="RERANK_", env_file=".env", extra="ignore"
    )

    model: str = "rerank-english-v3.0"
    api_key: str = ""
    candidate_pool: int = Field(50, ge=1)
    min_interval: float = Field(6.2, ge=0.0)
    max_retries: int = Field(5, ge=1)


class ChunkSettings(BaseSettings):
    """Legal chunking configuration."""

    model_config = SettingsConfigDict(
        env_prefix="CHUNK_", env_file=".env", extra="ignore"
    )

    max_chars: int = Field(1000, ge=100)
    min_chars: int = Field(150, ge=20)
    overlap_chars: int = Field(100, ge=0)


class IndexSettings(BaseSettings):
    """Index build/load settings."""

    model_config = SettingsConfigDict(env_prefix="INDEX_", env_file=".env", extra="ignore")

    max_records: int = Field(500, ge=1)
    index_path: str = "aus_legal_qa.tv"
    faiss_index_path: str = "aus_legal_qa.faiss"
    bm25_path: str = "bm25.pkl"
    metadata_path: str = "metadata.json"


class Settings(BaseModel):
    """Root settings object; sections mirror module boundaries."""

    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    rerank: RerankSettings = Field(default_factory=RerankSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    chunk: ChunkSettings = Field(default_factory=ChunkSettings)
    index: IndexSettings = Field(default_factory=IndexSettings)


settings = Settings()