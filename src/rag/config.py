"""Application configuration via pydantic-settings.

All tunables live here — nothing hardcoded in pipeline code. Values load
from environment variables / .env with per-section prefixes
(EMBEDDING_*, RETRIEVAL_*, LLM_*, INDEX_*). Flat env keys, nested
objects in code: ``settings.retrieval.threshold``.
"""

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingSettings(BaseSettings):
    """Gemini embedding provider settings."""

    model_config = SettingsConfigDict(
        env_prefix="EMBEDDING_", env_file=".env", extra="ignore"
    )

    # TODO: verify this model id against the Gemini API docs before the next index rebuild.
    model: str = "gemini-embedding-2"
    dim: int = Field(3072, ge=1)


class RetrievalSettings(BaseSettings):
    """Dense retrieval settings."""

    model_config = SettingsConfigDict(
        env_prefix="RETRIEVAL_", env_file=".env", extra="ignore"
    )

    threshold: float = Field(0.65, ge=0.0, le=1.0)
    top_k: int = Field(3, ge=1)


class LLMSettings(BaseSettings):
    """Groq generation settings."""

    model_config = SettingsConfigDict(env_prefix="LLM_", env_file=".env", extra="ignore")

    model: str = "llama-3.3-70b-versatile"
    temperature_legal: float = Field(0.0, ge=0.0, le=1.0)
    temperature_casual: float = Field(0.6, ge=0.0, le=1.0)
    max_tokens: int = Field(350, ge=1)


class IndexSettings(BaseSettings):
    """Index build/load settings."""

    model_config = SettingsConfigDict(env_prefix="INDEX_", env_file=".env", extra="ignore")

    max_records: int = Field(500, ge=1)
    index_path: str = "aus_legal_qa.tv"
    metadata_path: str = "metadata.json"


class Settings(BaseModel):
    """Root settings object; sections mirror module boundaries."""

    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    index: IndexSettings = Field(default_factory=IndexSettings)


settings = Settings()