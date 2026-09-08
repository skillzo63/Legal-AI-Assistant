"""Eval-harness configuration via pydantic-settings.

All NVIDIA judge settings load from environment variables with the ``EVAL_``
prefix (same ``.env`` file as the pipeline). Keeps eval config isolated from
``rag.config`` so the two can evolve independently.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EvalSettings(BaseSettings):
    """Settings for the M3 evaluation harness."""

    model_config = SettingsConfigDict(env_prefix="EVAL_", env_file=".env", extra="ignore")

    # NVIDIA NIM judge, defaulting to MiniMax-M3
    judge_api_key: str = ""
    judge_model: str = "minimaxai/minimax-m3"

    # Backward compatibility with EVAL_JUDGE_A_* in .env
    judge_a_api_key: str = ""
    judge_a_model: str = "minimaxai/minimax-m3"

    # Shared NVIDIA NIM base URL (OpenAI-compatible)
    judge_base_url: str = "https://integrate.api.nvidia.com/v1"

    # Golden set parameters. 40 auto samples: the retrieval eval costs
    # embeddings + rerank only (no LLM calls), so a bigger sample count is
    # cheap resolution - 2.5% steps instead of 15-sample 7% steps.
    golden_set_size: int = Field(40, ge=1, description="Auto samples drawn from the index")
    random_seed: int = Field(42, description="Seed for reproducible sampling")

    @property
    def active_judge_api_key(self) -> str:
        return self.judge_api_key or self.judge_a_api_key

    @property
    def active_judge_model(self) -> str:
        return self.judge_model or self.judge_a_model


eval_settings = EvalSettings()
