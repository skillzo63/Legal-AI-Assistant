"""Typed exceptions for external-provider and pipeline failures."""


class ProviderError(Exception):
    """Base class for external provider failures after retries are exhausted."""


class EmbeddingError(ProviderError):
    """Gemini embedding call failed after retries."""


class LLMError(ProviderError):
    """Groq chat call failed after retries."""


class RerankError(ProviderError):
    """Cohere rerank call failed after retries."""


class RetrievalError(Exception):
    """Vector store or hybrid retrieval failure."""