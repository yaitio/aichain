"""
``EmbeddingOpenAI``: the OpenAI Embeddings API (``POST /v1/embeddings``),
2,048 texts a request. OpenAI ignores ``input_type``.
See docs/tools/retrieval.md.
"""

from ._base import _OpenAIShaped

_BASE_URL = "https://api.openai.com"

# Models that accept the dimensions parameter
_SUPPORTS_DIMENSIONS = frozenset({
    "text-embedding-3-small",
    "text-embedding-3-large",
})


class EmbeddingOpenAI(_OpenAIShaped):
    """
    Embedder for the OpenAI Embeddings API.

    Parameters
    ----------
    model : str
        Model identifier, e.g. ``"text-embedding-3-large"``.
    api_key : str | None, optional
        Override ``OPENAI_API_KEY`` environment variable.
    dimensions, batch_size
        Settings of the embedder; a call takes only ``input_type``.

    Examples
    --------
    ::

        from yait_aichain.tools import embed

        embedder = embed("openai", model="text-embedding-3-large")
        result   = embedder.embed("What is machine learning?", input_type="query")
        print(result.dimensions)   # 3072
    """

    provider   = "openai"
    _DIMENSION_MODELS = _SUPPORTS_DIMENSIONS
    _ENV_KEY   = "OPENAI_API_KEY"
    _MAX_BATCH = 2048
    _MAX_TOKENS = 300_000
    _URL       = f"{_BASE_URL}/v1/embeddings"
