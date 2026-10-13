"""
tools.embed._openai — EmbeddingOpenAI
======================================

Wraps the OpenAI Embeddings API (``POST /v1/embeddings``).

Supported models
----------------
  text-embedding-3-small    1536-dim default; supports truncation to any size
  text-embedding-3-large    3072-dim default; supports truncation to any size
  text-embedding-ada-002    1536-dim; no dimension override

Model notes
-----------
* ``text-embedding-3-*`` accept a ``dimensions`` parameter (1–max-dim).
* ``text-embedding-ada-002`` has a fixed length; ``dimensions`` on it is refused when the
  embedder is made.
* OpenAI does not distinguish query vs document embeddings; ``input_type`` is
  accepted by the interface but ignored.
* The API accepts up to 2 048 strings per request.

Environment variable
--------------------
OPENAI_API_KEY
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
