"""
``EmbeddingVoyage``: the Voyage AI Embeddings API (``POST /v1/embeddings``),
1,000 texts a request. See docs/tools/retrieval.md.
"""

from ._base import _OpenAIShaped

_BASE_URL = "https://api.voyageai.com"


class EmbeddingVoyage(_OpenAIShaped):
    """
    Embedder for the Voyage AI Embeddings API.

    Parameters
    ----------
    model : str
        Model identifier, e.g. ``"voyage-3-large"``.
    api_key : str | None, optional
        Override ``VOYAGE_API_KEY`` environment variable.
    dimensions, batch_size
        Settings of the embedder; a call takes only ``input_type``.

    Examples
    --------
    ::

        from yait_aichain.tools import embed

        embedder = embed("voyage", model="voyage-3-large")
        result   = embedder.embed(
            "Explain transformer architecture",
            input_type="query",
        )
        print(result.dimensions)   # 1024
    """

    provider   = "voyage"
    _ENV_KEY   = "VOYAGE_API_KEY"
    _MAX_BATCH = 1000
    _MAX_TOKENS = 120_000          # voyage-3-large's; the lighter models take more
    _URL        = f"{_BASE_URL}/v1/embeddings"
    _DIMS_FIELD = "output_dimension"
    _FLOAT      = False
    _INPUT_TYPE = True
