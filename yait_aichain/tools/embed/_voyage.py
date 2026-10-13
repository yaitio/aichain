"""
tools.embed._voyage — EmbeddingVoyage
======================================

Wraps the Voyage AI Embeddings API (``POST /v1/embeddings``).

Supported models
----------------
  voyage-3-large     1024-dim default; best quality
  voyage-3           1024-dim default; balanced quality/cost
  voyage-3-lite      512-dim  default; lowest cost
  voyage-code-3      1024-dim; optimised for code retrieval
  voyage-finance-2   1024-dim; optimised for finance text
  voyage-law-2       1024-dim; optimised for legal text

Model notes
-----------
* All models support:
    - ``input_type``       — "query" or "document"; optional.
    - ``output_dimension`` — integer truncation; optional.
* The API accepts up to 1 000 strings (or 320 000 tokens) per request.
* Response includes token usage in ``usage.total_tokens``.

Universal ``input_type`` mapping
---------------------------------
  "query"    → "query"
  "document" → "document"
  None       → omitted from request body (provider default)

Environment variable
--------------------
VOYAGE_API_KEY
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
