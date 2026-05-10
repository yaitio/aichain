"""
tools.embedding._voyage — EmbeddingVoyage
==========================================

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

from ._base import Embedder

_BASE_URL = "https://api.voyageai.com"


class EmbeddingVoyage(Embedder):
    """
    Embedder for the Voyage AI Embeddings API.

    Parameters
    ----------
    model : str
        Model identifier, e.g. ``"voyage-3-large"``.
    api_key : str | None, optional
        Override ``VOYAGE_API_KEY`` environment variable.
    **defaults
        Default values for ``input_type``, ``dimensions``, etc.

    Examples
    --------
    ::

        from tools.embedding import Embedding

        embedder = Embedding("voyage/voyage-3-large")
        result   = embedder.embed(
            "Explain transformer architecture",
            input_type="query",
        )
        print(result.dimensions)   # 1024
    """

    provider   = "voyage"
    _ENV_KEY   = "VOYAGE_API_KEY"
    _MAX_BATCH = 1000

    def _embed_chunk(
        self,
        texts:      list[str],
        *,
        input_type: "str | None",
        dimensions: "int | None",
        **options,
    ) -> "tuple[list[list[float]], int | None, dict]":
        """
        Send one ``/v1/embeddings`` request and return
        ``(embeddings, total_tokens, metadata)``.

        ``input_type`` is forwarded as-is (Voyage uses "query"/"document"
        natively).  ``dimensions`` is forwarded as ``output_dimension``.
        """
        body: dict = {
            "model": self.model,
            "input": texts,
        }
        if input_type is not None:
            body["input_type"] = input_type
        if dimensions is not None:
            body["output_dimension"] = dimensions

        headers = {"Authorization": f"Bearer {self._api_key}"}
        data    = self._request(f"{_BASE_URL}/v1/embeddings", body, headers)

        # Response: {"data": [{"embedding": [...], "index": i}, ...], "usage": {...}}
        items = sorted(data["data"], key=lambda x: x["index"])
        embeddings = [item["embedding"] for item in items]

        usage        = data.get("usage", {})
        total_tokens = usage.get("total_tokens")

        return embeddings, total_tokens, {}


# Backward-compatible alias
VoyageEmbedder = EmbeddingVoyage
