"""
tools.embedding._cohere — EmbeddingCohere
==========================================

Wraps the Cohere Embed v2 API (``POST /v2/embed``).

Supported models
----------------
  embed-v4.0                 1024-dim default; supports output_dimension
  embed-english-v3.0         1024-dim; supports output_dimension
  embed-multilingual-v3.0    1024-dim; supports output_dimension

Model notes
-----------
* All v3 / v4 models support:
    - ``input_type``       — required by the API; defaults to "search_document"
      when ``None`` is passed so the request never errors.
    - ``output_dimension`` — truncate/expand the embedding vector size.
    - ``embedding_types``  — always sent as ``["float"]``; extended types
      (int8, uint8, binary, ubinary) are not exposed by this wrapper.
* The API accepts up to 96 strings per request.

Universal ``input_type`` mapping
---------------------------------
  "query"    → "search_query"
  "document" → "search_document"
  None       → "search_document"  (safe default)

Environment variable
--------------------
COHERE_API_KEY
"""

from ._base import Embedder

_BASE_URL = "https://api.cohere.com"

# Universal → Cohere native input_type mapping
_INPUT_TYPE_MAP = {
    "query":    "search_query",
    "document": "search_document",
}
_DEFAULT_INPUT_TYPE = "search_document"


class EmbeddingCohere(Embedder):
    """
    Embedder for the Cohere Embed API (v2).

    Parameters
    ----------
    model : str
        Model identifier, e.g. ``"embed-v4.0"``.
    api_key : str | None, optional
        Override ``COHERE_API_KEY`` environment variable.
    **defaults
        Default values for ``input_type``, ``dimensions``, etc.

    Examples
    --------
    ::

        from tools.embedding import Embedding

        embedder = Embedding("cohere/embed-v4.0")
        result   = embedder.embed(
            ["What is RAG?", "Retrieval-Augmented Generation combines …"],
            input_type="query",
        )
        print(result.dimensions)   # 1024
        print(result.metadata)     # {"billed_units": {"input_tokens": N}}
    """

    provider   = "cohere"
    _ENV_KEY   = "COHERE_API_KEY"
    _MAX_BATCH = 96

    def _embed_chunk(
        self,
        texts:      list[str],
        *,
        input_type: "str | None",
        dimensions: "int | None",
        **options,
    ) -> "tuple[list[list[float]], int | None, dict]":
        """
        Send one ``/v2/embed`` request and return
        ``(embeddings, total_tokens, metadata)``.

        ``input_type`` is mapped to Cohere's native terms.
        ``dimensions`` is forwarded as ``output_dimension`` when provided.
        """
        cohere_input_type = _INPUT_TYPE_MAP.get(
            input_type or "", _DEFAULT_INPUT_TYPE
        )

        body: dict = {
            "model":           self.model,
            "texts":           texts,
            "input_type":      cohere_input_type,
            "embedding_types": ["float"],
        }
        if dimensions is not None:
            body["output_dimension"] = dimensions

        headers = {
            "Authorization": f"Bearer {self._api_key}",
        }
        data = self._request(f"{_BASE_URL}/v2/embed", body, headers)

        # Response: {"embeddings": {"float": [[...], ...]}, "meta": {...}}
        embeddings = data["embeddings"]["float"]

        meta         = data.get("meta", {})
        billed       = meta.get("billed_units", {})
        total_tokens = billed.get("input_tokens")

        metadata = {"billed_units": billed} if billed else {}

        return embeddings, total_tokens, metadata


# Backward-compatible alias
CohereEmbedder = EmbeddingCohere
