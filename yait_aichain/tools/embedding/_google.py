"""
tools.embedding._google — EmbeddingGoogle
==========================================

Wraps the Google AI (Gemini) Embeddings API
(``POST /v1beta/models/{model}:batchEmbedContents``).

Supported models
----------------
  gemini-embedding-001          768-dim default (v1); supports task_type and
                                output_dimensionality
  gemini-embedding-exp-03-07    3072-dim default; experimental
  text-embedding-004            768-dim; older stable model

Model notes
-----------
* Authentication is via query-string ``?key=<GOOGLE_AI_API_KEY>`` (no header).
* The ``batchEmbedContents`` endpoint accepts a list of
  ``EmbedContentRequest`` objects in one POST, making it naturally batched.
* ``task_type`` (RETRIEVAL_QUERY, RETRIEVAL_DOCUMENT, etc.) is supported only
  by ``gemini-embedding-001``; it is silently omitted for other models.
* ``output_dimensionality`` truncates the vector; only ``gemini-embedding-001``
  and newer models support it — omitted otherwise.
* The API accepts up to 100 requests per batch call.
* Google does not return token counts from ``batchEmbedContents``; ``total_tokens``
  is always ``None``.

Universal ``input_type`` mapping
---------------------------------
  "query"    → "RETRIEVAL_QUERY"
  "document" → "RETRIEVAL_DOCUMENT"
  None       → omitted (provider default)

Environment variable
--------------------
GOOGLE_AI_API_KEY
"""

from ._base import Embedder

_BASE_URL = "https://generativelanguage.googleapis.com"

# Only these models support task_type; others must omit it
_SUPPORTS_TASK_TYPE = frozenset({
    "gemini-embedding-001",
    "text-embedding-004",
})

# Only these models support output_dimensionality truncation
_SUPPORTS_DIMENSIONS = frozenset({
    "gemini-embedding-001",
    "gemini-embedding-exp-03-07",
})

# Universal → Google task_type mapping
_INPUT_TYPE_MAP = {
    "query":    "RETRIEVAL_QUERY",
    "document": "RETRIEVAL_DOCUMENT",
}


class EmbeddingGoogle(Embedder):
    """
    Embedder for the Google AI (Gemini) Embeddings API.

    Parameters
    ----------
    model : str
        Model identifier, e.g. ``"gemini-embedding-001"``.
    api_key : str | None, optional
        Override ``GOOGLE_AI_API_KEY`` environment variable.
    **defaults
        Default values for ``input_type``, ``dimensions``, etc.

    Examples
    --------
    ::

        from tools.embedding import Embedding

        embedder = Embedding("google/gemini-embedding-001")
        result   = embedder.embed(
            ["What is RAG?", "RAG stands for Retrieval-Augmented Generation."],
            input_type="query",
        )
        print(result.dimensions)   # 768
    """

    provider   = "google"
    _ENV_KEY   = "GOOGLE_AI_API_KEY"
    _MAX_BATCH = 100

    def _embed_chunk(
        self,
        texts:      list[str],
        *,
        input_type: "str | None",
        dimensions: "int | None",
        **options,
    ) -> "tuple[list[list[float]], int | None, dict]":
        """
        Send one ``batchEmbedContents`` request and return
        ``(embeddings, None, metadata)``.

        Google's batch endpoint wraps each text in an ``EmbedContentRequest``
        with model + content fields.  ``task_type`` and
        ``output_dimensionality`` are included only when the model supports them.
        """
        task_type      = _INPUT_TYPE_MAP.get(input_type or "")
        supports_task  = self.model in _SUPPORTS_TASK_TYPE
        supports_dims  = self.model in _SUPPORTS_DIMENSIONS

        requests = []
        for text in texts:
            req: dict = {
                "model":   f"models/{self.model}",
                "content": {"parts": [{"text": text}]},
            }
            if task_type and supports_task:
                req["task_type"] = task_type
            if dimensions is not None and supports_dims:
                req["output_dimensionality"] = dimensions
            requests.append(req)

        body = {"requests": requests}
        path = (
            f"/v1beta/models/{self.model}:batchEmbedContents"
            f"?key={self._api_key}"
        )
        data = self._request(f"{_BASE_URL}{path}", body)

        # Response: {"embeddings": [{"values": [...]}]}
        embeddings = [item["values"] for item in data.get("embeddings", [])]

        # Google does not return token counts from batchEmbedContents
        return embeddings, None, {}


# Backward-compatible alias
GoogleEmbedder = EmbeddingGoogle
