"""
tools.embedding._openai — EmbeddingOpenAI
==========================================

Wraps the OpenAI Embeddings API (``POST /v1/embeddings``).

Supported models
----------------
  text-embedding-3-small    1536-dim default; supports truncation to any size
  text-embedding-3-large    3072-dim default; supports truncation to any size
  text-embedding-ada-002    1536-dim; no dimension override

Model notes
-----------
* ``text-embedding-3-*`` accept a ``dimensions`` parameter (1–max-dim).
* ``text-embedding-ada-002`` ignores ``dimensions``; the parameter is silently
  omitted from the request body for that model.
* OpenAI does not distinguish query vs document embeddings; ``input_type`` is
  accepted by the interface but ignored.
* The API accepts up to 2 048 strings per request.

Environment variable
--------------------
OPENAI_API_KEY
"""

from ._base import Embedder

_BASE_URL = "https://api.openai.com"

# Models that accept the dimensions parameter
_SUPPORTS_DIMENSIONS = frozenset({
    "text-embedding-3-small",
    "text-embedding-3-large",
})


class EmbeddingOpenAI(Embedder):
    """
    Embedder for the OpenAI Embeddings API.

    Parameters
    ----------
    model : str
        Model identifier, e.g. ``"text-embedding-3-large"``.
    api_key : str | None, optional
        Override ``OPENAI_API_KEY`` environment variable.
    **defaults
        Default values for ``input_type``, ``dimensions``, etc.

    Examples
    --------
    ::

        from tools.embedding import Embedding

        embedder = Embedding("text-embedding-3-large")
        result   = embedder.embed("What is machine learning?", input_type="query")
        print(result.dimensions)   # 3072
    """

    provider   = "openai"
    _ENV_KEY   = "OPENAI_API_KEY"
    _MAX_BATCH = 2048

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

        ``input_type`` is accepted for interface compatibility but ignored
        (OpenAI does not distinguish query vs document).
        ``dimensions`` is forwarded only for models that support it.
        """
        body: dict = {
            "model":           self.model,
            "input":           texts,
            "encoding_format": "float",
        }
        if dimensions is not None and self.model in _SUPPORTS_DIMENSIONS:
            body["dimensions"] = dimensions

        headers = {"Authorization": f"Bearer {self._api_key}"}
        data    = self._request(f"{_BASE_URL}/v1/embeddings", body, headers)

        # Response: {"data": [{"embedding": [...], "index": i}, ...], "usage": {...}}
        items = sorted(data["data"], key=lambda x: x["index"])
        embeddings = [item["embedding"] for item in items]

        usage        = data.get("usage", {})
        total_tokens = usage.get("total_tokens")

        return embeddings, total_tokens, {}


# Backward-compatible alias
OpenAIEmbedder = EmbeddingOpenAI
