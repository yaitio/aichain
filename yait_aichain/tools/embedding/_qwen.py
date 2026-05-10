"""
tools.embedding._qwen — EmbeddingQwen
=======================================

Wraps the Alibaba DashScope Embeddings API via the OpenAI-compatible
endpoint (``POST /compatible-mode/v1/embeddings``).

Supported models
----------------
  text-embedding-v4    Latest generation.  Supports ``dimensions`` (64–2048).
                       Default dimensions: 1024.
  text-embedding-v3    Supports ``dimensions`` (512–8192).
                       Default dimensions: 1024.
  text-embedding-v2    Fixed 1536 dimensions; no dimension override.

Model notes
-----------
* ``input_type`` is mapped to DashScope's ``input_type`` parameter:
    ``"query"``     → ``"query"``
    ``"document"``  → ``"document"``
    ``None``        → omitted (provider picks the default)
* ``text-embedding-v4`` and ``text-embedding-v3`` support variable
  ``dimensions``; the parameter is silently omitted for other models.
* The API accepts up to 25 strings per request; the base class handles
  automatic chunking for larger inputs.

Region
------
The base URL is derived from the ``DASHSCOPE_REGION`` environment variable
(or the ``region`` constructor argument):
  ap (default) — https://dashscope-intl.aliyuncs.com
  us           — https://dashscope-us.aliyuncs.com
  cn           — https://dashscope.aliyuncs.com
  hk           — https://cn-hongkong.dashscope.aliyuncs.com

Environment variables
---------------------
DASHSCOPE_API_KEY
DASHSCOPE_REGION  (optional; default: ap)
"""

from __future__ import annotations

import os

from ._base import Embedder


# Models that accept a variable ``dimensions`` parameter
_SUPPORTS_DIMENSIONS = frozenset({
    "text-embedding-v4",
    "text-embedding-v3",
})

# Maps universal input_type to DashScope's native values
_INPUT_TYPE_MAP: dict[str, str] = {
    "query":    "query",
    "document": "document",
}


class EmbeddingQwen(Embedder):
    """
    Embedder for the Alibaba DashScope Embeddings API.

    Uses the OpenAI-compatible ``/compatible-mode/v1/embeddings`` endpoint.

    Parameters
    ----------
    model : str
        Model identifier, e.g. ``"text-embedding-v4"``.
    api_key : str | None, optional
        Override the ``DASHSCOPE_API_KEY`` environment variable.
    region : str | None, optional
        Region selector: ``"ap"`` (default), ``"us"``, ``"cn"``, ``"hk"``.
        Overrides the ``DASHSCOPE_REGION`` env var.
    **defaults
        Default values for ``input_type``, ``dimensions``, etc., forwarded
        to every :meth:`~Embedder.embed` call.

    Examples
    --------
    ::

        from tools.embedding import Embedding

        # Provider-prefixed
        e = Embedding("qwen/text-embedding-v4")
        r = e.embed("What is machine learning?", input_type="query")
        print(r.dimensions)   # 1024

        # Custom dimensions
        e = Embedding("qwen/text-embedding-v4", dimensions=512)
        r = e.embed(["text A", "text B"])
        print(len(r.embeddings[0]))   # 512
    """

    provider   = "qwen"
    _ENV_KEY   = "DASHSCOPE_API_KEY"
    _MAX_BATCH = 25

    def __init__(
        self,
        model:   str,
        *,
        api_key: "str | None" = None,
        region:  "str | None" = None,
        **defaults,
    ) -> None:
        self._region = region
        super().__init__(model, api_key=api_key, **defaults)

    def _embed_chunk(
        self,
        texts:      list[str],
        *,
        input_type: "str | None",
        dimensions: "int | None",
        **options,
    ) -> "tuple[list[list[float]], int | None, dict]":
        """
        Send one ``/compatible-mode/v1/embeddings`` request to DashScope.

        Returns
        -------
        (embeddings, total_tokens, metadata)
        """
        from clients._qwen import resolve_qwen_base_url

        base_url = resolve_qwen_base_url(self._region)
        url      = base_url + "/compatible-mode/v1/embeddings"

        body: dict = {
            "model":           self.model,
            "input":           texts,
            "encoding_format": "float",
        }

        # Optional: variable dimensions (v3 and v4 only)
        if dimensions is not None and self.model in _SUPPORTS_DIMENSIONS:
            body["dimensions"] = dimensions

        # Optional: input_type hint
        native_type = _INPUT_TYPE_MAP.get(input_type or "")
        if native_type:
            body["input_type"] = native_type

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type":  "application/json",
        }

        data = self._request(url, body, headers)

        # Response: {"data": [{"embedding": [...], "index": i}, ...], "usage": {...}}
        items      = sorted(data["data"], key=lambda x: x["index"])
        embeddings = [item["embedding"] for item in items]

        usage        = data.get("usage", {})
        total_tokens = usage.get("total_tokens")

        return embeddings, total_tokens, {}
