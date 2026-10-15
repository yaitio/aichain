"""
``EmbeddingGoogle``: the Gemini Embeddings API (``batchEmbedContents``), 100
texts a request. It reports no token counts, so ``total_tokens`` is ``None``.
See docs/tools/retrieval.md.
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
    dimensions, batch_size
        Settings of the embedder; a call takes only ``input_type``.

    Examples
    --------
    ::

        from yait_aichain.tools import embed

        embedder = embed("google", model="gemini-embedding-001")
        result   = embedder.embed(
            ["What is RAG?", "RAG stands for Retrieval-Augmented Generation."],
            input_type="query",
        )
        print(result.dimensions)   # 768
    """

    provider   = "google"
    _DIMENSION_MODELS = _SUPPORTS_DIMENSIONS
    _ENV_KEY   = ("GOOGLE_AI_API_KEY", "GOOGLE_API_KEY")
    _MAX_BATCH = 100

    def _embed_chunk(
        self,
        texts:      list[str],
        *,
        input_type: "str | None",
        dimensions: "int | None",
        **options,
    ) -> "tuple[list[list[float]], int | None, dict]":
        """One ``batchEmbedContents`` request, a request object per text; ``task_type``
        and ``output_dimensionality`` only where the model takes them."""
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
        path = f"/v1beta/models/{self.model}:batchEmbedContents"
        # Key travels in the x-goog-api-key header — keeping it out of the
        # URL keeps it out of proxy/server logs and tracebacks.
        data = self._request(
            f"{_BASE_URL}{path}",
            body,
            extra_headers={"x-goog-api-key": self._api_key},
        )

        # Response: {"embeddings": [{"values": [...]}]}
        embeddings = [item["values"] for item in data.get("embeddings", [])]

        # Google does not return token counts from batchEmbedContents
        return embeddings, None, {}


