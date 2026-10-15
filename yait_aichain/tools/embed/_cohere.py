"""
``EmbeddingCohere``: the Cohere Embed v2 API (``POST /v2/embed``), 96 texts a
request. ``input_type`` is required there, so ``None`` is sent as
``search_document``. See docs/tools/retrieval.md.
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
    dimensions, batch_size
        Settings of the embedder; a call takes only ``input_type``.

    Examples
    --------
    ::

        from yait_aichain.tools import embed

        embedder = embed("cohere", model="embed-v4.0")
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
        """One ``/v2/embed`` request; ``dimensions`` goes as ``output_dimension``."""
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


