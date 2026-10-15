"""
``EmbeddingQwen``: DashScope embeddings through its OpenAI-compatible endpoint
(``/compatible-mode/v1/embeddings``); the host follows ``region`` or
``DASHSCOPE_REGION``. See docs/tools/retrieval.md.
"""

from __future__ import annotations


from ._base import _OpenAIShaped


# Models that accept a variable ``dimensions`` parameter
_SUPPORTS_DIMENSIONS = frozenset({
    "text-embedding-v4",
    "text-embedding-v3",
})


class EmbeddingQwen(_OpenAIShaped):
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
    dimensions, batch_size
        Settings of the embedder; a call takes only ``input_type``.

    Examples
    --------
    ::

        from yait_aichain.tools import embed

        e = embed("qwen", model="text-embedding-v4")
        r = e.embed("What is machine learning?", input_type="query")
        print(r.dimensions)   # 1024

        # Custom dimensions
        e = embed("qwen", model="text-embedding-v4", dimensions=512)
        r = e.embed(["text A", "text B"])
        print(len(r.embeddings[0]))   # 512
    """

    provider   = "qwen"
    _DIMENSION_MODELS = _SUPPORTS_DIMENSIONS
    _ENV_KEY   = "DASHSCOPE_API_KEY"
    _MAX_BATCH = 25
    _INPUT_TYPE = True

    def __init__(
        self,
        model:   str,
        *,
        api_key: str | None = None,
        region:  str | None = None,
        **defaults,
    ) -> None:
        self._region = region
        super().__init__(model, api_key=api_key, **defaults)

    def _max_rows(self) -> int:
        # text-embedding-v3 and v4 take ten rows a request; v1 and v2 take 25.
        return 10 if self.model.startswith(("text-embedding-v3", "text-embedding-v4")) else 25

    def _url(self) -> str:
        from ...clients._families.qwen import resolve_qwen_base_url
        return resolve_qwen_base_url(self._region) + "/compatible-mode/v1/embeddings"
