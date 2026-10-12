"""
tools.embed._qwen — EmbeddingQwen
=================================

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
  ``dimensions``; on any other model it is refused when the embedder is made.
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


from ._base import _OpenAIShaped


# Models that accept a variable ``dimensions`` parameter
_SUPPORTS_DIMENSIONS = frozenset({
    "text-embedding-v4",
    "text-embedding-v3",
})

# Maps universal input_type to DashScope's native values


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

    def _url(self) -> str:
        from ...clients._families.qwen import resolve_qwen_base_url
        return resolve_qwen_base_url(self._region) + "/compatible-mode/v1/embeddings"
