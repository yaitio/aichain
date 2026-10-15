"""``Vectors``: a vector database — Chroma, Pinecone or Qdrant — as a utility,
with ``create``/``upsert``/``fetch``/``delete``/``count`` for code and
``kb.query`` the one tool an agent gets. Underneath: ``VectorDB()`` builds the
``VectorStore`` (``kb.store``), and the ``Vector*Tool`` classes wrap it.
See docs/utilities/index.md.
"""

from ._base    import PartialUpsert, VectorRecord, VectorBackend, VectorStore
from ._query import VectorQueryTool
from ._upsert import VectorUpsertTool
from ._fetch import VectorFetchTool
from ._delete import VectorDeleteTool

from .providers._chroma   import ChromaBackend
from .providers._pinecone import PineconeBackend
from .providers._qdrant   import QdrantBackend


__all__ = ["Vectors", "VectorDB", "VectorStore", "VectorRecord", "VectorBackend", "ChromaBackend",
           "PineconeBackend", "QdrantBackend", "VectorQueryTool", "VectorUpsertTool",
           "VectorFetchTool", "VectorDeleteTool", "PartialUpsert"]


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

_PROVIDER_MAP: dict[str, type] = {
    "chroma":    ChromaBackend,
    "chromadb":  ChromaBackend,
    "pinecone":  PineconeBackend,
    "qdrant":    QdrantBackend,
}


# ---------------------------------------------------------------------------
# VectorDB factory
# ---------------------------------------------------------------------------

def VectorDB(
    provider:   str,
    collection: str,
    *,
    embedder:   "object | None" = None,
    url:        str | None      = None,
    api_key:    str | None      = None,
    **kwargs,
) -> VectorStore:
    """
    Return a :class:`VectorStore` for *provider* — what :class:`Vectors` is
    built on, and ``kb.store`` on one.

    Parameters
    ----------
    provider : str
        Backend name: ``"chroma"``, ``"pinecone"``, or ``"qdrant"``.
        Aliases: ``"chromadb"``.
    collection : str
        Collection (Chroma/Qdrant) or namespace (Pinecone) used for all
        operations.  It is the store's setting; a tool call cannot change it.
    embedder : Embedder | None, optional
        An embedder, as made by ``embed(...)``, used to auto-vectorise
        text on ``upsert()`` and ``query()``.  When ``None``, callers must
        supply pre-computed ``vector`` fields in their records.
    url : str | None, optional
        Base URL of the server.  Overrides the provider's environment variable
        default (``CHROMA_URL``, ``QDRANT_URL``).
    api_key : str | None, optional
        API key.  Overrides the provider's environment variable
        (``CHROMA_API_KEY``, ``PINECONE_API_KEY``, ``QDRANT_API_KEY``).
    **kwargs
        Additional provider-specific arguments forwarded to the backend
        constructor.  Common extras:
          ``index``  (str)  — Pinecone index name
          ``cloud``  (str)  — Pinecone cloud (default ``"aws"``)
          ``region`` (str)  — Pinecone region (default ``"us-east-1"``)

    Returns
    -------
    VectorStore
        Ready-to-use store with ``create``, ``query``, ``upsert``, ``fetch``,
        ``delete``, and ``count`` methods.

    Raises
    ------
    ValueError
        When *provider* is not recognised.

    Examples
    --------
    ::

        from yait_aichain.utilities.vectors import VectorDB
        from yait_aichain.tools import embed

        store = VectorDB(
            "chroma",
            collection = "docs",
            embedder   = embed("openai", model="text-embedding-3-small"),
        )
    """
    key = provider.lower().strip()
    cls = _PROVIDER_MAP.get(key)
    if cls is None:
        supported = ", ".join(sorted(_PROVIDER_MAP))
        raise ValueError(
            f"Unknown vector DB provider {provider!r}.  "
            f"Supported: {supported}."
        )

    backend_kwargs: dict = {}
    if url:
        backend_kwargs["url"] = url
    if api_key:
        backend_kwargs["api_key"] = api_key
    backend_kwargs.update(kwargs)

    backend = cls(**backend_kwargs)
    return VectorStore(backend=backend, collection=collection, embedder=embedder)


# ── The utility ──────────────────────────────────────────────────────────
from ._utility import Vectors
