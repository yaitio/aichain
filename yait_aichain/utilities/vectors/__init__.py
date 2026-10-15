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
        Backend name: ``"chroma"`` (or ``"chromadb"``), ``"pinecone"``, ``"qdrant"``.
    collection : str
        Collection (Chroma/Qdrant) or namespace (Pinecone); a store setting.
    embedder : Embedder | None, optional
        Vectorises text on ``upsert()`` and ``query()``; ``None`` means records
        carry their own ``vector``.
    url : str | None, optional
        Server base URL; overrides ``CHROMA_URL`` / ``QDRANT_URL``.
    api_key : str | None, optional
        Overrides ``CHROMA_API_KEY`` / ``PINECONE_API_KEY`` / ``QDRANT_API_KEY``.
    **kwargs
        Forwarded to the backend: Pinecone's ``index``, ``cloud`` (``"aws"``)
        and ``region`` (``"us-east-1"``).

    Returns
    -------
    VectorStore
        With ``create``, ``query``, ``upsert``, ``fetch``, ``delete`` and ``count``.

    Raises
    ------
    ValueError
        When *provider* is not recognised.

    Examples
    --------
    ::

        store = VectorDB("chroma", collection="docs",
                         embedder=embed("openai", model="text-embedding-3-small"))
    """
    key = provider.lower().strip()
    cls = _PROVIDER_MAP.get(key)
    if cls is None:
        supported = ", ".join(sorted(_PROVIDER_MAP))
        raise ValueError(
            f"Unknown vector DB provider {provider!r}.  "
            f"Supported: {supported}."
        )

    backend = cls(url=url or None, api_key=api_key or None, **kwargs)
    return VectorStore(backend=backend, collection=collection, embedder=embedder)


# ── The utility ──────────────────────────────────────────────────────────
from ._utility import Vectors
