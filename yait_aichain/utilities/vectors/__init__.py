"""
utilities.vectors
=================

A vector database as a utility, over Chroma, Pinecone or Qdrant.

Public API
----------
``Vectors(backend, collection, *, description, embedder, url, api_key, ...)``
    The way in: ``create``, ``upsert``, ``fetch``, ``delete`` and ``count``
    for code, and ``kb.query`` — the one tool an agent gets.

Underneath
----------
``VectorDB(provider, collection, *, embedder, url, api_key, ...)``
    Returns the :class:`VectorStore` a ``Vectors`` is built on (``kb.store``).

``VectorStore``
    A backend plus an optional embedder, bound to one collection.
    Methods: ``create``, ``query``, ``upsert``, ``fetch``, ``delete``, ``count``.

``VectorRecord``
    Uniform data model: ``id``, ``text``, ``score``, ``metadata``.

``VectorQueryTool``, ``VectorUpsertTool``, ``VectorFetchTool``, ``VectorDeleteTool``
    Tools constructed with a store. ``Vectors`` uses ``VectorQueryTool`` for
    ``kb.query``.

Backends
--------
``backend`` is a name, or an alias:

  "chroma"   / "chromadb"  → ChromaBackend   (default URL: http://localhost:8000)
  "pinecone"               → PineconeBackend (requires api_key or PINECONE_API_KEY)
  "qdrant"                 → QdrantBackend   (default URL: http://localhost:6333)

Further keyword arguments are forwarded to the backend:
  ``url``      — server base URL (overrides env var default)
  ``api_key``  — API key (overrides env var default)
  ``index``    — Pinecone index name (or set PINECONE_INDEX env var)
  ``cloud``    — Pinecone cloud for new index creation (default "aws")
  ``region``   — Pinecone region for new index creation (default "us-east-1")

Examples
--------
Qdrant, text embedded on the way in::

    from yait_aichain.utilities import Vectors
    from yait_aichain.tools import embed, chunk

    kb = Vectors("qdrant", collection="docs", description="the product documentation",
                 embedder=embed("openai", model="text-embedding-3-small"))
    kb.create(dimension=1536)

    pieces = chunk(max_chars=800).run(text)
    kb.upsert([{"id": f"doc-{i}", **p} for i, p in enumerate(pieces)])

    for r in kb.query.run("how does caching work?", {"n": 3}):
        print(r["score"], r["text"][:60])

Pinecone, vectors computed by the caller::

    kb = Vectors("pinecone", collection="my_namespace", description="support tickets",
                 index="my-index", api_key="pk-...")
    kb.upsert([{"id": "1", "text": "…", "vector": [0.1, 0.2, …]}])

Given to an agent::

    from yait_aichain.agent import Agent, step_count
    from yait_aichain.models import Model

    agent = Agent(Model("claude-haiku-5-5"), tools=[kb.query], stop_when=[step_count(10)])
"""

from ._base    import VectorRecord, VectorBackend, VectorStore
from ._query import VectorQueryTool
from ._upsert import VectorUpsertTool
from ._fetch import VectorFetchTool
from ._delete import VectorDeleteTool

from .providers._chroma   import ChromaBackend
from .providers._pinecone import PineconeBackend
from .providers._qdrant   import QdrantBackend


__all__ = ["Vectors", "VectorDB", "VectorStore", "VectorRecord", "VectorBackend", "ChromaBackend",
           "PineconeBackend", "QdrantBackend", "VectorQueryTool", "VectorUpsertTool",
           "VectorFetchTool", "VectorDeleteTool"]


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
