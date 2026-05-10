"""
tools.vectordb
==============

Provider-agnostic vector database interface for aichain.

Public API
----------
``VectorDB(provider, collection, *, embedder, url, api_key, ...)``
    Factory function — returns a :class:`VectorStore` bound to the
    requested backend.  Mirrors the ``Embedding(model)`` factory pattern.

``VectorStore``
    Orchestrator wrapping a backend + optional Embedder.
    Methods: ``create``, ``query``, ``upsert``, ``fetch``, ``delete``, ``count``.

``VectorRecord``
    Uniform data model: ``id``, ``text``, ``score``, ``metadata``.

Tools (bind to a VectorStore instance)
---------------------------------------
``vectorQuery(store)``   → VectorQueryTool   — semantic RAG retrieval  [main]
``vectorUpsert(store)``  → VectorUpsertTool  — insert / update documents
``vectorFetch(store)``   → VectorFetchTool   — retrieve by ID
``vectorDelete(store)``  → VectorDeleteTool  — delete by ID or filter

Provider routing
----------------
``provider`` may be a full name or common alias:

  "chroma"   / "chromadb"  → ChromaBackend   (default URL: http://localhost:8000)
  "pinecone"               → PineconeBackend (requires api_key or PINECONE_API_KEY)
  "qdrant"                 → QdrantBackend   (default URL: http://localhost:6333)

Constructor kwargs are forwarded to the backend:
  ``url``      — server base URL (overrides env var default)
  ``api_key``  — API key (overrides env var default)
  ``index``    — Pinecone index name (or set PINECONE_INDEX env var)
  ``cloud``    — Pinecone cloud for new index creation (default "aws")
  ``region``   — Pinecone region for new index creation (default "us-east-1")

Examples
--------
Chroma (local, auto-embed)::

    from tools.vectordb  import VectorDB, vectorQuery, vectorUpsert
    from tools.embedding import Embedding

    store = VectorDB(
        "chroma",
        collection = "knowledge_base",
        embedder   = Embedding("openai/text-embedding-3-small"),
        url        = "http://localhost:8000",
    )
    store.create(dimension=1536)

    vectorUpsert(store).run([
        {"id": "1", "text": "LLMs use KV cache to avoid recomputing attention."},
        {"id": "2", "text": "RAG grounds generation in retrieved documents."},
    ])

    results = vectorQuery(store).run("how does caching work in LLMs?", {"n": 3})
    for r in results:
        print(r["score"], r["text"][:60])

Pinecone (cloud, pre-computed vectors)::

    store = VectorDB(
        "pinecone",
        collection = "my_namespace",
        index      = "my-index",
        api_key    = "pk-...",
    )
    store.upsert([{"id": "1", "text": "…", "vector": [0.1, 0.2, …]}])

Qdrant (local)::

    store = VectorDB("qdrant", collection="docs",
                     embedder=Embedding("cohere/embed-v4.0"))
    store.create(dimension=1024, metric="dot")

In a Chain::

    from chain import Chain
    from skills import Skill
    from models import Model

    retriever = vectorQuery(store)
    answer_skill = Skill(
        model = Model("claude-sonnet-4-6"),
        input = {
            "messages": [{
                "role": "user",
                "parts": [{"type": "text",
                           "text": "Context:\n{context}\n\nQuestion: {question}"}],
            }]
        },
        output = {"modalities": ["text"]},
    )
    pipeline = Chain(
        steps = [
            (retriever, "context", {"input": "{question}"}),
            answer_skill,
        ]
    )
    pipeline.run(variables={"question": "How does KV caching work?"})
"""

from ._base    import VectorRecord, VectorBackend, VectorStore
from ._chunk   import VectorChunkTool,  vectorChunk
from ._query   import VectorQueryTool,  vectorQuery
from ._upsert  import VectorUpsertTool, vectorUpsert
from ._fetch   import VectorFetchTool,  vectorFetch
from ._delete  import VectorDeleteTool, vectorDelete

from .providers._chroma   import ChromaBackend
from .providers._pinecone import PineconeBackend
from .providers._qdrant   import QdrantBackend


__all__ = [
    # Factory
    "VectorDB",
    # Core classes
    "VectorStore",
    "VectorRecord",
    "VectorBackend",
    # Provider backends
    "ChromaBackend",
    "PineconeBackend",
    "QdrantBackend",
    # Tools — factories (camelCase, follow existing convention)
    "vectorChunk",
    "vectorQuery",
    "vectorUpsert",
    "vectorFetch",
    "vectorDelete",
    # Tool classes (PascalCase, for isinstance / subclassing)
    "VectorChunkTool",
    "VectorQueryTool",
    "VectorUpsertTool",
    "VectorFetchTool",
    "VectorDeleteTool",
]


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
    Return a :class:`VectorStore` for *provider*.

    Parameters
    ----------
    provider : str
        Backend name: ``"chroma"``, ``"pinecone"``, or ``"qdrant"``.
        Aliases: ``"chromadb"``.
    collection : str
        Default collection (Chroma/Qdrant) or namespace (Pinecone) used for
        all operations.  Can be overridden per-call via ``options["collection"]``.
    embedder : Embedder | None, optional
        An :class:`~tools.embedding.Embedder` instance used to auto-vectorise
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

        from tools.vectordb  import VectorDB
        from tools.embedding import Embedding

        store = VectorDB(
            "chroma",
            collection = "docs",
            embedder   = Embedding("openai/text-embedding-3-small"),
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
