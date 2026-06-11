"""
tools.vectordb._upsert
=======================

VectorUpsertTool — insert or update documents in a vector store.

Typical use: ingestion pipeline step that indexes new content before
the RAG query step can retrieve it.

Chain usage
-----------
::

    from tools.vectordb import VectorDB, vectorUpsert
    from tools.embedding import Embedding

    store = VectorDB(
        "qdrant",
        collection = "docs",
        embedder   = Embedding("openai/text-embedding-3-small"),
    )
    tool = vectorUpsert(store)

    # Standalone
    result = tool.run([
        {"id": "doc_1", "text": "LLMs use KV cache …", "metadata": {"source": "arxiv"}},
        {"id": "doc_2", "text": "Attention is all you need …"},
    ])
    # → {"upserted": 2}

    # Inside a Chain (receives a list of dicts from a previous step)
    (tool, "upsert_result", {"input": "{parsed_documents}"})
"""

from __future__ import annotations

from .._base import Tool
from ._base  import VectorStore


class VectorUpsertTool(Tool):
    """
    Insert or update records in a :class:`VectorStore`.

    When an :class:`~tools.embedding.Embedder` is configured on the store,
    text is embedded automatically in a single batched call.  Records that
    already include a ``vector`` field bypass embedding entirely.

    Parameters
    ----------
    store : VectorStore
        The configured vector store (backend + collection + embedder).

    ``run()`` input / options
    -------------------------
    input   : list[dict]
        Records to insert/update.  Each dict must have:
          ``id``       str         — unique identifier
          ``text``     str         — document content
          ``metadata`` dict        — optional payload / metadata
          ``vector``   list[float] — optional pre-computed embedding
    options : dict, optional
        ``collection`` str — override the store's default collection

    Returns
    -------
    dict
        ``{"upserted": N}`` — number of records written.
    """

    name        = "vector_upsert"
    description = (
        "Insert or update documents in the vector store.  Text is "
        "embedded automatically when an embedder is configured.  "
        "Use this as the ingestion step before RAG retrieval."
    )
    parameters  = {
        "type": "object",
        "properties": {
            "input": {
                "type":        "array",
                "description": "List of records to insert or update.",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type":        "string",
                            "description": "Unique document identifier.",
                        },
                        "text": {
                            "type":        "string",
                            "description": "Document text content.",
                        },
                        "metadata": {
                            "type":        "object",
                            "description": "Optional key-value metadata / payload.",
                        },
                        "vector": {
                            "type":        "array",
                            "items":       {"type": "number"},
                            "description": (
                                "Optional pre-computed embedding vector.  "
                                "When present, auto-embedding is skipped for this record."
                            ),
                        },
                    },
                    "required": ["id", "text"],
                },
            },
            "options": {
                "type":        "object",
                "description": "Upsert options.",
                "properties": {
                    "collection": {
                        "type":        "string",
                        "description": "Override the store's default collection / namespace.",
                    },
                },
            },
        },
        "required": ["input"],
    }

    def __init__(self, store: VectorStore) -> None:
        self._store = store

    def run(self, input: list[dict], options: dict | None = None) -> dict:
        opts  = options or {}
        store = self._with_collection(opts)
        n     = store.upsert(input)
        return {"upserted": n}

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _with_collection(self, opts: dict) -> VectorStore:
        col = opts.get("collection")
        if col and col != self._store._collection:
            return VectorStore(self._store._backend, col, self._store._embedder)
        return self._store

    def __repr__(self) -> str:
        return f"VectorUpsertTool(store={self._store!r})"


# ── Convenience alias ─────────────────────────────────────────────────────────
def vectorUpsert(store: VectorStore) -> VectorUpsertTool:
    """Return a :class:`VectorUpsertTool` bound to *store*."""
    return VectorUpsertTool(store)
