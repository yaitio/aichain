"""
tools.vectordb._query
======================

VectorQueryTool — semantic similarity search (main RAG tool).

This is the primary tool for retrieval-augmented generation pipelines.
Drop it into any Chain step; it embeds the query text and returns the
most relevant documents from the vector store.

Chain usage
-----------
::

    from tools.vectordb import VectorDB, vectorQuery
    from tools.embedding import Embedding

    store = VectorDB(
        "chroma",
        collection = "knowledge_base",
        embedder   = Embedding("openai/text-embedding-3-small"),
    )
    tool = vectorQuery(store)

    # Standalone
    results = tool.run("how does KV caching work?", {"n": 5})
    # → [{"id": "...", "text": "...", "score": 0.91, "metadata": {...}}, ...]

    # Inside a Chain step — output_key "context" flows into the next Skill
    (tool, "context", {"input": "{user_question}"})
"""

from __future__ import annotations

from .._base  import Tool
from ._base   import VectorStore, VectorRecord


class VectorQueryTool(Tool):
    """
    Semantic similarity search over a :class:`VectorStore`.

    Parameters
    ----------
    store : VectorStore
        The configured vector store (backend + collection + embedder).

    ``run()`` input / options
    -------------------------
    input   : str
        The query text.
    options : dict, optional
        ``n``          int  — number of results to return (default 5)
        ``filter``     dict — provider-native metadata filter
        ``collection`` str  — override the store's default collection

    Returns
    -------
    list[dict]
        JSON-serialisable list of records:
        ``[{"id": "...", "text": "...", "score": 0.91, "metadata": {...}}, ...]``
        Sorted by descending similarity score (highest first).
    """

    name        = "vector_query"
    description = (
        "Semantic similarity search — embed the query text and return the "
        "most relevant documents from the vector store.  Use this as the "
        "retrieval step in any RAG pipeline."
    )
    parameters  = {
        "type": "object",
        "properties": {
            "input": {
                "type":        "string",
                "description": "The query text to search for semantically similar documents.",
            },
            "options": {
                "type":        "object",
                "description": "Search configuration.",
                "properties": {
                    "n": {
                        "type":        "integer",
                        "description": "Number of results to return (default 5).",
                    },
                    "filter": {
                        "type":        "object",
                        "description": "Provider-native metadata filter applied before ranking.",
                    },
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

    def run(self, input: str, options: dict | None = None) -> list[dict]:
        opts   = options or {}
        n      = int(opts.get("n", 5))
        filter = opts.get("filter")
        store  = self._with_collection(opts)
        return [r.to_dict() for r in store.query(input, n=n, filter=filter)]

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _with_collection(self, opts: dict) -> VectorStore:
        """Return a store scoped to the overridden collection, if provided."""
        col = opts.get("collection")
        if col and col != self._store._collection:
            return VectorStore(self._store._backend, col, self._store._embedder)
        return self._store

    def __repr__(self) -> str:
        return f"VectorQueryTool(store={self._store!r})"


# ── Convenience alias (follows naming convention: camelCase factory) ───────────
def vectorQuery(store: VectorStore) -> VectorQueryTool:
    """Return a :class:`VectorQueryTool` bound to *store*."""
    return VectorQueryTool(store)
