"""
tools.vectordb._delete
=======================

VectorDeleteTool — remove documents from the vector store.

Supports deletion by explicit IDs, by metadata filter, or both.

Chain usage
-----------
::

    from tools.vectordb import VectorDB, vectorDelete

    store = VectorDB("chroma", collection="docs")
    tool  = vectorDelete(store)

    # Delete by IDs
    tool.run(["doc_1", "doc_2"])

    # Delete by filter (provider-native filter dict)
    tool.run(None, {"filter": {"source": "arxiv"}})

    # Both
    tool.run(["doc_3"], {"filter": {"year": 2020}})
"""

from __future__ import annotations

from .._base import Tool
from ._base  import VectorStore


class VectorDeleteTool(Tool):
    """
    Delete records from a :class:`VectorStore`.

    At least one of *input* (IDs) or ``options["filter"]`` must be provided.

    Parameters
    ----------
    store : VectorStore
        The configured vector store.

    ``run()`` input / options
    -------------------------
    input   : list[str] | None
        Document IDs to delete.  Pass ``None`` to delete by filter only.
    options : dict, optional
        ``filter``     dict — provider-native metadata filter for bulk delete
        ``collection`` str  — override the store's default collection

    Returns
    -------
    dict
        ``{"deleted": true}`` on success.
    """

    name        = "vector_delete"
    description = (
        "Remove documents from the vector store by ID or metadata filter.  "
        "Pass IDs for targeted deletion or a filter dict for bulk removal."
    )
    parameters  = {
        "type": "object",
        "properties": {
            "input": {
                "description": "List of document IDs to delete, or null to delete by filter only.",
                "oneOf": [
                    {
                        "type":  "array",
                        "items": {"type": "string"},
                    },
                    {"type": "null"},
                ],
            },
            "options": {
                "type":        "object",
                "description": "Delete options.",
                "properties": {
                    "filter": {
                        "type":        "object",
                        "description": "Provider-native metadata filter for bulk deletion.",
                    },
                    "collection": {
                        "type":        "string",
                        "description": "Override the store's default collection / namespace.",
                    },
                },
            },
        },
    }

    def __init__(self, store: VectorStore) -> None:
        self._store = store

    def run(
        self,
        input:   list[str] | None = None,
        options: dict | None       = None,
    ) -> dict:
        opts   = options or {}
        filter = opts.get("filter")
        store  = self._with_collection(opts)

        if input is None and filter is None:
            raise ValueError(
                "VectorDeleteTool: provide at least one of 'input' (IDs) "
                "or options['filter'] for bulk deletion."
            )
        store.delete(ids=input, filter=filter)
        return {"deleted": True}

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _with_collection(self, opts: dict) -> VectorStore:
        col = opts.get("collection")
        if col and col != self._store._collection:
            return VectorStore(self._store._backend, col, self._store._embedder)
        return self._store

    def __repr__(self) -> str:
        return f"VectorDeleteTool(store={self._store!r})"


# ── Convenience alias ─────────────────────────────────────────────────────────
def vectorDelete(store: VectorStore) -> VectorDeleteTool:
    """Return a :class:`VectorDeleteTool` bound to *store*."""
    return VectorDeleteTool(store)
