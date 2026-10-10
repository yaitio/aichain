"""
utilities.vectors._fetch
========================

VectorFetchTool — retrieve specific records by ID (no similarity ranking).

Use this when you already know which IDs you need — e.g. after a query
that returned IDs, or to verify what was indexed.

Usage
-----
::

    from yait_aichain.utilities.vectors import VectorDB, VectorFetchTool

    store = VectorDB("qdrant", collection="docs")
    tool  = VectorFetchTool(store)

    records = tool.run(["doc_1", "doc_2"])
    # → [{"id": "doc_1", "text": "…", "metadata": {…}}, …]
"""

from __future__ import annotations

from ...tools._base import Tool
from ._base  import VectorStore


class VectorFetchTool(Tool):
    """
    Retrieve records from a :class:`VectorStore` by exact ID.

    Parameters
    ----------
    store : VectorStore
        The configured vector store.

    ``run()`` input / options
    -------------------------
    input   : list[str]
        Document IDs to retrieve.
    options : dict, optional

    Returns
    -------
    list[dict]
        Records without a ``score`` field (score is only set on query results).
    """

    name        = "vector_fetch"
    description = (
        "Retrieve specific documents from the vector store by ID.  "
        "Returns full text and metadata without similarity ranking."
    )
    parameters  = {
        "type": "object",
        "properties": {
            "input": {
                "type":        "array",
                "items":       {"type": "string"},
                "description": "List of document IDs to retrieve.",
            },
            "options": {
                "type":        "object",
                "description": "Fetch options.",
                "properties": {
                },
            },
        },
        "required": ["input"],
    }

    def __init__(self, store: VectorStore) -> None:
        self._store = store

    def run(self, input: list[str], options: dict | None = None) -> list[dict]:
        opts  = options or {}
        store = self._with_collection(opts)
        return [r.to_dict() for r in store.fetch(input)]

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _with_collection(self, opts: dict) -> VectorStore:
        """The store. Its collection is a setting, not a call's: a
        ``collection`` option is refused, so a model cannot read or write
        another collection than the one the code chose."""
        if "collection" in opts:
            raise ValueError(f"{self.name}: collection is the store's setting, not a call's — "
                             "VectorDB(…, collection=…) or Vectors(…, collection=…)")
        return self._store

    def __repr__(self) -> str:
        return f"VectorFetchTool(store={self._store!r})"


