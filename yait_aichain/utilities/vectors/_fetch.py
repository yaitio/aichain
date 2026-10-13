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

from ._tool  import _VectorTool


class VectorFetchTool(_VectorTool):
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

    risk = "read"   # reads the collection

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


    def run(self, input: list[str], options: dict | None = None) -> list[dict]:
        opts  = options or {}
        store = self._with_collection(opts)
        return [r.to_dict() for r in store.fetch(input)]
