"""``VectorFetchTool``: records by id, with no similarity ranking and no score —
for ids already known, or to check what was indexed. See docs/utilities/index.md.
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
        return [r.to_dict() for r in self._with_collection(options or {}).fetch(input)]
