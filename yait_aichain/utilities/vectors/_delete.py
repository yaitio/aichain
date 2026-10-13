"""
utilities.vectors._delete
=========================

VectorDeleteTool — remove documents from the vector store.

Supports deletion by explicit IDs, by metadata filter, or both.

Usage
-----
::

    from yait_aichain.utilities.vectors import VectorDB, VectorDeleteTool

    store = VectorDB("chroma", collection="docs")
    tool  = VectorDeleteTool(store)

    # Delete by IDs
    tool.run(["doc_1", "doc_2"])

    # Delete by filter (provider-native filter dict)
    tool.run(None, {"filter": {"source": "arxiv"}})

    # Both
    tool.run(["doc_3"], {"filter": {"year": 2020}})
"""

from __future__ import annotations

from ._tool  import _VectorTool


class VectorDeleteTool(_VectorTool):
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

    Returns
    -------
    dict
        ``{"deleted": true}`` on success.
    """

    risk = "destructive"   # removes records, by a filter the model writes

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
                },
            },
        },
    }


    def run(
        self,
        input:   list[str] | None = None,
        options: dict | None       = None,
    ) -> dict:
        opts   = options or {}
        filter = opts.get("filter")
        store  = self._with_collection(opts)

        if not input and not filter:
            raise ValueError(
                "VectorDeleteTool: provide at least one non-empty 'input' "
                "(IDs) or options['filter'] for bulk deletion.  An empty "
                "list is rejected to avoid wiping the collection."
            )
        store.delete(ids=input, filter=filter)
        return {"deleted": True}
