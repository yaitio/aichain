"""``VectorDeleteTool``: remove records from a vector store by ids or by a
backend-native ``filter`` — exactly one, since "both" means different things
on different backends. See docs/utilities/index.md.
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
        opts = options or {}
        # The store refuses an empty delete, so an empty list cannot wipe the collection.
        self._with_collection(opts).delete(ids=input, filter=opts.get("filter"))
        return {"deleted": True}
