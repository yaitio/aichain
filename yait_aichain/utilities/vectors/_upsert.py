"""``VectorUpsertTool``: insert or update records — ``{"id", "text",
"metadata"}`` dicts, embedded on the way in when the store has an embedder —
answering ``{"upserted": n}``. See docs/utilities/index.md.
"""

from __future__ import annotations

from ._tool  import _VectorTool


class VectorUpsertTool(_VectorTool):
    """
    Insert or update records in a :class:`VectorStore`.

    When an embedder (``embed(...)``) is configured on the store,
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

    Returns
    -------
    dict
        ``{"upserted": N}`` — number of records written.
    """

    risk = "write"   # writes to the collection

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
                },
            },
        },
        "required": ["input"],
    }


    def run(self, input: list[dict], options: dict | None = None) -> dict:
        return {"upserted": self._with_collection(options or {}).upsert(input)}
