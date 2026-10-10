"""
utilities.vectors._upsert
=========================

VectorUpsertTool — insert or update documents in a vector store.

Typical use: ingestion pipeline step that indexes new content before
the RAG query step can retrieve it.

Usage
-----
::

    from yait_aichain.utilities.vectors import VectorDB, VectorUpsertTool
    from yait_aichain.tools import embed

    store = VectorDB(
        "qdrant",
        collection = "docs",
        embedder   = embed("openai", model="text-embedding-3-small"),
    )
    tool = VectorUpsertTool(store)

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

from ...tools._base import Tool
from ._base  import VectorStore


class VectorUpsertTool(Tool):
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

    def __init__(self, store: VectorStore) -> None:
        self._store = store

    def run(self, input: list[dict], options: dict | None = None) -> dict:
        opts  = options or {}
        store = self._with_collection(opts)
        n     = store.upsert(input)
        return {"upserted": n}

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
        return f"VectorUpsertTool(store={self._store!r})"


