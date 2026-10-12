"""
utilities.vectors._query
========================

VectorQueryTool — semantic similarity search (main RAG tool).

It embeds the query text and returns the most relevant documents from the
vector store. ``Vectors`` builds one as ``kb.query``, the tool an agent gets.

Usage
-----
::

    from yait_aichain.utilities.vectors import VectorDB, VectorQueryTool
    from yait_aichain.tools import embed

    store = VectorDB(
        "chroma",
        collection = "knowledge_base",
        embedder   = embed("openai", model="text-embedding-3-small"),
    )
    tool = VectorQueryTool(store)

    # Standalone
    results = tool.run("how does KV caching work?", {"n": 5})
    # → [{"id": "...", "text": "...", "score": 0.91, "metadata": {...}}, ...]

    # Inside a Chain step — output_key "context" flows into the next Skill
    (tool, "context", {"input": "{user_question}"})
"""

from __future__ import annotations

from ._tool   import _VectorTool


#: The most results one query returns to a model: enough for any context
#: window worth filling, few enough that a model asking for 10,000 does not
#: pull a collection into its prompt.
MAX_N = 100


class VectorQueryTool(_VectorTool):
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
                        "minimum":     1,
                        "maximum":     MAX_N,
                        "description": "Number of results to return (default 5).",
                    },
                    "filter": {
                        "type":        "object",
                        "description": "Provider-native metadata filter applied before ranking.",
                    },
                },
            },
        },
        "required": ["input"],
    }


    def run(self, input: str, options: dict | None = None) -> list[dict]:
        opts   = options or {}
        n      = int(opts.get("n", 5))
        if not 1 <= n <= MAX_N:
            raise ValueError(f"{self.name}: n is between 1 and {MAX_N}, not {n}")
        filter = opts.get("filter")
        store  = self._with_collection(opts)
        return [r.to_dict() for r in store.query(input, n=n, filter=filter)]
