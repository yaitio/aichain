"""utilities.vectors._utility — ``Vectors``, a vector database as a utility."""

from __future__ import annotations

from .._base import Utility


class Vectors(Utility):
    """
    A vector database: ``create``, ``upsert``, ``fetch``, ``delete`` and
    ``count`` for code; ``query`` a tool an agent can take.

    ::

        kb = Vectors("qdrant", collection="docs", description="the product documentation",
                     embedder=embed("openai", model="text-embedding-3-small"))
        kb.create(dimension=1536)
        pieces = chunk(max_chars=800).run(text)                 # no ids: a record needs one
        kb.upsert([{"id": f"doc-{i}", **p} for i, p in enumerate(pieces)])
        kb.query.run("how does caching work?", {"n": 5})
        Agent(model, tools=[kb.query])            # the agent searches, and can do nothing else

    ``description`` says what the database holds, and is required: an agent
    with two knowledge bases cannot tell them apart without it, and the search
    tool's own description — "semantic similarity search" — says nothing of
    what is searched. ``query`` reads "Search the product documentation — …".

    ``backend`` is ``chroma``, ``pinecone`` or ``qdrant``; ``collection``,
    ``embedder``, ``url``, ``api_key`` and the backend's own settings
    (Pinecone's ``index``, ``cloud``, ``region``) are what ``VectorDB`` takes.
    The store underneath is ``kb.store``.
    """

    TOOLS = ("query",)

    def __init__(self, backend: str, collection: str, *, description: str, embedder=None,
                 url: "str | None" = None, api_key: "str | None" = None, **settings) -> None:
        from . import VectorDB, VectorQueryTool
        if not isinstance(description, str) or not description.strip():
            raise ValueError("Vectors needs a description of what the database holds — an agent's "
                             "search tool says what it searches by it")
        self.description = description.strip()
        self.store = VectorDB(backend, collection, embedder=embedder, url=url, api_key=api_key, **settings)
        self.query = VectorQueryTool(self.store)
        self.query.description = (f"Search {self.description.rstrip('.')} — the passages most relevant "
                                  "to a question, by meaning rather than by exact words.")

    def create(self, dimension: int, metric: str = "cosine") -> None:
        """Create the collection, *dimension* long, if it does not exist."""
        return self.store.create(dimension, metric)

    def upsert(self, records: list, batch_size: "int | None" = None) -> int:
        """Insert or replace records — each ``{"id", "text", …}``; returns how many."""
        return self.store.upsert(records, batch_size)

    def fetch(self, ids: list) -> list:
        """The records with these ids."""
        return self.store.fetch(ids)

    def delete(self, ids: "list | None" = None, filter: "dict | None" = None) -> None:
        """Remove records by id or by a metadata filter."""
        return self.store.delete(ids, filter)

    def count(self) -> int:
        """How many records the collection holds."""
        return self.store.count()
