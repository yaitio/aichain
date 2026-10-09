"""utilities.vectors._utility — ``Vectors``, a vector database as a utility."""

from __future__ import annotations

from .._base import Utility


class Vectors(Utility):
    """
    A vector database: ``create``, ``upsert``, ``fetch``, ``delete`` and
    ``count`` for code; ``query`` a tool an agent can take.

    ::

        kb = Vectors("qdrant", collection="docs", embedder=embed("openai", model="text-embedding-3-small"))
        kb.create(dimension=1536)
        kb.upsert(chunk(max_chars=800).run(text))
        kb.query.run("how does caching work?", {"n": 5})
        Agent(model, tools=[kb.query])            # the agent searches, and can do nothing else

    ``backend`` is ``chroma``, ``pinecone`` or ``qdrant``; ``collection``,
    ``embedder``, ``url``, ``api_key`` and the backend's own settings
    (Pinecone's ``index``, ``cloud``, ``region``) are what ``VectorDB`` takes.
    The store underneath is ``kb.store``.
    """

    TOOLS = ("query",)

    def __init__(self, backend: str, collection: str, *, embedder=None, url: "str | None" = None,
                 api_key: "str | None" = None, **settings) -> None:
        from . import VectorDB, VectorQueryTool
        self.store = VectorDB(backend, collection, embedder=embedder, url=url, api_key=api_key, **settings)
        self.query = VectorQueryTool(self.store)

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
