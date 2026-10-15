"""``ChromaBackend``: ChromaDB over its REST API v2 (the tenant/database-scoped
layout; v1 is gone from Chroma 0.6 and Chroma Cloud), with urllib3 and no SDK.
Configured by ``CHROMA_URL``, ``CHROMA_API_KEY``, ``CHROMA_TENANT`` and
``CHROMA_DATABASE``. See docs/utilities/index.md.
"""

from __future__ import annotations

import os

from ....clients._errors import APIError
from .._base import VectorBackend, VectorRecord, _one_of, normalise_score


def _columns(ids: list, *columns: list):
    """Chroma answers in parallel columns; one row per id, a short or absent
    column read as empty rather than shifting the rest."""
    return zip(ids, *(list(c) + [None] * (len(ids) - len(c)) for c in columns))


class ChromaBackend(VectorBackend):
    """
    VectorBackend for ChromaDB (REST API v2).

    Parameters
    ----------
    url : str | None
        Base URL of the Chroma HTTP server.
        Defaults to the ``CHROMA_URL`` env var or ``http://localhost:8000``.
    api_key : str | None
        API key for Chroma Cloud.  Omit for local/self-hosted deployments.
        Defaults to ``CHROMA_API_KEY`` env var.
    tenant : str | None
        Tenant name.  Defaults to ``CHROMA_TENANT`` env var or
        ``default_tenant``.
    database : str | None
        Database name.  Defaults to ``CHROMA_DATABASE`` env var or
        ``default_database``.

    Examples
    --------
    Chroma Cloud::

        store = VectorDB("chroma", collection="my_docs", embedder=embed("openai"),
                         url="https://api.trychroma.com", api_key="ch-...")
    """

    provider     = "chroma"
    FILTER       = ('Chroma "where": {"source": "arxiv"}, {"year": {"$gte": 2020}}, '
                    '{"$and": [{...}, {...}]}')
    _METRICS     = {"cosine": "cosine", "euclidean": "l2", "dot": "ip"}
    _ENV_URL, _ENV_KEY, _DEFAULT_URL = "CHROMA_URL", "CHROMA_API_KEY", "http://localhost:8000"

    def __init__(
        self,
        *,
        url:      str | None = None,
        api_key:  str | None = None,
        tenant:   str | None = None,
        database: str | None = None,
    ) -> None:
        super().__init__(url=url, api_key=api_key)
        tenant   = tenant   or os.environ.get("CHROMA_TENANT",   "default_tenant")
        database = database or os.environ.get("CHROMA_DATABASE", "default_database")
        self._prefix = f"/api/v2/tenants/{tenant}/databases/{database}"
        # name → (id, hnsw_space) cache — avoids repeated GET per operation
        self._collection_cache: dict[str, tuple[str, str]] = {}

    # ── Collection ID resolution ───────────────────────────────────────────────

    def _resolve(self, name: str) -> tuple[str, str]:
        """Return (collection_id, hnsw_space) for *name*, cached after first call;
        Chroma's own default space is l2, for a collection created without one."""
        if name not in self._collection_cache:
            data = self._get(f"{self._prefix}/collections/{name}")
            self._collection_cache[name] = (data["id"], (data.get("metadata") or {}).get("hnsw:space") or "l2")
        return self._collection_cache[name]

    def _on(self, name: str, verb: str, body: dict | None = None):
        """One request to collection *name*'s *verb*, and its space. A cached id
        that answers 404 — the collection was recreated elsewhere — is looked up once more."""
        for attempt in (0, 1):
            cid, space = self._resolve(name)
            try:
                return self._call("GET" if body is None else "POST",
                                  f"{self._prefix}/collections/{cid}/{verb}", body), space
            except APIError as exc:
                if exc.status != 404 or attempt:
                    raise
                self._collection_cache.pop(name, None)

    @staticmethod
    def _record(rid, doc, meta, distance=None, space="cosine") -> VectorRecord:
        """One row of a Chroma answer. A distance is ``1 − cos``, ``1 − dot`` or a
        squared euclidean, turned back into what the other backends return."""
        if distance is None:
            score = None
        elif space == "l2":
            score = normalise_score(distance, "sqeuclidean")
        else:
            score = normalise_score(1.0 - distance, "cosine" if space == "cosine" else "dot")
        return VectorRecord(id=rid, text=doc or "", metadata=meta or {}, score=score, raw_score=distance)

    # ── Abstract method implementations ───────────────────────────────────────

    def create_collection(
        self,
        collection: str,
        dimension:  int,
        metric:     str = "cosine",
    ) -> None:
        # Idempotent: get_or_create answers with a collection that exists, and
        # its own space and dimension, not the ones asked for, are what count.
        body = {"name": collection, "metadata": {"hnsw:space": self._native(metric)}, "get_or_create": True}
        data = self._post(f"{self._prefix}/collections", body)
        metric = self._adopt(f"collection {collection!r}", dimension, metric, data.get("dimension"),
                             (data.get("metadata") or {}).get("hnsw:space"))
        self._collection_cache[collection] = (data["id"], self._METRICS[metric])

    def query(
        self,
        collection: str,
        vector:     list[float],
        n:          int = 5,
        filter:     dict | None = None,
    ) -> list[VectorRecord]:
        body: dict = {"query_embeddings": [vector], "n_results": n,
                      "include": ["documents", "metadatas", "distances"]}
        if filter:
            body["where"] = filter
        data, space = self._on(collection, "query", body)
        first = [(data.get(k) or [[]])[0] for k in ("ids", "documents", "metadatas", "distances")]
        return [self._record(*row, space=space) for row in _columns(*first)]

    def upsert(
        self,
        collection: str,
        records:    list[VectorRecord],
        vectors:    list[list[float]],
    ) -> None:
        self._on(collection, "upsert", {"ids": [r.id for r in records], "embeddings": vectors,
                                        "documents": [r.text for r in records],
                                        "metadatas": [r.metadata for r in records]})

    def fetch(
        self,
        collection: str,
        ids:        list[str],
    ) -> list[VectorRecord]:
        data, _ = self._on(collection, "get", {"ids": ids, "include": ["documents", "metadatas"]})
        return [self._record(*row) for row in _columns(*(data.get(k) or [] for k in
                                                         ("ids", "documents", "metadatas")))]

    def delete(
        self,
        collection: str,
        ids:        list[str] | None = None,
        filter:     dict | None      = None,
    ) -> None:
        _one_of(ids, filter)
        self._on(collection, "delete", {"ids": ids} if ids else {"where": filter})

    def count(self, collection: str) -> int:
        data, _ = self._on(collection, "count")
        # Chroma returns a bare integer (not wrapped in an object)
        if isinstance(data, (int, float)):
            return int(data)
        return int(data.get("count", 0)) if isinstance(data, dict) else 0
