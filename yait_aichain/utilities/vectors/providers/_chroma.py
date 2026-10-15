"""``ChromaBackend``: ChromaDB over its REST API v2 (the tenant/database-scoped
layout; v1 is gone from Chroma 0.6 and Chroma Cloud), with urllib3 and no SDK.
Configured by ``CHROMA_URL``, ``CHROMA_API_KEY``, ``CHROMA_TENANT`` and
``CHROMA_DATABASE``. See docs/utilities/index.md.
"""

from __future__ import annotations

import os

from ....clients._errors import APIError
from .._base import VectorBackend, VectorRecord, _one_of, check_metric, normalise_score


# ── Metric mapping ─────────────────────────────────────────────────────────────

# aichain metric name → Chroma hnsw:space value
_METRIC_MAP: dict[str, str] = {
    "cosine":    "cosine",
    "euclidean": "l2",
    "dot":       "ip",
}


def _columns(ids: list, *columns: list):
    """Chroma answers in parallel columns; one row per id, a short or absent
    column read as empty rather than shifting the rest."""
    return zip(ids, *(list(c) + [None] * (len(ids) - len(c)) for c in columns))


def _normalise_score(distance: float, space: str) -> float:
    """A Chroma distance (``1 − cos``, ``1 − dot``, or squared euclidean for l2)
    turned back into what the other backends return, then scored by the
    shared rule."""
    if space == "cosine":
        return normalise_score(1.0 - distance, "cosine")
    if space == "ip":
        return normalise_score(1.0 - distance, "dot")
    return normalise_score(distance, "sqeuclidean")


# ── ChromaBackend ──────────────────────────────────────────────────────────────

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
    Local server::

        from yait_aichain.utilities.vectors import VectorDB
        from yait_aichain.tools import embed

        store = VectorDB(
            "chroma",
            collection = "my_docs",
            embedder   = embed("openai", model="text-embedding-3-small"),
            url        = "http://localhost:8000",
        )

    Chroma Cloud::

        store = VectorDB(
            "chroma",
            collection = "my_docs",
            embedder   = embed("openai", model="text-embedding-3-small"),
            url        = "https://api.trychroma.com",
            api_key    = "ch-...",
        )
    """

    provider    = "chroma"
    FILTER      = ('Chroma "where": {"source": "arxiv"}, {"year": {"$gte": 2020}}, '
                   '{"$and": [{...}, {...}]}')
    _DEFAULT_URL = "http://localhost:8000"

    def __init__(
        self,
        *,
        url:      str | None = None,
        api_key:  str | None = None,
        tenant:   str | None = None,
        database: str | None = None,
    ) -> None:
        super().__init__(
            url     = url     or os.environ.get("CHROMA_URL", self._DEFAULT_URL),
            api_key = api_key or os.environ.get("CHROMA_API_KEY"),
        )
        tenant   = tenant   or os.environ.get("CHROMA_TENANT",   "default_tenant")
        database = database or os.environ.get("CHROMA_DATABASE", "default_database")
        self._prefix = f"/api/v2/tenants/{tenant}/databases/{database}"
        # name → (id, hnsw_space) cache — avoids repeated GET per operation
        self._collection_cache: dict[str, tuple[str, str]] = {}

    # ── Collection ID resolution ───────────────────────────────────────────────

    def _resolve(self, name: str) -> tuple[str, str]:
        """Return (collection_id, hnsw_space) for *name*, cached after first call."""
        if name in self._collection_cache:
            return self._collection_cache[name]
        data  = self._get(f"{self._prefix}/collections/{name}")
        cid   = data["id"]
        # Chroma's own default is l2, for a collection created elsewhere
        # without a space.
        space = (data.get("metadata") or {}).get("hnsw:space") or "l2"
        self._collection_cache[name] = (cid, space)
        return cid, space

    def _on(self, name: str, verb: str, body: dict | None = None):
        """One request to collection *name*'s *verb*, and its space. A cached
        id that answers 404 — the collection was recreated under the same
        name elsewhere — is dropped and looked up once more."""
        for attempt in (0, 1):
            cid, space = self._resolve(name)
            try:
                return self._call("GET" if body is None else "POST",
                                  f"{self._prefix}/collections/{cid}/{verb}", body), space
            except APIError as exc:
                if exc.status != 404 or attempt:
                    raise
                self._collection_cache.pop(name, None)

    # ── Abstract method implementations ───────────────────────────────────────

    def create_collection(
        self,
        collection: str,
        dimension:  int,
        metric:     str = "cosine",
    ) -> None:
        space = _METRIC_MAP[check_metric(metric, _METRIC_MAP, "Chroma")]
        body  = {
            "name":          collection,
            "metadata":      {"hnsw:space": space},
            # Idempotent create: an existing collection is returned instead
            # of an error (matches Pinecone-backend semantics).
            "get_or_create": True,
        }
        data = self._post(f"{self._prefix}/collections", body)
        cid  = data["id"]
        self._collection_cache[collection] = (cid, space)

    def query(
        self,
        collection: str,
        vector:     list[float],
        n:          int = 5,
        filter:     dict | None = None,
    ) -> list[VectorRecord]:
        body: dict = {
            "query_embeddings": [vector],
            "n_results":        n,
            "include":          ["documents", "metadatas", "distances"],
        }
        if filter:
            body["where"] = filter
        data, space = self._on(collection, "query", body)

        ids       = (data.get("ids")       or [[]])[0]
        docs      = (data.get("documents") or [[]])[0]
        metas     = (data.get("metadatas") or [[]])[0]
        distances = (data.get("distances") or [[]])[0]

        return [VectorRecord(id=rid, text=doc or "", metadata=meta or {},
                             score=_normalise_score(dist, space) if dist is not None else None,
                             raw_score=dist)
                for rid, doc, meta, dist in _columns(ids, docs, metas, distances)]

    def upsert(
        self,
        collection: str,
        records:    list[VectorRecord],
        vectors:    list[list[float]],
    ) -> None:
        body   = {
            "ids":        [r.id       for r in records],
            "embeddings": vectors,
            "documents":  [r.text     for r in records],
            "metadatas":  [r.metadata for r in records],
        }
        self._on(collection, "upsert", body)

    def fetch(
        self,
        collection: str,
        ids:        list[str],
    ) -> list[VectorRecord]:
        data, _ = self._on(collection, "get", {"ids": ids, "include": ["documents", "metadatas"]})

        ret_ids = data.get("ids")       or []
        docs    = data.get("documents") or []
        metas   = data.get("metadatas") or []

        return [VectorRecord(id=rid, text=doc or "", metadata=meta or {})
                for rid, doc, meta in _columns(ret_ids, docs, metas)]

    def delete(
        self,
        collection: str,
        ids:        list[str] | None = None,
        filter:     dict | None      = None,
    ) -> None:
        # An empty ids+filter would tell Chroma to delete the whole collection.
        # Guard at the backend too (defence in depth), mirroring Qdrant.
        _one_of(ids, filter)
        body: dict = {"ids": ids} if ids else {"where": filter}
        self._on(collection, "delete", body)

    def count(self, collection: str) -> int:
        data, _ = self._on(collection, "count")
        # Chroma returns a bare integer (not wrapped in an object)
        if isinstance(data, (int, float)):
            return int(data)
        return int(data.get("count", 0)) if isinstance(data, dict) else 0
