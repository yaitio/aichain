"""
tools.vectordb.providers._chroma
=================================

ChromaBackend — VectorBackend adapter for ChromaDB via its REST API v1.

Uses pure urllib3, no ``chromadb`` SDK required.

Environment variables
---------------------
``CHROMA_URL``      — base URL of the Chroma server  (default: http://localhost:8000)
``CHROMA_API_KEY``  — API key for Chroma Cloud        (optional for self-hosted)

Chroma REST API v1 reference
-----------------------------
POST   /api/v1/collections                     — create collection
GET    /api/v1/collections/{name}              — describe (returns id, name, metadata)
POST   /api/v1/collections/{id}/upsert         — upsert documents
POST   /api/v1/collections/{id}/query          — similarity query
POST   /api/v1/collections/{id}/get            — fetch by IDs or filter
POST   /api/v1/collections/{id}/delete         — delete by IDs or filter
GET    /api/v1/collections/{id}/count          — record count

Distance normalisation
-----------------------
Chroma returns raw distances (lower = more similar).  For cosine distance
the range is [0, 2].  We normalise to a score in [0, 1]:

    score = 1.0 − distance / 2.0

For l2 (Euclidean) the range is unbounded so we use:

    score = 1.0 / (1.0 + distance)

Inner-product (ip) results are returned as negated similarity; we negate back.
"""

from __future__ import annotations

import os

from .._base import VectorBackend, VectorRecord


# ── Metric mapping ─────────────────────────────────────────────────────────────

# aichain metric name → Chroma hnsw:space value
_METRIC_MAP: dict[str, str] = {
    "cosine":    "cosine",
    "euclidean": "l2",
    "dot":       "ip",
}


def _normalise_score(distance: float, space: str) -> float:
    """Convert a Chroma distance to a similarity score in [0, 1]."""
    if space == "cosine":
        return max(0.0, 1.0 - distance / 2.0)
    if space == "ip":
        return float(-distance)          # ip stores negated dot-product
    # l2 / unknown
    return 1.0 / (1.0 + distance)


# ── ChromaBackend ──────────────────────────────────────────────────────────────

class ChromaBackend(VectorBackend):
    """
    VectorBackend for ChromaDB (REST API v1).

    Parameters
    ----------
    url : str | None
        Base URL of the Chroma HTTP server.
        Defaults to the ``CHROMA_URL`` env var or ``http://localhost:8000``.
    api_key : str | None
        API key for Chroma Cloud.  Omit for local/self-hosted deployments.
        Defaults to ``CHROMA_API_KEY`` env var.

    Examples
    --------
    Local server::

        from tools.vectordb import VectorDB
        from tools.embedding import Embedding

        store = VectorDB(
            "chroma",
            collection = "my_docs",
            embedder   = Embedding("openai/text-embedding-3-small"),
            url        = "http://localhost:8000",
        )

    Chroma Cloud::

        store = VectorDB(
            "chroma",
            collection = "my_docs",
            embedder   = Embedding("openai/text-embedding-3-small"),
            url        = "https://api.trychroma.com",
            api_key    = "ch-...",
        )
    """

    provider    = "chroma"
    _DEFAULT_URL = "http://localhost:8000"

    def __init__(
        self,
        *,
        url:     str | None = None,
        api_key: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            url     = url     or os.environ.get("CHROMA_URL", self._DEFAULT_URL),
            api_key = api_key or os.environ.get("CHROMA_API_KEY"),
        )
        # name → (id, hnsw_space) cache — avoids repeated GET per operation
        self._collection_cache: dict[str, tuple[str, str]] = {}

    # ── Collection ID resolution ───────────────────────────────────────────────

    def _resolve(self, name: str) -> tuple[str, str]:
        """Return (collection_id, hnsw_space) for *name*, cached after first call."""
        if name in self._collection_cache:
            return self._collection_cache[name]
        data  = self._get(f"/api/v1/collections/{name}")
        cid   = data["id"]
        space = (data.get("metadata") or {}).get("hnsw:space", "cosine")
        self._collection_cache[name] = (cid, space)
        return cid, space

    # ── Abstract method implementations ───────────────────────────────────────

    def create_collection(
        self,
        collection: str,
        dimension:  int,
        metric:     str = "cosine",
    ) -> None:
        space = _METRIC_MAP.get(metric, "cosine")
        body  = {
            "name":     collection,
            "metadata": {"hnsw:space": space},
        }
        data = self._post("/api/v1/collections", body)
        cid  = data["id"]
        self._collection_cache[collection] = (cid, space)

    def query(
        self,
        collection: str,
        vector:     list[float],
        n:          int = 5,
        filter:     dict | None = None,
    ) -> list[VectorRecord]:
        cid, space = self._resolve(collection)
        body: dict = {
            "query_embeddings": [vector],
            "n_results":        n,
            "include":          ["documents", "metadatas", "distances"],
        }
        if filter:
            body["where"] = filter
        data = self._post(f"/api/v1/collections/{cid}/query", body)

        ids       = (data.get("ids")       or [[]])[0]
        docs      = (data.get("documents") or [[]])[0]
        metas     = (data.get("metadatas") or [[]])[0]
        distances = (data.get("distances") or [[]])[0]

        records: list[VectorRecord] = []
        for i, rid in enumerate(ids):
            dist  = distances[i] if i < len(distances) else None
            score = _normalise_score(dist, space) if dist is not None else None
            records.append(VectorRecord(
                id       = rid,
                text     = docs[i]  if i < len(docs)  else "",
                score    = score,
                metadata = metas[i] if i < len(metas) else {},
            ))
        return records

    def upsert(
        self,
        collection: str,
        records:    list[VectorRecord],
        vectors:    list[list[float]],
    ) -> None:
        cid, _ = self._resolve(collection)
        body   = {
            "ids":        [r.id       for r in records],
            "embeddings": vectors,
            "documents":  [r.text     for r in records],
            "metadatas":  [r.metadata for r in records],
        }
        self._post(f"/api/v1/collections/{cid}/upsert", body)

    def fetch(
        self,
        collection: str,
        ids:        list[str],
    ) -> list[VectorRecord]:
        cid, _ = self._resolve(collection)
        body   = {"ids": ids, "include": ["documents", "metadatas"]}
        data   = self._post(f"/api/v1/collections/{cid}/get", body)

        ret_ids = data.get("ids")       or []
        docs    = data.get("documents") or []
        metas   = data.get("metadatas") or []

        return [
            VectorRecord(
                id       = ret_ids[i],
                text     = docs[i]  if i < len(docs)  else "",
                metadata = metas[i] if i < len(metas) else {},
            )
            for i in range(len(ret_ids))
        ]

    def delete(
        self,
        collection: str,
        ids:        list[str] | None = None,
        filter:     dict | None      = None,
    ) -> None:
        cid, _ = self._resolve(collection)
        body: dict = {}
        if ids:
            body["ids"] = ids
        if filter:
            body["where"] = filter
        self._post(f"/api/v1/collections/{cid}/delete", body)

    def count(self, collection: str) -> int:
        cid, _ = self._resolve(collection)
        data   = self._get(f"/api/v1/collections/{cid}/count")
        # Chroma returns a bare integer (not wrapped in an object)
        if isinstance(data, (int, float)):
            return int(data)
        return int(data.get("count", 0)) if isinstance(data, dict) else 0
