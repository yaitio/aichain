"""
tools.vectordb.providers._qdrant
==================================

QdrantBackend — VectorBackend adapter for Qdrant via its REST API.

Uses pure urllib3, no ``qdrant-client`` SDK required.

Qdrant concepts vs aichain mapping
------------------------------------
Qdrant concept     aichain concept
──────────────     ────────────────────────────────────────
Collection         collection  (1-to-1 mapping)
Point              VectorRecord
Payload            metadata + text (stored as payload["text"])
Filter             passed through as-is (Qdrant native filter JSON)

Text storage
  Qdrant stores only vectors + payload dicts.  We persist the original
  text as payload["text"] so fetch() and query() can reconstruct
  VectorRecord.text without an extra lookup.

Environment variables
---------------------
``QDRANT_URL``     — base URL of the Qdrant server  (default: http://localhost:6333)
``QDRANT_API_KEY`` — API key for Qdrant Cloud        (optional for self-hosted)

Qdrant REST endpoints used
---------------------------
PUT    /collections/{name}                        — create collection
GET    /collections/{name}                        — describe (not used directly)
PUT    /collections/{name}/points                 — upsert points
POST   /collections/{name}/points/search          — similarity search
POST   /collections/{name}/points                 — retrieve points by IDs
POST   /collections/{name}/points/delete          — delete by IDs or filter
POST   /collections/{name}/points/count           — exact record count

Auth
  Qdrant Cloud uses the ``api-key`` header (not ``Authorization: Bearer``).
  Self-hosted Qdrant typically requires no auth header.
"""

from __future__ import annotations

import os

from .._base import VectorBackend, VectorRecord


# ── Metric mapping ─────────────────────────────────────────────────────────────

# aichain metric name → Qdrant Distance enum string
_METRIC_MAP: dict[str, str] = {
    "cosine":    "Cosine",
    "euclidean": "Euclid",
    "dot":       "Dot",
    "manhattan": "Manhattan",
}


# ── QdrantBackend ──────────────────────────────────────────────────────────────

class QdrantBackend(VectorBackend):
    """
    VectorBackend for Qdrant (REST API).

    Parameters
    ----------
    url : str | None
        Base URL of the Qdrant server.
        Defaults to the ``QDRANT_URL`` env var or ``http://localhost:6333``.
    api_key : str | None
        API key for Qdrant Cloud.  Omit for local/self-hosted deployments.
        Defaults to ``QDRANT_API_KEY`` env var.

    Examples
    --------
    Local server::

        from tools.vectordb import VectorDB
        from tools.embedding import Embedding

        store = VectorDB(
            "qdrant",
            collection = "knowledge_base",
            embedder   = Embedding("openai/text-embedding-3-small"),
            url        = "http://localhost:6333",
        )

    Qdrant Cloud::

        store = VectorDB(
            "qdrant",
            collection = "knowledge_base",
            embedder   = Embedding("openai/text-embedding-3-small"),
            url        = "https://xyz.us-east4-0.gcp.cloud.qdrant.io",
            api_key    = "qdrant-...",
        )
    """

    provider     = "qdrant"
    _DEFAULT_URL = "http://localhost:6333"

    def __init__(
        self,
        *,
        url:     str | None = None,
        api_key: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            url     = url     or os.environ.get("QDRANT_URL", self._DEFAULT_URL),
            api_key = api_key or os.environ.get("QDRANT_API_KEY"),
        )

    # ── Auth header override ───────────────────────────────────────────────────

    def _headers(self) -> dict:
        # Qdrant uses "api-key" header, not "Authorization: Bearer"
        h: dict = {
            "Content-Type": "application/json",
            "Accept":       "application/json",
        }
        if self._api_key:
            h["api-key"] = self._api_key
        return h

    # ── Abstract method implementations ───────────────────────────────────────

    def create_collection(
        self,
        collection: str,
        dimension:  int,
        metric:     str = "cosine",
    ) -> None:
        distance = _METRIC_MAP.get(metric, "Cosine")
        body     = {
            "vectors": {
                "size":     dimension,
                "distance": distance,
            }
        }
        self._put(f"/collections/{collection}", body)

    def query(
        self,
        collection: str,
        vector:     list[float],
        n:          int = 5,
        filter:     dict | None = None,
    ) -> list[VectorRecord]:
        body: dict = {
            "vector":       vector,
            "limit":        n,
            "with_payload": True,
        }
        if filter:
            body["filter"] = filter
        data = self._post(f"/collections/{collection}/points/search", body)

        records: list[VectorRecord] = []
        for point in data.get("result", []):
            payload = dict(point.get("payload") or {})
            text    = payload.pop("text", payload.pop("document", ""))
            records.append(VectorRecord(
                id       = str(point["id"]),
                text     = text,
                score    = point.get("score"),
                metadata = payload,
            ))
        return records

    def upsert(
        self,
        collection: str,
        records:    list[VectorRecord],
        vectors:    list[list[float]],
    ) -> None:
        points = []
        for rec, vec in zip(records, vectors):
            payload = {**rec.metadata, "text": rec.text}
            points.append({
                "id":      rec.id,
                "vector":  vec,
                "payload": payload,
            })
        self._put(f"/collections/{collection}/points", {"points": points})

    def fetch(
        self,
        collection: str,
        ids:        list[str],
    ) -> list[VectorRecord]:
        # Qdrant: POST /collections/{name}/points with body {"ids": [...]}
        body = {"ids": ids, "with_payload": True}
        data = self._post(f"/collections/{collection}/points", body)

        records: list[VectorRecord] = []
        for point in data.get("result", []):
            payload = dict(point.get("payload") or {})
            text    = payload.pop("text", "")
            records.append(VectorRecord(
                id       = str(point["id"]),
                text     = text,
                metadata = payload,
            ))
        return records

    def delete(
        self,
        collection: str,
        ids:        list[str] | None = None,
        filter:     dict | None      = None,
    ) -> None:
        body: dict = {}
        if ids:
            body["points"] = ids
        elif filter:
            body["filter"] = filter
        else:
            # delete all: empty filter matches every point
            body["filter"] = {}
        self._post(f"/collections/{collection}/points/delete", body)

    def count(self, collection: str) -> int:
        data = self._post(
            f"/collections/{collection}/points/count",
            {"exact": True},
        )
        return int((data.get("result") or {}).get("count", 0))
