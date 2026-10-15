"""``QdrantBackend``: Qdrant over its REST API, with urllib3 and no SDK. A point
is a ``VectorRecord``, its text kept in ``payload["text"]``, filters passed
through as Qdrant's own JSON. ``QDRANT_URL`` and ``QDRANT_API_KEY`` (sent as
``api-key``, for Qdrant Cloud). See docs/utilities/index.md.
"""

from __future__ import annotations

import os
import uuid

from ....clients._errors import APIError
from .._base import ID_KEY as _ORIGINAL_ID_KEY, VectorBackend, VectorRecord, _one_of, check_metric, record_from_payload

def _point_id(rec_id: str | int) -> int | str:
    """A record id as a Qdrant point id, which must be an unsigned integer or a
    UUID: a string id becomes a UUIDv5, the original kept in the payload under
    ``_ORIGINAL_ID_KEY`` and restored on every read."""
    s = str(rec_id)
    # Only an integer's own spelling: "007" and "7" must not be one point, and
    # "²" passes isdigit() but raises in int().
    if s.isascii() and s.isdigit() and s == str(int(s)):
        return int(s)
    try:
        return str(uuid.UUID(s))
    except ValueError:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"aichain:{s}"))


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

        from yait_aichain.utilities.vectors import VectorDB
        from yait_aichain.tools import embed

        store = VectorDB(
            "qdrant",
            collection = "knowledge_base",
            embedder   = embed("openai", model="text-embedding-3-small"),
            url        = "http://localhost:6333",
        )

    Qdrant Cloud::

        store = VectorDB(
            "qdrant",
            collection = "knowledge_base",
            embedder   = embed("openai", model="text-embedding-3-small"),
            url        = "https://xyz.us-east4-0.gcp.cloud.qdrant.io",
            api_key    = "qdrant-...",
        )
    """

    provider     = "qdrant"
    FILTER       = ('Qdrant filter: {"must": [{"key": "source", "match": {"value": "arxiv"}}]}, '
                    '{"must": [{"key": "year", "range": {"gte": 2020}}]}')
    _DEFAULT_URL = "http://localhost:6333"

    def __init__(
        self,
        *,
        url:     str | None = None,
        api_key: str | None = None,
        metric:  str        = "cosine",
    ) -> None:
        super().__init__(
            url     = url     or os.environ.get("QDRANT_URL", self._DEFAULT_URL),
            api_key = api_key or os.environ.get("QDRANT_API_KEY"),
        )
        self._metric = check_metric(metric, _METRIC_MAP, "Qdrant")

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
        distance = _METRIC_MAP[check_metric(metric, _METRIC_MAP, "Qdrant")]
        body     = {
            "vectors": {
                "size":     dimension,
                "distance": distance,
            }
        }
        try:
            self._put(f"/collections/{collection}", body)
        except APIError as exc:
            # 409 on current Qdrant; 400 "already exists" on older releases.
            if not (exc.status == 409 or (exc.status == 400 and "exist" in str(exc).lower())):
                raise
            # It exists already, which is the goal — as on every other
            # backend — and its own metric is the one its scores are on.
            info = self._get(f"/collections/{collection}").get("result") or {}
            vectors = ((info.get("config") or {}).get("params") or {}).get("vectors") or {}
            if vectors.get("size") not in (None, dimension):
                raise ValueError(f"Qdrant collection {collection!r} exists with dimension "
                                 f"{vectors.get('size')}, not {dimension}") from None
            back = {v: k for k, v in _METRIC_MAP.items()}
            metric = back.get(vectors.get("distance"), metric)
        self._metric = metric

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

        return [record_from_payload(p["id"], p.get("payload"), p.get("score"), self._metric)
                for p in data.get("result", [])]

    #: The metric this store was created with. Declared, not discovered.
    #:
    #: Asking the collection adds a network round trip with retries in front
    #: of the first query, and when the host is unreachable that is a minute
    #: of backoff before a search that would otherwise have failed fast. A hidden request is a bad trade
    #: for a value the caller already knows — and `raw_score` keeps the
    #: original either way, so a wrong declaration is visible rather than
    #: baked into a number.
    _metric = "cosine"

    def upsert(
        self,
        collection: str,
        records:    list[VectorRecord],
        vectors:    list[list[float]],
    ) -> None:
        points = []
        for rec, vec in zip(records, vectors):
            payload = {
                **rec.metadata,
                "text":           rec.text,
                _ORIGINAL_ID_KEY: str(rec.id),
            }
            points.append({
                "id":      _point_id(rec.id),
                "vector":  vec,
                "payload": payload,
            })
        # wait=true: Qdrant acknowledges a write before it is searchable, and a
        # query straight after an upsert found nothing.
        self._put(f"/collections/{collection}/points?wait=true", {"points": points})

    def fetch(
        self,
        collection: str,
        ids:        list[str],
    ) -> list[VectorRecord]:
        # Qdrant: POST /collections/{name}/points with body {"ids": [...]}
        body = {"ids": [_point_id(i) for i in ids], "with_payload": True}
        data = self._post(f"/collections/{collection}/points", body)

        return [record_from_payload(p["id"], p.get("payload")) for p in data.get("result", [])]

    def delete(
        self,
        collection: str,
        ids:        list[str] | None = None,
        filter:     dict | None      = None,
    ) -> None:
        # The VectorStore layer guarantees at least one non-empty argument
        # (an empty list/dict would otherwise mean "delete everything").
        _one_of(ids, filter)
        body: dict = {"points": [_point_id(i) for i in ids]} if ids else {"filter": filter}
        self._post(f"/collections/{collection}/points/delete?wait=true", body)

    def count(self, collection: str) -> int:
        data = self._post(
            f"/collections/{collection}/points/count",
            {"exact": True},
        )
        return int((data.get("result") or {}).get("count", 0))
