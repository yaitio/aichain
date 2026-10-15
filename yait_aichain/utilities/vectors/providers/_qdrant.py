"""``QdrantBackend``: Qdrant over its REST API, with urllib3 and no SDK. A point
is a ``VectorRecord``, its text kept in ``payload["text"]``, filters passed
through as Qdrant's own JSON. ``QDRANT_URL`` and ``QDRANT_API_KEY`` (sent as
``api-key``, for Qdrant Cloud). See docs/utilities/index.md.
"""

from __future__ import annotations

import uuid

from ....clients._errors import APIError
from .._base import ID_KEY, VectorBackend, VectorRecord, _one_of, check_metric, record_from_payload


def _point_id(rec_id: str | int) -> int | str:
    """A record id as a Qdrant point id, which must be an unsigned integer or a
    UUID: a string id becomes a UUIDv5, the original kept in the payload under
    ``ID_KEY`` and restored on every read."""
    s = str(rec_id)
    # Only an integer's own spelling: "007" and "7" must not be one point, and
    # "²" passes isdigit() but raises in int().
    if s.isascii() and s.isdigit() and s == str(int(s)):
        return int(s)
    try:
        return str(uuid.UUID(s))
    except ValueError:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"aichain:{s}"))


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
    _METRICS     = {"cosine": "Cosine", "euclidean": "Euclid", "dot": "Dot", "manhattan": "Manhattan"}
    _AUTH        = "api-key"
    _ENV_URL, _ENV_KEY, _DEFAULT_URL = "QDRANT_URL", "QDRANT_API_KEY", "http://localhost:6333"

    def __init__(
        self,
        *,
        url:     str | None = None,
        api_key: str | None = None,
        metric:  str        = "cosine",
    ) -> None:
        super().__init__(url=url, api_key=api_key)
        self._metric = check_metric(metric, self._METRICS, "Qdrant")

    def _record(self, point: dict) -> VectorRecord:
        return record_from_payload(point["id"], point.get("payload"), point.get("score"), self._metric)

    # ── Abstract method implementations ───────────────────────────────────────

    def create_collection(
        self,
        collection: str,
        dimension:  int,
        metric:     str = "cosine",
    ) -> None:
        try:
            self._put(f"/collections/{collection}",
                      {"vectors": {"size": dimension, "distance": self._native(metric)}})
        except APIError as exc:
            # 409 on current Qdrant; 400 "already exists" on older releases.
            if not (exc.status == 409 or (exc.status == 400 and "exist" in str(exc).lower())):
                raise
            info = self._get(f"/collections/{collection}").get("result") or {}
            vectors = ((info.get("config") or {}).get("params") or {}).get("vectors") or {}
            metric = self._adopt(f"collection {collection!r}", dimension, metric,
                                 vectors.get("size"), vectors.get("distance"))
        self._metric = metric

    def query(
        self,
        collection: str,
        vector:     list[float],
        n:          int = 5,
        filter:     dict | None = None,
    ) -> list[VectorRecord]:
        body: dict = {"vector": vector, "limit": n, "with_payload": True}
        if filter:
            body["filter"] = filter
        data = self._post(f"/collections/{collection}/points/search", body)
        return [self._record(p) for p in data.get("result", [])]

    def upsert(
        self,
        collection: str,
        records:    list[VectorRecord],
        vectors:    list[list[float]],
    ) -> None:
        points = [{"id": _point_id(rec.id), "vector": vec,
                   "payload": {**rec.metadata, "text": rec.text, ID_KEY: str(rec.id)}}
                  for rec, vec in zip(records, vectors)]
        # wait=true: Qdrant acknowledges a write before it is searchable, and a
        # query straight after an upsert found nothing.
        self._put(f"/collections/{collection}/points?wait=true", {"points": points})

    def fetch(
        self,
        collection: str,
        ids:        list[str],
    ) -> list[VectorRecord]:
        data = self._post(f"/collections/{collection}/points",
                          {"ids": [_point_id(i) for i in ids], "with_payload": True})
        return [self._record(p) for p in data.get("result", [])]

    def delete(
        self,
        collection: str,
        ids:        list[str] | None = None,
        filter:     dict | None      = None,
    ) -> None:
        _one_of(ids, filter)
        body: dict = {"points": [_point_id(i) for i in ids]} if ids else {"filter": filter}
        self._post(f"/collections/{collection}/points/delete?wait=true", body)

    def count(self, collection: str) -> int:
        data = self._post(f"/collections/{collection}/points/count", {"exact": True})
        return int((data.get("result") or {}).get("count", 0))
