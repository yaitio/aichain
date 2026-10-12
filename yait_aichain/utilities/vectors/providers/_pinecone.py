"""
utilities.vectors.providers._pinecone
=====================================

PineconeBackend — VectorBackend adapter for Pinecone via its REST API.

Uses pure urllib3, no ``pinecone`` SDK required.

Pinecone concepts vs aichain mapping
--------------------------------------
Pinecone concept    aichain concept
─────────────────   ──────────────────────────────────
Index               backend-level (set via ``index=`` at construction)
Namespace           collection  (each VectorStore maps to one namespace)
Vector              embedding + text stored in metadata["text"]

Why text in metadata?
  Pinecone stores only dense vectors + key-value metadata.  We persist
  the original text as metadata["text"] so fetch() and query() can
  reconstruct VectorRecord.text.

Environment variables
---------------------
``PINECONE_API_KEY``   — required; get from console.pinecone.io
``PINECONE_INDEX``     — default index name (optional; can pass index= directly)

Pinecone REST endpoints
------------------------
Control plane  https://api.pinecone.io
  POST   /indexes                   — create serverless index
  GET    /indexes/{name}            — describe index (returns host URL)

Data plane     https://{index-host}
  POST   /vectors/upsert            — upsert vectors
  POST   /query                     — similarity search
  GET    /vectors/fetch             — fetch by IDs
  POST   /vectors/delete            — delete by IDs / filter / deleteAll
  GET    /describe_index_stats      — namespace stats (includes vector count)
"""

from __future__ import annotations

import os
from urllib.parse import urlencode

from ....clients._errors import APIError
from .._base import VectorBackend, VectorRecord, normalise_score


# ── Metric mapping ─────────────────────────────────────────────────────────────

_METRIC_MAP: dict[str, str] = {
    "cosine":    "cosine",
    "euclidean": "euclidean",
    "dot":       "dotproduct",
}

_CONTROL_URL = "https://api.pinecone.io"


# ── PineconeBackend ────────────────────────────────────────────────────────────

class PineconeBackend(VectorBackend):
    """
    VectorBackend for Pinecone (serverless REST API).

    Parameters
    ----------
    index : str
        Name of the Pinecone index.  Defaults to ``PINECONE_INDEX`` env var.
    api_key : str | None
        Pinecone API key.  Defaults to ``PINECONE_API_KEY`` env var.
    url : str | None
        Index host URL (``https://<index-host>``).  When omitted the host
        is auto-discovered from the control plane on the first data call.
    cloud : str
        Cloud for new index creation (default ``"aws"``).
    region : str
        Region for new index creation (default ``"us-east-1"``).

    Examples
    --------
    ::

        from yait_aichain.utilities.vectors import VectorDB
        from yait_aichain.tools import embed

        store = VectorDB(
            "pinecone",
            collection = "my_namespace",
            embedder   = embed("openai", model="text-embedding-3-small"),
            index      = "my-index",
        )
        store.upsert([{"id": "1", "text": "hello world"}])
        results = store.query("greeting", n=3)
    """

    #: Default at class level as well as in `__init__`: half-built instances
    #: exist (a test constructs one with `__new__`), and a missing attribute
    #: there would turn a scoring question into an AttributeError.
    _metric = "cosine"

    provider = "pinecone"

    def __init__(
        self,
        *,
        index:   str | None = None,
        api_key: str | None = None,
        url:     str | None = None,
        cloud:   str        = "aws",
        region:  str        = "us-east-1",
        metric:  str        = "cosine",
        **kwargs,
    ) -> None:
        # Named here because the data plane cannot discover it: the metric is
        # a property of the index, on the control plane. Without it `score`
        # would carry a cosine similarity in [-1, 1] on one collection and a
        # squared distance — lower is nearer — on another, under one name.
        self._metric = metric
        resolved_key = api_key or os.environ.get("PINECONE_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "No Pinecone API key found.  "
                "Pass api_key= or set the 'PINECONE_API_KEY' environment variable."
            )
        self._index     = index or os.environ.get("PINECONE_INDEX", "")
        self._index_url = url or ""    # cached after first describe-index call
        self._cloud     = cloud
        self._region    = region

        # The control plane and the index host are both full URLs, so one pool
        # — the base class's — serves both. _url is set once the host is known.
        super().__init__(url=url or "", api_key=resolved_key)

    # ── Auth headers ───────────────────────────────────────────────────────────

    def _headers(self) -> dict:
        # Pinecone uses Api-Key header, not Bearer
        return {
            "Api-Key":      self._api_key,
            "Content-Type": "application/json",
            "Accept":       "application/json",
        }

    # ── Index host resolution ──────────────────────────────────────────────────

    def _index_url_resolved(self) -> str:
        if self._index_url:
            return self._index_url
        if not self._index:
            raise ValueError(
                "No Pinecone index name configured.  "
                "Pass index= to VectorDB() or set the 'PINECONE_INDEX' env var."
            )
        host = self._call("GET", f"{_CONTROL_URL}/indexes/{self._index}").get("host", "")
        self._index_url = host if host.startswith("http") else f"https://{host}"
        self._url       = self._index_url   # sync base URL for shared helpers
        return self._index_url

    # ── Abstract method implementations ───────────────────────────────────────

    def create_collection(
        self,
        collection: str,
        dimension:  int,
        metric:     str = "cosine",
    ) -> None:
        """
        Create a serverless Pinecone index.

        In Pinecone the index is shared; *collection* becomes a namespace
        inside it.  This method creates the index if it does not exist.
        HTTP 409 (already exists) is treated as success.
        """
        if not self._index:
            self._index = collection  # use collection name as index name if unset
        body = {
            "name":      self._index,
            "dimension": dimension,
            "metric":    _METRIC_MAP.get(metric, "cosine"),
            "spec": {
                "serverless": {
                    "cloud":  self._cloud,
                    "region": self._region,
                }
            },
        }
        try:
            self._call("POST", f"{_CONTROL_URL}/indexes", body)
        except APIError as exc:
            if exc.status != 409:                 # 409: it exists already, which is the goal
                raise

    def query(
        self,
        collection: str,
        vector:     list[float],
        n:          int = 5,
        filter:     dict | None = None,
    ) -> list[VectorRecord]:
        host = self._index_url_resolved()
        body: dict = {
            "vector":           vector,
            "topK":             n,
            "namespace":        collection,
            "includeMetadata":  True,
            "includeValues":    False,
        }
        if filter:
            body["filter"] = filter

        data = self._call("POST", f"{host}/query", body)

        records: list[VectorRecord] = []
        for match in data.get("matches", []):
            meta = dict(match.get("metadata") or {})
            # NB: not meta.pop("text", meta.pop("document", "")) — Python
            # evaluates the default eagerly, which would silently strip a
            # user's own "document" metadata key even when "text" exists.
            text = meta.pop("text", None)
            if text is None:
                text = meta.pop("document", "")
            raw = match.get("score")
            records.append(VectorRecord(
                id        = match["id"],
                text      = text,
                # Pinecone's metric is a property of the index and lives on
                # the control plane, which this data-plane host cannot ask.
                # It is taken at construction (`metric=`) and defaults to the
                # same cosine everything else defaults to; `raw_score` keeps
                # the original either way, so a caller who set a different
                # metric and did not say so can still see what came back.
                score     = normalise_score(raw, self._metric),
                raw_score = raw,
                metadata  = meta,
            ))
        return records

    def upsert(
        self,
        collection: str,
        records:    list[VectorRecord],
        vectors:    list[list[float]],
    ) -> None:
        host    = self._index_url_resolved()
        payload = []
        for rec, vec in zip(records, vectors):
            # store text inside metadata so fetch() can reconstruct it
            meta = {**rec.metadata, "text": rec.text}
            payload.append({"id": rec.id, "values": vec, "metadata": meta})
        body = {"vectors": payload, "namespace": collection}
        self._call("POST", f"{host}/vectors/upsert", body)

    def fetch(
        self,
        collection: str,
        ids:        list[str],
    ) -> list[VectorRecord]:
        host   = self._index_url_resolved()
        params = urlencode([("ids", i) for i in ids] + [("namespace", collection)])
        data = self._call("GET", f"{host}/vectors/fetch?{params}")

        records: list[VectorRecord] = []
        for rid, vec_data in (data.get("vectors") or {}).items():
            meta = dict((vec_data.get("metadata") or {}))
            text = meta.pop("text", "")
            records.append(VectorRecord(id=rid, text=text, metadata=meta))
        return records

    def delete(
        self,
        collection: str,
        ids:        list[str] | None = None,
        filter:     dict | None      = None,
    ) -> None:
        host  = self._index_url_resolved()
        body: dict = {"namespace": collection}
        if ids:
            body["ids"] = ids
        if filter:
            body["filter"] = filter
        if not ids and not filter:
            body["deleteAll"] = True
        self._call("POST", f"{host}/vectors/delete", body)

    def count(self, collection: str) -> int:
        host = self._index_url_resolved()
        data = self._call("GET", f"{host}/describe_index_stats")
        ns   = (data.get("namespaces") or {}).get(collection, {})
        return int(ns.get("vectorCount", 0))
