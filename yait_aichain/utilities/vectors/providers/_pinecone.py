"""``PineconeBackend``: Pinecone over its REST API, with urllib3 and no SDK. The
index is set at construction (``index=`` or ``PINECONE_INDEX``), a collection is
a namespace, and a record's text is kept in ``metadata["text"]`` since Pinecone
stores only vectors and metadata. ``PINECONE_API_KEY`` is required. See docs/utilities/index.md.
"""

from __future__ import annotations

import os
import time
from urllib.parse import urlencode

from ....clients._errors import APIError
from .._base import VectorBackend, VectorRecord, _one_of, check_metric, record_from_payload

_CONTROL_URL = "https://api.pinecone.io"


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

    provider = "pinecone"
    FILTER   = ('Pinecone metadata filter: {"source": {"$eq": "arxiv"}}, '
                '{"year": {"$gte": 2020}}, {"$and": [{...}, {...}]}')
    _METRICS = {"cosine": "cosine", "euclidean": "euclidean", "dot": "dotproduct"}
    _AUTH, _ENV_KEY = "Api-Key", "PINECONE_API_KEY"
    #: How long a new index may take to come up before create gives up.
    READY_TIMEOUT = 120.0

    def __init__(
        self,
        *,
        index:   str | None = None,
        api_key: str | None = None,
        url:     str | None = None,
        cloud:   str        = "aws",
        region:  str        = "us-east-1",
        metric:  str        = "cosine",
    ) -> None:
        # Named here because the data plane cannot discover it: the metric is
        # a property of the index, on the control plane.
        self._metric = check_metric(metric, self._METRICS, "Pinecone")
        if not (api_key or os.environ.get("PINECONE_API_KEY")):
            raise ValueError(
                "No Pinecone API key found.  "
                "Pass api_key= or set the 'PINECONE_API_KEY' environment variable."
            )
        # The control plane and the index host are both full URLs, so one pool
        # serves both; the host is discovered on the first data call when not given.
        super().__init__(url=url, api_key=api_key)
        self._index  = index or os.environ.get("PINECONE_INDEX", "")
        self._index_url, self._cloud, self._region = url or "", cloud, region

    def _host(self) -> str:
        """The index host, asked of the control plane once."""
        if not self._index_url:
            if not self._index:
                raise ValueError("No Pinecone index name configured.  "
                                 "Pass index= to VectorDB() or set the 'PINECONE_INDEX' env var.")
            host = self._describe().get("host", "")
            self._index_url = self._url = host if host.startswith("http") else f"https://{host}"
        return self._index_url

    def _describe(self) -> dict:
        return self._call("GET", f"{_CONTROL_URL}/indexes/{self._index}")

    def _record(self, match: dict) -> VectorRecord:
        """One match or fetched vector; Pinecone's euclidean is the squared distance."""
        metric = "sqeuclidean" if self._metric == "euclidean" else self._metric
        return record_from_payload(match["id"], match.get("metadata"), match.get("score"), metric)

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
        self._index = self._index or collection   # the collection name as the index name if unset
        body = {"name": self._index, "dimension": dimension, "metric": self._native(metric),
                "spec": {"serverless": {"cloud": self._cloud, "region": self._region}}}
        try:
            self._call("POST", f"{_CONTROL_URL}/indexes", body)
        except APIError as exc:
            if exc.status != 409:
                raise
            info = self._describe()
            metric = self._adopt(f"index {self._index!r}", dimension, metric,
                                 info.get("dimension"), info.get("metric"))
        self._metric = metric
        # A serverless index takes seconds to come up, and a write before then is refused.
        deadline = time.monotonic() + self.READY_TIMEOUT
        while not (self._describe().get("status") or {}).get("ready", True):
            if time.monotonic() > deadline:
                raise TimeoutError(f"Pinecone index {self._index!r} was not ready after "
                                   f"{self.READY_TIMEOUT:.0f}s")
            time.sleep(1.0)

    def query(
        self,
        collection: str,
        vector:     list[float],
        n:          int = 5,
        filter:     dict | None = None,
    ) -> list[VectorRecord]:
        body: dict = {"vector": vector, "topK": n, "namespace": collection,
                      "includeMetadata": True, "includeValues": False}
        if filter:
            body["filter"] = filter
        data = self._call("POST", f"{self._host()}/query", body)
        return [self._record(m) for m in data.get("matches", [])]

    def upsert(
        self,
        collection: str,
        records:    list[VectorRecord],
        vectors:    list[list[float]],
    ) -> None:
        # The text rides in the metadata: Pinecone stores only vectors and metadata.
        payload = [{"id": rec.id, "values": vec, "metadata": {**rec.metadata, "text": rec.text}}
                   for rec, vec in zip(records, vectors)]
        self._call("POST", f"{self._host()}/vectors/upsert", {"vectors": payload, "namespace": collection})

    def fetch(
        self,
        collection: str,
        ids:        list[str],
    ) -> list[VectorRecord]:
        params = urlencode([("ids", i) for i in ids] + [("namespace", collection)])
        data = self._call("GET", f"{self._host()}/vectors/fetch?{params}")
        return [self._record({**v, "id": rid}) for rid, v in (data.get("vectors") or {}).items()]

    def delete(
        self,
        collection: str,
        ids:        list[str] | None = None,
        filter:     dict | None      = None,
    ) -> None:
        _one_of(ids, filter)   # nothing named would be `deleteAll` here — the whole namespace
        body = {"namespace": collection, **({"ids": ids} if ids else {"filter": filter})}
        self._call("POST", f"{self._host()}/vectors/delete", body)

    def count(self, collection: str) -> int:
        data = self._call("GET", f"{self._host()}/describe_index_stats")
        return int((data.get("namespaces") or {}).get(collection, {}).get("vectorCount", 0))
