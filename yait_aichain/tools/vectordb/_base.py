"""
tools.vectordb._base
====================

VectorRecord  — uniform data model across all vector DB providers.
VectorBackend — abstract base class for provider REST adapters.
VectorStore   — orchestrator: wraps a backend + optional Embedder.

Design
------
Every concrete backend:
  1. Sets ``provider`` class attribute.
  2. Implements the six abstract methods:
       create_collection, query, upsert, fetch, delete, count
  3. Uses the shared HTTP helpers (_get, _post, _put, _delete_req)
     which all use urllib3 — no provider SDKs required.

VectorStore sits above the backend and:
  • Holds the active collection name (so callers don't repeat it)
  • Owns the optional Embedder for auto-vectorising text
  • Exposes a clean public API: create / query / upsert / fetch / delete / count

Use the VectorDB() factory in __init__.py rather than constructing
VectorStore + backend directly.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import urllib3
from yait_aichain.clients._base import make_http

from ...clients._constants import DEFAULT_TIMEOUT, DEFAULT_RETRIES


# ---------------------------------------------------------------------------
# VectorRecord
# ---------------------------------------------------------------------------

@dataclass
class VectorRecord:
    """
    Uniform data model for a single vector DB entry.

    Attributes
    ----------
    id       : str            — unique document identifier
    text     : str            — original text content
    score    : float | None   — similarity score (populated on query results only)
    metadata : dict           — arbitrary key-value payload / metadata
    """

    id:       str
    text:     str
    score:    float | None = None
    metadata: dict         = field(default_factory=dict)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Return a plain dict (JSON-serialisable)."""
        d: dict = {"id": self.id, "text": self.text, "metadata": self.metadata}
        if self.score is not None:
            d["score"] = self.score
        return d

    # ── Dunder helpers ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        preview   = self.text[:50].replace("\n", " ")
        score_str = f", score={self.score:.4f}" if self.score is not None else ""
        return f"VectorRecord(id={self.id!r}, text={preview!r}{score_str})"


# ---------------------------------------------------------------------------
# VectorBackend — abstract
# ---------------------------------------------------------------------------

class VectorBackend:
    """
    Abstract base class for vector DB provider REST adapters.

    Subclasses must:
      1. Set ``provider`` (class attribute).
      2. Call ``super().__init__(url=..., api_key=...)`` in their ``__init__``.
      3. Implement all six abstract methods.

    Shared HTTP helpers (_get, _post, _put, _delete_req) handle JSON
    encoding/decoding and error checking so subclasses stay concise.

    Environment variables
    ---------------------
    Each backend reads its URL and API key from environment variables when
    the caller does not pass them explicitly.  Subclasses document the
    specific variable names.
    """

    provider: str = ""

    # ── Construction ──────────────────────────────────────────────────────────

    def __init__(
        self,
        *,
        url:     str,
        api_key: str | None = None,
        **_kwargs: Any,
    ) -> None:
        self._url     = url.rstrip("/")
        self._api_key = api_key
        self._http    = make_http(
            timeout = DEFAULT_TIMEOUT,
            retries = DEFAULT_RETRIES,
        )

    # ── Abstract methods — must be implemented by every subclass ──────────────

    def create_collection(
        self,
        collection: str,
        dimension:  int,
        metric:     str = "cosine",
    ) -> None:
        """Create a new collection / index with the given vector dimension."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement create_collection()"
        )

    def query(
        self,
        collection: str,
        vector:     list[float],
        n:          int = 5,
        filter:     dict | None = None,
    ) -> list[VectorRecord]:
        """Return the *n* most similar records to *vector*."""
        raise NotImplementedError(f"{type(self).__name__} must implement query()")

    def upsert(
        self,
        collection: str,
        records:    list[VectorRecord],
        vectors:    list[list[float]],
    ) -> None:
        """Insert or update *records* together with their *vectors*."""
        raise NotImplementedError(f"{type(self).__name__} must implement upsert()")

    def fetch(
        self,
        collection: str,
        ids:        list[str],
    ) -> list[VectorRecord]:
        """Return records matching *ids* (no score populated)."""
        raise NotImplementedError(f"{type(self).__name__} must implement fetch()")

    def delete(
        self,
        collection: str,
        ids:        list[str] | None = None,
        filter:     dict | None      = None,
    ) -> None:
        """Delete records by *ids* or metadata *filter* (at least one required)."""
        raise NotImplementedError(f"{type(self).__name__} must implement delete()")

    def count(self, collection: str) -> int:
        """Return the number of records in *collection*."""
        raise NotImplementedError(f"{type(self).__name__} must implement count()")

    # ── Shared HTTP helpers ───────────────────────────────────────────────────

    def _headers(self) -> dict:
        """Base headers.  Subclasses may override to change auth scheme."""
        h: dict = {
            "Content-Type": "application/json",
            "Accept":       "application/json",
        }
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

    def _get(self, path: str, params: dict | None = None) -> Any:
        from urllib.parse import urlencode
        url = self._url + path
        if params:
            url += "?" + urlencode(params)
        resp = self._http.request("GET", url, headers=self._headers())
        return self._parse(resp)

    def _post(self, path: str, body: dict) -> Any:
        resp = self._http.request(
            "POST",
            self._url + path,
            body    = json.dumps(body).encode(),
            headers = self._headers(),
        )
        return self._parse(resp)

    def _put(self, path: str, body: dict) -> Any:
        resp = self._http.request(
            "PUT",
            self._url + path,
            body    = json.dumps(body).encode(),
            headers = self._headers(),
        )
        return self._parse(resp)

    def _delete_req(self, path: str, body: dict | None = None) -> Any:
        kw: dict = {"headers": self._headers()}
        if body is not None:
            kw["body"] = json.dumps(body).encode()
        resp = self._http.request("DELETE", self._url + path, **kw)
        return self._parse(resp)

    def _parse(self, resp: urllib3.BaseHTTPResponse) -> Any:
        """Decode JSON response; raise RuntimeError on non-2xx status."""
        raw = resp.data.decode("utf-8", errors="replace")
        if not (200 <= resp.status < 300):
            raise RuntimeError(
                f"{self.provider} API error [{resp.status}]: {raw[:400]}"
            )
        if not raw.strip():
            return {}
        return json.loads(raw)

    # ── Dunder helpers ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"{type(self).__name__}(provider={self.provider!r}, url={self._url!r})"


# ---------------------------------------------------------------------------
# VectorStore — orchestrator
# ---------------------------------------------------------------------------

class VectorStore:
    """
    Provider-agnostic vector store.

    Wraps a :class:`VectorBackend` and an optional :class:`~tools.embedding.Embedder`.
    Fixes the active ``collection`` so callers never repeat it.

    Parameters
    ----------
    backend    : VectorBackend
        A concrete backend (Chroma, Pinecone, Qdrant, …).
    collection : str
        Default collection / namespace for all operations.
    embedder   : Embedder | None, optional
        When provided, text is automatically vectorised before storage and
        at query time.  When omitted, callers must supply pre-computed
        ``vector`` fields in their records.

    Examples
    --------
    With auto-embedding::

        from tools.embedding import Embedding
        from tools.vectordb  import VectorDB

        store = VectorDB(
            "chroma",
            collection = "knowledge_base",
            embedder   = Embedding("openai/text-embedding-3-small"),
        )
        store.upsert([{"id": "1", "text": "LLMs use KV cache …"}])
        results = store.query("how does caching work?", n=3)

    Without embedding (supply vectors yourself)::

        store = VectorDB("qdrant", collection="dense_index")
        store.upsert([{"id": "1", "text": "…", "vector": [0.1, 0.2, …]}])
    """

    def __init__(
        self,
        backend:    VectorBackend,
        collection: str,
        embedder:   Any | None = None,
    ) -> None:
        self._backend    = backend
        self._collection = collection
        self._embedder   = embedder

    # ── Public API ────────────────────────────────────────────────────────────

    def create(self, dimension: int, metric: str = "cosine") -> None:
        """
        Create the collection in the backend.

        Parameters
        ----------
        dimension : int
            Vector size (must match the embedder's output dimension).
        metric    : ``"cosine"`` | ``"euclidean"`` | ``"dot"``
            Distance metric used for similarity ranking.
        """
        self._backend.create_collection(self._collection, dimension, metric)

    def query(
        self,
        text:   str,
        n:      int = 5,
        filter: dict | None = None,
    ) -> list[VectorRecord]:
        """
        Semantic similarity search.

        Embeds *text* (requires an embedder) and returns the *n* most
        similar records, sorted by descending similarity score.

        Parameters
        ----------
        text   : str          — query string
        n      : int          — number of results (default 5)
        filter : dict | None  — provider-native metadata filter

        Returns
        -------
        list[VectorRecord]
            Records with ``score`` populated (highest score = most similar).
        """
        vector = self._embed_query(text)
        return self._backend.query(self._collection, vector, n=n, filter=filter)

    # Records per backend write.  Sized for the strictest provider limit:
    # Pinecone caps upsert requests at 2 MB / 1000 vectors — 50 records of
    # 1536-dim JSON-encoded floats stay safely under both.
    _UPSERT_BATCH_SIZE: int = 50

    def upsert(self, records: list[dict], batch_size: "int | None" = None) -> int:
        """
        Insert or update records.

        Each dict must have:
          ``id``       : str            — unique identifier
          ``text``     : str            — document content
          ``metadata`` : dict           — optional payload / metadata
          ``vector``   : list[float]    — optional pre-computed vector
                                          (required when no embedder is set)

        Missing vectors are auto-computed in a single batched embedding call.
        Writes are sent to the backend in batches of *batch_size* records
        (default 50) so large ingests don't exceed provider request limits.

        Returns
        -------
        int
            Number of records upserted.
        """
        recs, vectors = self._prepare_records(records)
        step = batch_size or self._UPSERT_BATCH_SIZE
        if step <= 0:
            raise ValueError(f"batch_size must be positive; got {step}")
        for i in range(0, len(recs), step):
            self._backend.upsert(
                self._collection, recs[i : i + step], vectors[i : i + step]
            )
        return len(recs)

    def fetch(self, ids: list[str]) -> list[VectorRecord]:
        """Retrieve specific records by ID (no score)."""
        return self._backend.fetch(self._collection, ids)

    def delete(
        self,
        ids:    list[str] | None = None,
        filter: dict | None      = None,
    ) -> None:
        """
        Remove records.

        Pass *ids* to delete specific records, *filter* to delete by
        metadata condition, or both.  At least one must be provided and
        non-empty: an empty list/dict is rejected rather than treated as
        "match everything", because some backends (Pinecone, Qdrant) would
        otherwise interpret it as a full-collection wipe.
        """
        if not ids and not filter:
            raise ValueError(
                "Provide at least one non-empty argument: ids, filter. "
                "An empty list/dict is rejected to avoid accidentally "
                "deleting the whole collection."
            )
        # Normalise empties so backends never see ids=[] / filter={}.
        self._backend.delete(
            self._collection,
            ids=ids or None,
            filter=filter or None,
        )

    def count(self) -> int:
        """Return the total number of records in the collection."""
        return self._backend.count(self._collection)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _embed_query(self, text: str) -> list[float]:
        if self._embedder is None:
            raise ValueError(
                "No embedder configured.  Pass embedder= to VectorDB() "
                "or supply pre-computed vectors in your records."
            )
        result = self._embedder.embed(text, input_type="query")
        return result.embeddings[0]

    def _prepare_records(
        self,
        raw: list[dict],
    ) -> tuple[list[VectorRecord], list[list[float]]]:
        """
        Build (VectorRecord list, vector list) from raw dicts.

        Records that already have a ``vector`` key are used as-is.
        The rest are batched into a single embedder call.
        """
        recs:    list[VectorRecord] = []
        vectors: list[list[float]]  = []

        needs_embed = [r for r in raw if "vector" not in r]

        if needs_embed:
            if self._embedder is None:
                missing = needs_embed[0]["id"]
                raise ValueError(
                    f"Record {missing!r} has no 'vector' and no embedder is set.  "
                    f"Pass embedder= to VectorDB() or include 'vector' in every record."
                )
            texts  = [r["text"] for r in needs_embed]
            result = self._embedder.embed(texts, input_type="document")
            # Pair embeddings to records POSITIONALLY (in input order), not by id.
            # `needs_embed` follows the order of `raw`, so the same iterator
            # consumed below assigns each freshly-embedded vector to its own
            # record — correct even when two records share an id (an id-keyed
            # map would collapse duplicates and mis-assign vectors).
            embeds = iter(result.embeddings)
        else:
            embeds = iter(())

        for r in raw:
            rec = VectorRecord(
                id       = str(r["id"]),
                text     = r["text"],
                metadata = r.get("metadata", {}),
            )
            vec = r["vector"] if "vector" in r else next(embeds)
            recs.append(rec)
            vectors.append(vec)

        return recs, vectors

    # ── Dunder helpers ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"VectorStore(provider={self._backend.provider!r}, "
            f"collection={self._collection!r})"
        )
