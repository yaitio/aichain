"""The vector data model and its two layers: ``VectorRecord`` (``id``, ``text``,
``score``, ``metadata``), ``VectorBackend`` (a provider's REST API over
urllib3, no SDK) and ``VectorStore`` (a backend, an optional embedder and one
collection). Scores are normalised by one shared rule, ``normalise_score``.
See docs/utilities/index.md.
"""

from __future__ import annotations

from ...tools._key import ApiKey

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from yait_aichain.clients._base import make_http

from ...clients._http import request
from ...clients._constants import TOOL_TIMEOUT, DEFAULT_RETRIES


# ---------------------------------------------------------------------------
# VectorRecord
# ---------------------------------------------------------------------------

#: Metrics the library names, and how each backend's raw number becomes a
#: similarity in [0, 1] where higher is better.
#:
#: The formulas are not interchangeable and the direction is the reason:
#: cosine and dot are *similarities* (higher is nearer), euclidean is a
#: *distance* (lower is nearer). Folding them into one field without
#: converting is how a threshold tuned on one collection silently inverts on
#: another.
METRICS = ("cosine", "euclidean", "dot")


def normalise_score(raw: float | None, metric: str) -> float | None:
    """A backend's raw number as a similarity in [0, 1], higher is better.

    * **cosine** — a similarity in [-1, 1]; mapped linearly onto [0, 1].
    * **euclidean** — an unbounded distance, lower being nearer; ``1/(1+d)``,
      which is monotonic, bounded, and inverts the direction.
    * **dot** — unbounded and scale-dependent, so it cannot be mapped onto
      [0, 1] without knowing the vectors' magnitudes. Passed through, and
      that is stated rather than papered over: a made-up bound would be worse
      than an honest raw number, because it would look comparable.

    ``None`` in, ``None`` out — a fetch by id has no score and inventing one
    would make an exact lookup look like a ranked hit.
    """
    if raw is None:
        return None
    if metric == "cosine":
        return max(0.0, min(1.0, (float(raw) + 1.0) / 2.0))
    if metric in ("euclidean", "manhattan"):
        return 1.0 / (1.0 + max(0.0, float(raw)))
    if metric == "sqeuclidean":                 # Pinecone's and Chroma's l2: the square
        return 1.0 / (1.0 + max(0.0, float(raw)) ** 0.5)
    if metric == "dot":
        return float(raw)
    raise ValueError(f"no metric {metric!r}; one of {list(METRICS)}")


#: The metrics a store may be created with; a backend takes those it has.
#: An unknown one is refused, not quietly created as cosine.
METRICS = ("cosine", "euclidean", "dot", "manhattan")


def check_metric(metric: str, accepted, who: str) -> str:
    if metric not in accepted:
        raise ValueError(f"{who} has no metric {metric!r}; it takes {sorted(accepted)}")
    return metric


def _one_of(ids, filter) -> None:
    """A delete names records by id or by filter — exactly one: "both" means
    three things across three backends (Qdrant drops the filter, Chroma takes
    both), and "neither" the whole collection on Pinecone."""
    if ids and filter:
        raise ValueError("delete takes ids or a filter, not both — the backends disagree on "
                         "what both would mean; call it twice")
    if not ids and not filter:
        raise ValueError("delete needs ids or a filter; nothing named would delete the "
                         "whole collection on some backends, and is refused on all")


class PartialUpsert(RuntimeError):
    """An upsert that stopped part way: ``written`` of ``total`` records are
    stored, and the rest are not. The cause is chained."""

    def __init__(self, written: int, total: int, cause: BaseException) -> None:
        super().__init__(f"upsert stopped after {written} of {total} records: "
                         f"{type(cause).__name__}: {cause}")
        self.written, self.total = written, total


@dataclass
class VectorRecord:
    """
    Uniform data model for a single vector DB entry.

    Attributes
    ----------
    id       : str            — unique document identifier
    text     : str            — original text content
    score    : float | None   — similarity in [0, 1], higher is more similar
    raw_score: float | None   — what the backend returned, unconverted
    metadata : dict           — arbitrary key-value payload / metadata

    **``score`` means one thing on every backend**: a similarity in [0, 1],
    higher is closer, whatever the backend's metric. The backends answer on
    three scales — a cosine in [-1, 1], a normalised score, a euclidean
    *distance* where lower is closer — and a threshold or a comparison of
    retrievers read across them would be reading three scales as one.

    ``raw_score`` keeps whatever the backend said. Normalising is a
    conversion, and a conversion that throws the original away cannot be
    checked.
    """

    id:        str
    text:      str
    score:     float | None = None
    metadata:  dict         = field(default_factory=dict)
    raw_score: float | None = None

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Return a plain dict (JSON-serialisable)."""
        d: dict = {"id": self.id, "text": self.text, "metadata": self.metadata}
        if self.score is not None:
            d["score"] = self.score
        if self.raw_score is not None:
            d["raw_score"] = self.raw_score
        return d

    # ── Dunder helpers ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        preview   = self.text[:50].replace("\n", " ")
        score_str = f", score={self.score:.4f}" if self.score is not None else ""
        return f"VectorRecord(id={self.id!r}, text={preview!r}{score_str})"


# ---------------------------------------------------------------------------
# VectorBackend — abstract
# ---------------------------------------------------------------------------

#: Keys a backend keeps beside the metadata in the same payload; a record's
#: own metadata may not use them, or it would overwrite the text or the id.
ID_KEY   = "_aichain_id"
RESERVED = ("text", ID_KEY)


def record_from_payload(rid: str, payload: dict | None, raw: float | None = None,
                        metric: str = "cosine") -> VectorRecord:
    """A record from a backend that keeps text and metadata in one payload.

    The text is under ``text``; ``document`` is read when ``text`` is absent,
    for collections written by other tools. A string id that a backend could
    not store as its own sits under ``ID_KEY``.
    """
    meta = dict(payload or {})
    text = meta.pop("text", None)
    if text is None:
        text = meta.pop("document", "")
    rid = meta.pop(ID_KEY, None) or rid
    return VectorRecord(id=str(rid), text=text or "", metadata=meta,
                        score=normalise_score(raw, metric), raw_score=raw)


class VectorBackend(ABC):
    """
    Abstract base class for vector DB provider REST adapters.

    Subclasses must:
      1. Set ``provider`` (class attribute).
      2. Call ``super().__init__(url=..., api_key=...)`` in their ``__init__``.
      3. Implement the six abstract methods, and name the dialect its
         ``filter`` is written in as ``FILTER``.

    Shared HTTP helpers (_get, _post, _put, _delete_req) handle JSON
    encoding/decoding and error checking so subclasses stay concise.

    Environment variables
    ---------------------
    Each backend reads its URL and API key from environment variables when
    the caller does not pass them explicitly.  Subclasses document the
    specific variable names.
    """

    _api_key = ApiKey()            # a string, or resolved per request — see tools._key

    provider: str = ""
    #: How a ``filter`` is written for this backend, shown to a model.
    FILTER: str = ""

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
            timeout = TOOL_TIMEOUT,
            retries = DEFAULT_RETRIES,
        )

    # ── What every backend implements ─────────────────────────────────────────

    @abstractmethod
    def create_collection(
        self,
        collection: str,
        dimension:  int,
        metric:     str = "cosine",
    ) -> None:
        """Create a new collection / index with the given vector dimension."""

    @abstractmethod
    def query(
        self,
        collection: str,
        vector:     list[float],
        n:          int = 5,
        filter:     dict | None = None,
    ) -> list[VectorRecord]:
        """Return the *n* most similar records to *vector*."""

    @abstractmethod
    def upsert(
        self,
        collection: str,
        records:    list[VectorRecord],
        vectors:    list[list[float]],
    ) -> None:
        """Insert or update *records* together with their *vectors*."""

    @abstractmethod
    def fetch(
        self,
        collection: str,
        ids:        list[str],
    ) -> list[VectorRecord]:
        """Return records matching *ids* (no score populated)."""

    @abstractmethod
    def delete(
        self,
        collection: str,
        ids:        list[str] | None = None,
        filter:     dict | None      = None,
    ) -> None:
        """Delete records by *ids* or metadata *filter* (at least one required)."""

    @abstractmethod
    def count(self, collection: str) -> int:
        """Return the number of records in *collection*."""

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

    def _call(self, method: str, path: str, body: dict | None = None) -> Any:
        """One request to the backend — *path* under its URL, or a full URL — and
        its JSON, ``{}`` for an empty body, or a typed `APIError` naming the backend."""
        url = path if path.startswith(("http://", "https://")) else self._url + path
        raw = request(self._http, method, url, json=body, headers=self._headers(),
                      what=f"{self.provider} API").data.decode("utf-8", errors="replace")
        return json.loads(raw) if raw.strip() else {}

    def _get(self, path: str, params: dict | None = None) -> Any:
        from urllib.parse import urlencode
        return self._call("GET", path + ("?" + urlencode(params) if params else ""))

    def _post(self, path: str, body: dict) -> Any:
        return self._call("POST", path, body)

    def _put(self, path: str, body: dict) -> Any:
        return self._call("PUT", path, body)

    def _delete_req(self, path: str, body: dict | None = None) -> Any:
        return self._call("DELETE", path, body)

    # ── Dunder helpers ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"{type(self).__name__}(provider={self.provider!r}, url={self._url!r})"


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------

class VectorStore:
    """
    Provider-agnostic vector store.

    Wraps a :class:`VectorBackend` and an optional embedder (``embed(...)``).
    This is what ``Vectors`` is built on — ``kb.store``.
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

        from yait_aichain.tools import embed
        from yait_aichain.utilities.vectors  import VectorDB

        store = VectorDB(
            "chroma",
            collection = "knowledge_base",
            embedder   = embed("openai", model="text-embedding-3-small"),
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
        # Part of every request's path: one that walks it, or ends it, is refused.
        import re
        if not isinstance(collection, str) or not re.fullmatch(r"[A-Za-z0-9_\-][A-Za-z0-9_.\-]{0,254}",
                                                                collection) or collection in (".", ".."):
            raise ValueError(f"a collection name is letters, digits, '_', '-' and '.'; "
                             f"got {collection!r}")
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
        check_metric(metric, METRICS, "a vector store")
        self._backend.create_collection(self._collection, dimension, metric)

    def query(
        self,
        text:   str,
        n:      int = 5,
        filter: dict | None = None,
    ) -> list[VectorRecord]:
        """Semantic similarity search: the *n* records most similar to *text*.

        Parameters
        ----------
        text   : str          — query string (needs an embedder)
        n      : int          — number of results (default 5)
        filter : dict | None  — provider-native metadata filter

        Returns
        -------
        list[VectorRecord]
            Most similar first; ``score`` in [0, 1] on every backend except
            for the unbounded ``dot`` metric, ``raw_score`` as the backend gave it.
        """
        vector = self._embed_query(text)
        return self._backend.query(self._collection, vector, n=n, filter=filter)

    # Records per backend write.  Sized for the strictest provider limit:
    # Pinecone caps upsert requests at 2 MB / 1000 vectors — 50 records of
    # 1536-dim JSON-encoded floats stay safely under both.
    _UPSERT_BATCH_SIZE: int = 50
    # And by size: 50 records of 3,072 floats is 3.4 MB of JSON, past
    # Pinecone's 2 MB. A float is about 20 bytes written out.
    _UPSERT_MAX_BYTES: int = 1_500_000

    def upsert(self, records: list[dict], batch_size: int | None = None) -> int:
        """Insert or update records: dicts of ``id``, ``text``, optional
        ``metadata`` and ``vector`` (required without an embedder).

        Written in batches of *batch_size* (default 50); a failed one keeps those before it.

        Returns
        -------
        int
            Number of records upserted.

        Raises
        ------
        PartialUpsert
            When a batch fails; ``written`` counts the records stored before it.
        """
        step = self._UPSERT_BATCH_SIZE if batch_size is None else batch_size
        if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
            raise ValueError(f"batch_size must be a positive whole number; got {batch_size!r}")
        records = list(records)
        self._check_records(records)              # a bad record is refused before anything is written
        written = 0
        for batch in self._write_batches(records, step):
            try:
                recs, vectors = self._prepare_records(batch)
                for part in self._by_bytes(recs, vectors):
                    self._backend.upsert(self._collection, *part)
                    written += len(part[0])
            except Exception as exc:
                raise PartialUpsert(written, len(records), exc) from exc
        return written

    @staticmethod
    def _write_batches(records: list[dict], step: int):
        for i in range(0, len(records), step):
            yield records[i:i + step]

    def _by_bytes(self, recs: list, vectors: list):
        """Split one batch where its estimated JSON passes the byte limit."""
        start, size = 0, 0
        for i, (rec, vec) in enumerate(zip(recs, vectors)):
            n = 20 * len(vec) + len(rec.text.encode("utf-8")) + 64 * (len(rec.metadata) + 1)
            if i > start and size + n > self._UPSERT_MAX_BYTES:
                yield recs[start:i], vectors[start:i]
                start, size = i, 0
            size += n
        if start < len(recs):
            yield recs[start:], vectors[start:]

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

        Pass *ids* to delete specific records or *filter* to delete by
        metadata condition — one of them, non-empty. Both are refused, since
        the backends disagree on what both would mean, and neither is
        refused rather than read as "everything".
        """
        _one_of(ids, filter)
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

    def _check_records(self, raw: list[dict]) -> None:
        for n, r in enumerate(raw):
            if not isinstance(r, dict):
                raise TypeError(f"record {n} is a {type(r).__name__}, not a dict with id and text")
            rid = r.get("id")
            if isinstance(rid, bool) or not isinstance(rid, (str, int)) or str(rid) == "":
                raise ValueError(f"record {n} has no id; every record needs a string or integer id")
            if not isinstance(r.get("text"), str):
                raise ValueError(f"record {rid!r} has no text")
            if not isinstance(r.get("metadata") or {}, dict):
                raise TypeError(f"record {rid!r}: metadata is a dict, not "
                                f"{type(r.get('metadata')).__name__}")
        for r in raw:
            clash = sorted(set(r.get("metadata") or {}) & set(RESERVED))
            if clash:
                raise ValueError(f"record {r['id']!r}: metadata may not use {clash} — the store "
                                 "keeps the record's text and id under those names")
        if self._embedder is None:
            for r in raw:
                if "vector" not in r:
                    raise ValueError(
                        f"Record {r['id']!r} has no 'vector' and no embedder is set.  "
                        f"Pass embedder= to VectorDB() or include 'vector' in every record.")

    def _prepare_records(
        self,
        raw: list[dict],
    ) -> tuple[list[VectorRecord], list[list[float]]]:
        """``(records, vectors)`` from raw dicts; those without a ``vector`` are
        embedded in one batched call."""
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
            clash = sorted(set(r.get("metadata") or {}) & set(RESERVED))
            if clash:
                raise ValueError(f"record {r['id']!r}: metadata may not use {clash} — the store "
                                 "keeps the record's text and id under those names")
            rec = VectorRecord(
                id       = str(r["id"]),
                text     = r["text"],
                metadata = r.get("metadata") or {},
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
