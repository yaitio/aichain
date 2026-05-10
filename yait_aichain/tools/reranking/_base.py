"""
tools.reranking._base
======================

RerankResult  — uniform return type for every reranking call.
RerankBase    — abstract base class for all provider implementations.

Design
------
Every concrete reranker:
  1. Sets ``provider`` and ``_ENV_KEY`` class attributes.
  2. Calls ``super().__init__(model, api_key=api_key)`` in ``__init__``.
  3. Implements ``_rerank_request(query, documents, top_n)``
     which sends one API request and returns
     ``(results, total_tokens, metadata)`` where ``results`` is a list of
     ``{"index": int, "text": str, "score": float}`` dicts sorted by
     descending score.

Everything above that layer — document normalisation, run() interface,
result assembly — is provided by this base class.

Document normalisation
----------------------
All rerankers accept documents as:
  • ``list[str]``       — plain text strings
  • ``list[dict]``      — dicts with a ``"text"`` key (direct output from
                          :class:`~tools.vectordb.VectorQueryTool`)

The original dict fields (``id``, ``metadata``, etc.) are preserved and
merged back into the result so the caller never loses context.

run() interface
---------------
``run(input, options)`` follows the standard Tool contract:
  ``input``   : list[str | dict]  — documents to rerank
  ``options`` : dict
      ``query``  str  — the search query  (required)
      ``top_n``  int  — how many results to return (default: all)

This makes the reranker a drop-in Chain step after
:class:`~tools.vectordb.VectorQueryTool`:

::

    (vectorQuery(store), "candidates", {"input": "{question}"}),
    (reranker,           "top5",       {"input": "{candidates}",
                                        "options": {"query": "{question}", "top_n": 5}}),
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import urllib3

from ...clients._constants import DEFAULT_TIMEOUT, DEFAULT_RETRIES


# ---------------------------------------------------------------------------
# RerankResult
# ---------------------------------------------------------------------------

@dataclass
class RerankResult:
    """
    Uniform return type for every :class:`RerankBase` call.

    Attributes
    ----------
    results : list[dict]
        Reranked documents, sorted by descending relevance score.
        Each item:
          ``index``    int   — position in the original input list
          ``text``     str   — document text
          ``score``    float — relevance score (higher = more relevant)
          All original dict fields from the input are preserved when
          documents were passed as dicts.
    model        : str
    provider     : str
    total_tokens : int | None
    metadata     : dict
        Provider-specific extras (e.g. Cohere billed units).

    Examples
    --------
    ::

        result = reranker.rerank("how does caching work?", documents)
        for item in result:
            print(item["score"], item["text"][:60])

        texts = result.texts()    # list[str], top to bottom
        scores = result.scores()  # list[float]
    """

    results:      list[dict]
    model:        str
    provider:     str
    total_tokens: int | None = None
    metadata:     dict       = field(default_factory=dict)

    def texts(self) -> list[str]:
        """Return document texts in ranked order."""
        return [r["text"] for r in self.results]

    def scores(self) -> list[float]:
        """Return relevance scores in ranked order."""
        return [r["score"] for r in self.results]

    def __len__(self) -> int:
        return len(self.results)

    def __getitem__(self, idx: int) -> dict:
        return self.results[idx]

    def __iter__(self):
        return iter(self.results)

    def __repr__(self) -> str:
        top = self.results[0]["score"] if self.results else None
        return (
            f"RerankResult(n={len(self.results)}, top_score={top}, "
            f"model={self.model!r}, provider={self.provider!r})"
        )


# ---------------------------------------------------------------------------
# RerankBase — abstract
# ---------------------------------------------------------------------------

class RerankBase:
    """
    Abstract base class for all reranking provider implementations.

    Subclasses must:
      1. Set ``provider`` and ``_ENV_KEY``.
      2. Implement ``_rerank_request(query, documents, top_n)``
         returning ``(results, total_tokens, metadata)``.

    Public methods
    --------------
    :meth:`rerank`  — core method; returns :class:`RerankResult`.
    :meth:`run`     — standard ``run(input, options)`` Tool interface.
    """

    provider: str = ""
    _ENV_KEY: str = ""

    # ── Construction ──────────────────────────────────────────────────────────

    def __init__(
        self,
        model:   str,
        *,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        key = api_key or os.environ.get(self._ENV_KEY, "")
        if not key:
            raise ValueError(
                f"No {self.provider} API key found.  "
                f"Pass api_key= or set the {self._ENV_KEY!r} environment variable."
            )
        self.model    = model
        self._api_key = key
        self._http    = urllib3.PoolManager(
            timeout = DEFAULT_TIMEOUT,
            retries = DEFAULT_RETRIES,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def rerank(
        self,
        query:     str,
        documents: list[str | dict],
        *,
        top_n: int | None = None,
    ) -> RerankResult:
        """
        Rerank *documents* by relevance to *query*.

        Parameters
        ----------
        query : str
            The search query.
        documents : list[str | dict]
            Documents to rerank.  Accepts plain strings or dicts with a
            ``"text"`` key (direct output from :class:`~tools.vectordb.VectorQueryTool`).
        top_n : int | None, optional
            Return only the top *N* results.  ``None`` returns all.

        Returns
        -------
        RerankResult
            Results sorted by descending relevance score.
        """
        texts, originals = self._normalise(documents)
        raw, total_tokens, metadata = self._rerank_request(query, texts, top_n)

        # Merge original dict fields back into each result
        results: list[dict] = []
        for item in raw:
            idx    = item["index"]
            merged = {}
            if originals[idx] is not None:
                merged = {k: v for k, v in originals[idx].items() if k != "text"}
            merged.update({
                "index": idx,
                "text":  item["text"],
                "score": item["score"],
            })
            results.append(merged)

        return RerankResult(
            results      = results,
            model        = self.model,
            provider     = self.provider,
            total_tokens = total_tokens,
            metadata     = metadata,
        )

    def run(
        self,
        input:   list[str | dict],
        options: dict | None = None,
    ) -> list[dict]:
        """
        Standard Tool interface for use in Chains and Agents.

        Parameters
        ----------
        input : list[str | dict]
            Documents to rerank (same format as :meth:`rerank`).
        options : dict | None
            ``query``  str — the search query (required)
            ``top_n``  int — number of results to return (default: all)

        Returns
        -------
        list[dict]
            JSON-serialisable ranked results (same as
            :attr:`RerankResult.results`).
        """
        opts  = options or {}
        query = opts.get("query", "")
        top_n = opts.get("top_n")
        return self.rerank(query, input, top_n=top_n).results

    # ── Abstract method ───────────────────────────────────────────────────────

    def _rerank_request(
        self,
        query:     str,
        documents: list[str],
        top_n:     int | None,
    ) -> tuple[list[dict], int | None, dict]:
        """
        Send a single reranking API request.

        Must be implemented by every subclass.

        Parameters
        ----------
        query     : str         — the query string
        documents : list[str]   — plain-text documents (already normalised)
        top_n     : int | None  — max results, or None for all

        Returns
        -------
        tuple[list[dict], int | None, dict]
          * results      — ``[{"index": int, "text": str, "score": float}, ...]``
                           sorted by descending score
          * total_tokens — token count, or None
          * metadata     — provider extras dict
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _rerank_request()"
        )

    # ── Shared helpers ────────────────────────────────────────────────────────

    def _normalise(
        self,
        documents: list[str | dict],
    ) -> tuple[list[str], list[dict | None]]:
        """
        Extract plain text strings from *documents*.

        Returns
        -------
        texts     : list[str]          — text for each document
        originals : list[dict | None]  — original dict (or None for str inputs)
                    used to merge fields back after reranking
        """
        texts:     list[str]        = []
        originals: list[dict | None] = []
        for doc in documents:
            if isinstance(doc, str):
                texts.append(doc)
                originals.append(None)
            elif isinstance(doc, dict):
                texts.append(doc.get("text", str(doc)))
                originals.append(doc)
            else:
                texts.append(str(doc))
                originals.append(None)
        return texts, originals

    def _post(self, url: str, body: dict, extra_headers: dict | None = None) -> dict:
        """POST *body* as JSON to *url* and return the parsed response."""
        headers = {
            "Content-Type":  "application/json",
            "Accept":        "application/json",
            "Authorization": f"Bearer {self._api_key}",
            **(extra_headers or {}),
        }
        response = self._http.request(
            "POST",
            url,
            body    = json.dumps(body).encode("utf-8"),
            headers = headers,
        )
        raw = response.data.decode("utf-8", errors="replace")
        if not (200 <= response.status < 300):
            raise RuntimeError(
                f"{self.provider} Rerank API error [{response.status}]: {raw[:500]}"
            )
        return json.loads(raw)

    # ── Dunder helpers ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r}, provider={self.provider!r})"
