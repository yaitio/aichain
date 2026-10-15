"""
``RerankResult``, the return type of every reranking call, and ``RerankBase``,
the base class of every provider.

A provider sets ``provider`` and ``_ENV_KEY`` and implements
``_rerank_request``; the base normalises the documents, merges a record's own
fields back into its result and provides ``run()``. See docs/tools/retrieval.md.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from yait_aichain.clients._base import make_http

from ...clients._constants import TOOL_TIMEOUT, DEFAULT_RETRIES
from ...clients._http import request_json
from .._base import Tool, text_of


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

class RerankBase(Tool):
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

    # A tool: code, a chain and an agent call it alike.
    risk        = "read"
    priced      = True
    _safe_for_untrusted_load = True      # see Tool._safe_for_untrusted_load
    parameters  = {
        "type": "object",
        "properties": {
            "input":   {"type": "array", "items": {"type": "string"},
                        "description": "The documents to put in order — texts, or records with a 'text' field."},
            "options": {"type": "object", "description": "What to order them by.", "properties": {
                "query": {"type": "string", "description": "The question the documents should answer."},
                "top_n": {"type": "integer", "description": "How many to return, best first (default: all)."},
            }, "required": ["query"]},
        },
        "required": ["input", "options"],
    }

    # ── Construction ──────────────────────────────────────────────────────────

    def __init__(
        self,
        model:   str,
        *,
        api_key: str | None = None,
        name:        str | None = None,
        description: str | None = None,
        **kwargs: Any,
    ) -> None:
        if kwargs:
            # A call's option given here would be dropped, and every document
            # returned in silence.
            hint = (" — top_n and query are options of the call: .run(docs, {\"query\": …, \"top_n\": …})"
                    if set(kwargs) & {"top_n", "query"} else "")
            raise TypeError(f"{type(self).__name__} takes model, api_key, name and description; "
                            f"got {', '.join(sorted(kwargs))}{hint}")
        key = api_key or os.environ.get(self._ENV_KEY, "")
        if not key:
            raise ValueError(
                f"No {self.provider} API key found.  "
                f"Pass api_key= or set the {self._ENV_KEY!r} environment variable."
            )
        self.model    = model
        self._api_key = key
        label = type(self).__name__.replace("Rerank", "", 1) or self.provider
        self.name = name or f"rerank{label}"
        self.description = description or (
            f"Put documents in order of how well they answer a query, with {label}'s {model}; "
            "returns them best first, each with its score.")
        self._http    = make_http(
            timeout = TOOL_TIMEOUT,
            retries = DEFAULT_RETRIES,
        )

    def _serialise_init_args(self) -> dict:
        """What `Chain.save` writes: the model, never the key."""
        out = {"model": self.model}
        if getattr(self, "_region", None):
            out["region"] = self._region
        return out

    # ── Public API ────────────────────────────────────────────────────────────

    def rerank(
        self,
        query:     str,
        documents: list[str | dict],
        *,
        top_n: int | None = None,
    ) -> RerankResult:
        """Rerank *documents* by relevance to *query*.

        Parameters
        ----------
        query : str
            The search query.
        documents : list[str | dict]
            Strings, or dicts with a ``"text"`` key (``Vectors(...).query`` output).
        top_n : int | None, optional
            Return only the top *N* results; ``None`` returns all.

        Returns
        -------
        RerankResult
            Results sorted by descending relevance score.
        """
        if isinstance(query, bool) or not isinstance(query, str) or not query.strip():
            raise ValueError(f"{type(self).__name__}: rerank needs a query — options={{'query': …}}")
        if isinstance(documents, (str, bytes)) or isinstance(documents, Mapping):
            # A string is a sequence of characters: one document per character.
            raise TypeError(f"{type(self).__name__}: documents is a list of documents, "
                            f"not one {type(documents).__name__}")
        if top_n is not None:
            if isinstance(top_n, bool) or not isinstance(top_n, (int, str)) \
                    or not str(top_n).strip().isdigit() or int(top_n) < 1:
                raise ValueError(f"{type(self).__name__}: top_n is a whole number of at least 1, "
                                 f"not {top_n!r}")
            top_n = int(top_n)
        documents = list(documents)
        if not documents:
            return RerankResult(results=[], model=self.model, provider=self.provider,
                                total_tokens=None, metadata={})
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
        """Rerank *input* by relevance to ``options["query"]``.

        Parameters
        ----------
        input : list[str | dict]
            Documents to rerank (same format as :meth:`rerank`).
        options : dict | None
            ``query`` (required) and ``top_n`` (default: all).

        Returns
        -------
        list[dict]
            The ranked results, as :attr:`RerankResult.results`.
        """
        opts  = options or {}
        query = opts.get("query")
        top_n = opts.get("top_n")
        return self.rerank(query, input, top_n=top_n).results

    # ── Abstract method ───────────────────────────────────────────────────────

    def _rerank_request(
        self,
        query:     str,
        documents: list[str],
        top_n:     int | None,
    ) -> tuple[list[dict], int | None, dict]:
        """One request over plain-text documents. Returns ``(results, total_tokens or
        None, metadata)``, results ``{index, text, score}`` by descending score."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement _rerank_request()"
        )

    # ── Shared helpers ────────────────────────────────────────────────────────

    def _normalise(
        self,
        documents: list[str | dict],
    ) -> tuple[list[str], list[dict | None]]:
        """The text of each document, and its record (``None`` for a string) to
        merge back into the result."""
        texts:     list[str]        = []
        originals: list[dict | None] = []
        for doc in documents:
            texts.append(text_of(doc, type(self).__name__))
            if isinstance(doc, Mapping):
                originals.append(dict(doc))
            elif callable(getattr(doc, "to_dict", None)):
                originals.append(doc.to_dict())
            else:
                originals.append(None)
        return texts, originals

    @staticmethod
    def _ranked(items: list, documents: list) -> list[dict]:
        """``{index, text, score}`` by descending score, with the text that was sent:
        every provider answers with an index and a relevance_score, so none is asked
        to send the documents back."""
        out = [{"index": i["index"], "text": documents[i["index"]], "score": i["relevance_score"]}
               for i in items]
        return sorted(out, key=lambda x: x["score"], reverse=True)

    def _post(self, url: str, body: dict, extra_headers: dict | None = None) -> dict:
        """POST *body* as JSON to *url* and return the parsed response."""
        return request_json(self._http, "POST", url, json=body, what=f"{self.provider} Rerank API",
                            headers={"Accept": "application/json",
                                     "Authorization": f"Bearer {self._api_key}", **(extra_headers or {})})

    # ── Dunder helpers ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r}, provider={self.provider!r})"
