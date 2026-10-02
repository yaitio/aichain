"""
tools.search.exa — searchExa
============================

Wraps the Exa Search API (``POST /search``) behind the standard
:class:`~tools.search.Search` interface.

Endpoint  : POST https://api.exa.ai/search
Auth      : ``x-api-key`` header
Docs      : https://exa.ai/docs/reference/search — the OpenAPI spec there is the
            source of truth; read 2026-10-02 and checked live the same day.

Exa is a search engine built for agents: the same request that finds the pages
can return query-relevant **highlights** from each one, so the agent reads the
part that answers it without fetching the page. Highlights are on by default
here; ``text_chars`` adds the page text itself, cut to a length.

The result is the same numbered plain text the other search tools return, so
an orchestrating model reads titles and URLs directly::

    Search results for "KV cache explained" (2 results):

    [1] Understanding and Coding the KV Cache in LLMs from Scratch
        URL: https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms
        Published: 2025-06-17 · Author: Sebastian Raschka, PhD
        · KV caches are one of the most critical techniques for efficient …

Of the standard search options, ``language`` has no field on this API and is
not declared, so asking for it is reported as an unknown option rather than
dropped. ``recency`` becomes a published-after date.

Environment variable
--------------------
``EXA_API_KEY`` — from https://dashboard.exa.ai/api-keys
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

from yait_aichain.clients._base import make_http

from ._base import Search, SEARCH_RETRIES, SEARCH_TIMEOUT

_ENDPOINT = "https://api.exa.ai/search"
_ENV_KEY  = "EXA_API_KEY"

#: The standard ``recency`` words, as the window Exa's ``startPublishedDate``
#: is set back by. Exa filters on a date, not on a word.
_RECENCY = {"hour": timedelta(hours=1), "day": timedelta(days=1),
            "week": timedelta(weeks=1), "month": timedelta(days=31),
            "year": timedelta(days=366)}

#: The API's own documented ceiling on ``numResults``.
_MAX_RESULTS = 100


class searchExa(Search):
    """
    Web search powered by the Exa Search API — results with the passages that
    answer the query.

    Parameters
    ----------
    api_key : str | None, optional
        Exa API key. When omitted, ``EXA_API_KEY`` is used.

    Raises
    ------
    ValueError
        At construction, when no key is found.

    Examples
    --------
    ::

        from yait_aichain.tools.search import searchExa

        tool   = searchExa()
        result = tool(input="KV cache in LLM inference explained",
                      options={"max_results": 5, "recency": "year"})
        print(result.output if result else result.error)

        # News only, from two domains, with up to 2 000 characters of each page:
        text = tool.run(input="EU AI Act enforcement", options={
            "category": "news", "domains": ["reuters.com", "ft.com"],
            "text_chars": 2000,
        })
    """

    name        = "searchExa"
    description = (
        "Search the web with Exa, a search engine built for agents. Returns "
        "titles, URLs, dates and the passages of each page most relevant to "
        "the query."
    )
    parameters  = {
        "type": "object",
        "properties": {
            "input": {
                "type":        "string",
                "description": "The search query, in natural language.",
            },
            "options": {
                "type":        "object",
                "description": "Optional search filters and configuration.",
                "properties": {
                    "max_results": {
                        "type":        "integer",
                        "description": f"Number of results to return (1–{_MAX_RESULTS}, default 10).",
                    },
                    "recency": {
                        "type":        "string",
                        "enum":        list(_RECENCY),
                        "description": "Only pages published within this window.",
                    },
                    "country": {
                        "type":        "string",
                        "description": "Two-letter ISO country code of the user, e.g. 'US'.",
                    },
                    "domains": {
                        "type":        "array",
                        "items":       {"type": "string"},
                        "description": "Restrict results to these domains (or domain/path prefixes).",
                    },
                    "exclude_domains": {
                        "type":        "array",
                        "items":       {"type": "string"},
                        "description": "Leave out results from these domains.",
                    },
                    "category": {
                        "type":        "string",
                        "enum":        ["company", "publication", "news", "personal site",
                                        "financial report", "people"],
                        "description": "Focus on one kind of page.",
                    },
                    "search_type": {
                        "type":        "string",
                        "enum":        ["instant", "fast", "auto", "deep-lite", "deep",
                                        "deep-reasoning"],
                        "description": ("'auto' (default) balances quality and speed; "
                                        "'instant'/'fast' trade quality for latency; "
                                        "the 'deep' modes search iteratively and cost more."),
                    },
                    "start_date": {
                        "type":        "string",
                        "description": "Only pages published after this ISO 8601 date.",
                    },
                    "end_date": {
                        "type":        "string",
                        "description": "Only pages published before this ISO 8601 date.",
                    },
                    "text_chars": {
                        "type":        "integer",
                        "description": ("Also return each page's text, cut to this many "
                                        "characters. Off by default — highlights usually "
                                        "carry the answer."),
                    },
                },
            },
        },
        "required": ["input"],
    }

    def __init__(self, api_key: "str | None" = None) -> None:
        key = api_key or os.environ.get(_ENV_KEY)
        if not key:
            raise ValueError(
                f"No Exa API key found. Pass api_key= or set the {_ENV_KEY} "
                "environment variable."
            )
        self._api_key = key
        self._http    = make_http(timeout=SEARCH_TIMEOUT, retries=SEARCH_RETRIES)

    # ------------------------------------------------------------------
    # Request
    # ------------------------------------------------------------------

    @staticmethod
    def _body(query: str, opts: dict, now: "datetime | None" = None) -> dict:
        """The request body for *query* — kept apart so it can be checked
        without a network."""
        count = opts.get("max_results", 10)
        body: dict = {
            "query":      query,
            "type":       opts.get("search_type", "auto"),
            "numResults": max(1, min(int(count), _MAX_RESULTS)),
            "contents":   {"highlights": True},
        }
        if opts.get("text_chars"):
            body["contents"]["text"] = {"maxCharacters": int(opts["text_chars"])}
        if opts.get("domains"):
            body["includeDomains"] = list(opts["domains"])
        if opts.get("exclude_domains"):
            body["excludeDomains"] = list(opts["exclude_domains"])
        if opts.get("category"):
            body["category"] = opts["category"]
        if opts.get("country"):
            body["userLocation"] = opts["country"]

        start = opts.get("start_date")
        if opts.get("recency"):
            window = _RECENCY.get(opts["recency"])
            if window is None:
                raise ValueError(f"recency must be one of {', '.join(_RECENCY)}; "
                                 f"got {opts['recency']!r}")
            since = (now or datetime.now(timezone.utc)) - window
            # An explicit start date and a recency window both bound the same
            # side; the later of the two is the one both promises keep.
            since_iso = since.strftime("%Y-%m-%dT%H:%M:%S.000Z")
            start = max(start, since_iso) if start else since_iso
        if start:
            body["startPublishedDate"] = start
        if opts.get("end_date"):
            body["endPublishedDate"] = opts["end_date"]
        return body

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def run(self, input: str, options: "dict | None" = None) -> str:
        """
        Search Exa for *input* and return a numbered plain-text list of results
        with their highlights.

        Raises ``RuntimeError`` on any non-2xx answer, with the status and the
        API's own message.
        """
        body = self._body(input, options or {})
        response = self._http.request(
            "POST", _ENDPOINT,
            body=json.dumps(body).encode("utf-8"),
            headers={"x-api-key": self._api_key,
                     "Content-Type": "application/json",
                     "Accept": "application/json"},
        )
        raw = response.data.decode("utf-8", errors="replace")
        if not 200 <= response.status < 300:
            raise RuntimeError(f"Exa Search API error [{response.status}]: {raw}")
        return self._format(input, json.loads(raw))

    @staticmethod
    def _format(query: str, data: dict) -> str:
        results = data.get("results") or []
        lines = [f'Search results for "{query}" ({len(results)} results):\n']
        for i, r in enumerate(results, 1):
            lines.append(f"[{i}] {r.get('title') or '(untitled)'}")
            lines.append(f"    URL: {r.get('url', '')}")
            meta = []
            if r.get("publishedDate"):
                meta.append(f"Published: {r['publishedDate'][:10]}")
            if r.get("author"):
                meta.append(f"Author: {r['author']}")
            if meta:
                lines.append("    " + " · ".join(meta))
            for h in r.get("highlights") or []:
                lines.append("    · " + " ".join(str(h).split()))
            if r.get("text"):
                lines.append("    Text: " + " ".join(str(r["text"]).split()))
            lines.append("")
        return "\n".join(lines).rstrip()
