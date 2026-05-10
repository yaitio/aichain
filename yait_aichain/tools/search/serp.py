"""
tools.search.serp — searchSerp
================================

Wraps the SerpAPI (https://serpapi.com/search) behind the standard
:class:`~tools.search.Search` interface.

The key differentiator from other search tools: the ``engine`` option
lets callers choose from 50+ underlying search engines — Google, Bing,
Yahoo, DuckDuckGo, Baidu, Yandex, and many more — via a single tool.

Endpoint  : GET https://serpapi.com/search
Auth      : ``api_key`` query parameter
Docs      : https://serpapi.com/search-api

The tool returns a human-readable plain-text string so that an orchestrating
LLM can read page titles and **URLs directly** from the context variable
preview without needing JSON parsing or variable injection::

    Search results for "query text" via google (5 results):

    [1] Page Title
        URL: https://example.com/page
        Date: Mar 15, 2025
        Summary: Brief excerpt describing the page content.

    [2] Another Title
        URL: https://another.com/article
        Summary: …

Environment variable
---------------------
``SERPAPI_API_KEY`` — API key obtained from https://serpapi.com/manage-api-key
"""

from __future__ import annotations

import json
import os
import urllib.parse

import urllib3

from ._base import Search

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ENDPOINT = "https://serpapi.com/search"
_ENV_KEY  = "SERPAPI_API_KEY"

_SUPPORTED_ENGINES = [
    "google", "bing", "yahoo", "duckduckgo", "baidu", "yandex",
    "google_news", "bing_news", "google_images", "google_shopping",
    "google_maps", "google_jobs", "google_scholar",
]


# ---------------------------------------------------------------------------
# searchSerp
# ---------------------------------------------------------------------------

class searchSerp(Search):
    """
    Multi-engine web search powered by SerpAPI.

    The ``engine`` option selects which underlying search engine to use
    (Google, Bing, Yahoo, DuckDuckGo, Baidu, Yandex, and 50+ others).

    Parameters
    ----------
    api_key : str | None, optional
        SerpAPI key.  When omitted the value of the ``SERPAPI_API_KEY``
        environment variable is used.

    Raises
    ------
    ValueError
        At construction time if no API key is found in either the argument
        or the environment.

    Examples
    --------
    Google search (default engine)::

        from tools.search import searchSerp

        tool   = searchSerp()
        result = tool(input="Python asyncio tutorial",
                      options={"max_results": 5})

        if result:
            print(result.output)
        else:
            print("Error:", result.error)

    DuckDuckGo search::

        text = tool.run(
            input   = "climate change 2025",
            options = {"engine": "duckduckgo", "max_results": 10},
        )
        print(text)

    Baidu search in Chinese::

        text = tool.run(input="人工智能 2025", options={"engine": "baidu"})
        print(text)
    """

    name        = "searchSerp"
    description = (
        "Search the web via SerpAPI supporting multiple engines "
        "(Google, Bing, DuckDuckGo, Baidu, Yandex, etc.)."
    )
    parameters  = {
        "type": "object",
        "properties": {
            "input": {
                "type":        "string",
                "description": "The search query (required).",
            },
            "options": {
                "type":        "object",
                "description": "Optional search filters and configuration.",
                "properties": {
                    "engine": {
                        "type":        "string",
                        "description": (
                            "The search engine to use.  Common values: 'google' (default), "
                            "'bing', 'yahoo', 'duckduckgo', 'baidu', 'yandex', "
                            "'google_news', 'bing_news', 'google_images', "
                            "'google_shopping', 'google_scholar', 'google_maps', "
                            "'google_jobs'.  Defaults to 'google'."
                        ),
                    },
                    "max_results": {
                        "type":        "integer",
                        "description": "Number of results to return (default 10, max 100).",
                    },
                    "location": {
                        "type":        "string",
                        "description": (
                            "Geographic location to use for localised results "
                            "(e.g. 'Austin, Texas', 'Paris, France').  "
                            "Supported by Google, Bing, and Yahoo."
                        ),
                    },
                    "country": {
                        "type":        "string",
                        "description": (
                            "Two-letter ISO 3166-1 country code that biases results "
                            "(e.g. 'us', 'gb', 'de').  Google engine only."
                        ),
                    },
                    "language": {
                        "type":        "string",
                        "description": (
                            "Language code for the interface/results "
                            "(e.g. 'en', 'fr', 'de').  Google engine only."
                        ),
                    },
                    "safe": {
                        "type":        "string",
                        "enum":        ["active", "off"],
                        "description": (
                            "Safe search filter.  'active' enables filtering; "
                            "'off' disables it.  Google engine only."
                        ),
                    },
                    "time_filter": {
                        "type":        "string",
                        "description": (
                            "Advanced time-based search filter.  "
                            "Use 'qdr:d' (past day), 'qdr:w' (past week), "
                            "'qdr:m' (past month), 'qdr:y' (past year).  "
                            "Google engine only."
                        ),
                    },
                    "start": {
                        "type":        "integer",
                        "description": (
                            "Offset for pagination — the index of the first result "
                            "to return.  Default 0.  Google engine only."
                        ),
                    },
                    "no_cache": {
                        "type":        "boolean",
                        "description": (
                            "When True, forces a live search and bypasses SerpAPI's "
                            "cached results.  Useful for real-time data.  Default False."
                        ),
                    },
                },
            },
        },
        "required": ["input"],
    }

    # ------------------------------------------------------------------

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or os.environ.get(_ENV_KEY)
        if not key:
            raise ValueError(
                f"No SerpAPI key found.  Pass api_key= or set the "
                f"{_ENV_KEY} environment variable."
            )
        self._api_key = key
        self._http    = urllib3.PoolManager()

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def run(self, input: str, options: "dict | None" = None) -> str:
        """
        Execute a SerpAPI search and return a formatted plain-text string.

        The result is formatted so that an LLM orchestrator can read page
        titles and URLs directly from the text without any JSON parsing::

            Search results for "query text" via google (5 results):

            [1] Page Title
                URL: https://example.com/page
                Date: Mar 15, 2025
                Summary: Brief excerpt…

        Parameters
        ----------
        input : str
            Search query (required).
        options : dict | None
            Optional configuration keys:
              ``engine``      — search engine name (default ``"google"``).
              ``max_results`` — number of results (default 10).
              ``location``    — geographic location for localised results.
              ``country``     — two-letter country code → ``gl`` param (Google only).
              ``language``    — language code → ``hl`` param (Google only).
              ``safe``        — safe-search filter ``"active"`` / ``"off"`` (Google only).
              ``time_filter`` — time-based filter e.g. ``"qdr:w"`` → ``tbs`` (Google only).
              ``start``       — pagination offset (Google only).
              ``no_cache``    — skip SerpAPI cache and force a live search.

        Returns
        -------
        str
            Human-readable search results with numbered entries, titles,
            URLs, optional dates, and description snippets.

        Raises
        ------
        RuntimeError
            On any non-2xx HTTP response from the SerpAPI.
        """
        opts = options or {}

        engine   = opts.get("engine", "google")
        num      = opts.get("max_results", 10)
        location = opts.get("location")
        gl       = opts.get("country")
        hl       = opts.get("language")
        safe     = opts.get("safe")
        tbs      = opts.get("time_filter")
        start    = opts.get("start")
        no_cache = opts.get("no_cache", False)

        # ── Build query parameters ────────────────────────────────────
        params: dict = {
            "q":       input,
            "engine":  engine,
            "num":     num,
            "api_key": self._api_key,
        }

        if location is not None:  params["location"] = location
        if gl       is not None:  params["gl"]       = gl
        if hl       is not None:  params["hl"]       = hl
        if safe     is not None:  params["safe"]     = safe
        if tbs      is not None:  params["tbs"]      = tbs
        if start    is not None:  params["start"]    = start
        if no_cache:              params["no_cache"] = "true"

        url = f"{_ENDPOINT}?{urllib.parse.urlencode(params)}"

        # ── Send request ──────────────────────────────────────────────
        response = self._http.request("GET", url)

        raw = response.data.decode("utf-8", errors="replace")

        if not (200 <= response.status < 300):
            raise RuntimeError(
                f"SerpAPI error [{response.status}]: {raw[:400]}"
            )

        data = json.loads(raw)

        # ── Check for API-level errors ────────────────────────────────
        if "error" in data:
            raise RuntimeError(f"SerpAPI error: {data['error']}")

        # ── Parse organic results ─────────────────────────────────────
        organic = data.get("organic_results", [])

        # Some engines (Google News, Bing News) use a different key
        if not organic:
            organic = data.get("news_results", [])

        results = []
        for item in organic:
            entry: dict = {
                "title":   item.get("title", ""),
                "url":     item.get("link", ""),
                "snippet": item.get("snippet", ""),
            }
            if item.get("date"):
                entry["date"] = item["date"]
            if item.get("source"):
                entry["source"] = item["source"]
            results.append(entry)

        # ── Format as readable text ───────────────────────────────────
        lines = [f'Search results for "{input}" via {engine} ({len(results)} results):\n']
        for i, r in enumerate(results, 1):
            lines.append(f"[{i}] {r['title']}")
            lines.append(f"    URL: {r['url']}")
            if r.get("date"):
                lines.append(f"    Date: {r['date']}")
            if r.get("source"):
                lines.append(f"    Source: {r['source']}")
            if r["snippet"]:
                lines.append(f"    Summary: {r['snippet']}")
            lines.append("")   # blank line between results

        if not results:
            lines.append("(no results found)")

        return "\n".join(lines).rstrip()
