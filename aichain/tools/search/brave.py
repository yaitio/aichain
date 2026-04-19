"""
tools.search.brave — searchBrave
=================================

Wraps the Brave Search API (POST /res/v1/web/search) behind the standard
:class:`~tools.search.Search` interface.

Endpoint  : POST https://api.search.brave.com/res/v1/web/search
Auth      : ``x-subscription-token`` header
Docs      : https://api-dashboard.search.brave.com/api-reference/web/search/post

The tool returns a human-readable plain-text string so that an orchestrating
LLM can read page titles and **URLs directly** from the context variable
preview without needing JSON parsing or variable injection::

    Search results for "query text" (5 results):

    [1] Page Title
        URL: https://example.com/page
        Summary: Brief excerpt describing the page content.

    [2] Another Title
        URL: https://another.com/article
        Summary: …

This format makes it straightforward for the agent to extract a specific
URL and pass it as a literal string to MarkItDownTool.

Environment variable
---------------------
``BRAVE_SEARCH_API_KEY`` — API key obtained from
https://api-dashboard.search.brave.com
"""

from __future__ import annotations

import json
import os

import urllib3

from ._base import Search

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
_ENV_KEY   = "BRAVE_SEARCH_API_KEY"


# ---------------------------------------------------------------------------
# searchBrave
# ---------------------------------------------------------------------------

class searchBrave(Search):
    """
    Web search powered by the Brave Search API.

    Parameters
    ----------
    api_key : str | None, optional
        Brave Search subscription token.  When omitted the value of the
        ``BRAVE_SEARCH_API_KEY`` environment variable is used.

    Raises
    ------
    ValueError
        At construction time if no API key is found in either the argument
        or the environment.

    Examples
    --------
    Basic search::

        from tools.search import searchBrave

        tool   = searchBrave()
        result = tool(input="Python asyncio tutorial",
                      options={"max_results": 5})

        if result:
            print(result.output)   # plain-text numbered list with URLs
        else:
            print("Error:", result.error)

    Direct call with all options::

        text = tool.run(
            input   = "climate change solutions",
            options = {
                "max_results":    10,
                "country":        "GB",
                "language":       "en",
                "safe_search":    "moderate",
                "freshness":      "pw",
                "extra_snippets": True,
            },
        )
        print(text)
    """

    name        = "searchBrave"
    description = (
        "Search the web using the Brave Search engine. "
        "Returns titles, URLs, and description snippets."
    )
    parameters  = {
        "type": "object",
        "properties": {
            "input": {
                "type":        "string",
                "description": "The search query (required, 1–400 characters).",
            },
            "options": {
                "type":        "object",
                "description": "Optional search filters and configuration.",
                "properties": {
                    "max_results": {
                        "type":        "integer",
                        "description": "Number of results to return (1–20, default 10).",
                    },
                    "country": {
                        "type":        "string",
                        "description": (
                            "Two-letter ISO 3166-1 country code that biases results "
                            "(e.g. 'US', 'GB', 'DE').  Defaults to 'US'."
                        ),
                    },
                    "language": {
                        "type":        "string",
                        "description": (
                            "Language code for results (e.g. 'en', 'de', 'fr').  "
                            "Defaults to 'en'."
                        ),
                    },
                    "safe_search": {
                        "type":        "string",
                        "enum":        ["off", "moderate", "strict"],
                        "description": "Adult-content filter level.  Defaults to 'moderate'.",
                    },
                    "freshness": {
                        "type":        "string",
                        "description": (
                            "Filter results by page age.  "
                            "Use 'pd' (past day), 'pw' (past week), 'pm' (past month), "
                            "'py' (past year), or a date range 'YYYY-MM-DDtoYYYY-MM-DD'."
                        ),
                    },
                    "extra_snippets": {
                        "type":        "boolean",
                        "description": (
                            "When True, up to 5 additional excerpt snippets are "
                            "included per result.  Defaults to False."
                        ),
                    },
                    "result_filter": {
                        "type":        "string",
                        "description": (
                            "Comma-separated list of result types to include.  "
                            "Allowed values: discussions, faq, infobox, news, query, "
                            "summarizer, videos, web, locations.  "
                            "Defaults to all types."
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
                f"No Brave Search API key found.  Pass api_key= or set the "
                f"{_ENV_KEY} environment variable."
            )
        self._api_key = key
        self._http    = urllib3.PoolManager()

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def run(self, input: str, options: "dict | None" = None) -> str:
        """
        Execute a Brave Search query and return a formatted plain-text string.

        The result is formatted so that an LLM orchestrator can read page
        titles and URLs directly from the text without any JSON parsing::

            Search results for "query text" (5 results):

            [1] Page Title
                URL: https://example.com/page
                Summary: Brief excerpt…

        Parameters
        ----------
        input : str
            Search query (required).
        options : dict | None
            Optional configuration keys:
              ``max_results``    — number of results (1–20, default 10).
              ``country``        — ISO 3166-1 alpha-2 country code.
              ``language``       — language code for results (e.g. ``"en"``).
              ``safe_search``    — ``"off"``, ``"moderate"``, or ``"strict"``.
              ``freshness``      — ``"pd"`` / ``"pw"`` / ``"pm"`` / ``"py"`` or
                                   ``"YYYY-MM-DDtoYYYY-MM-DD"``.
              ``extra_snippets`` — include up to 5 extra snippet excerpts per result.
              ``result_filter``  — comma-separated result type names to include.

        Returns
        -------
        str
            Human-readable search results with numbered entries, titles,
            URLs, and description snippets.

        Raises
        ------
        RuntimeError
            On any non-2xx HTTP response from the Brave API.
        """
        opts = options or {}

        count          = opts.get("max_results", 10)
        country        = opts.get("country")
        search_lang    = opts.get("language")
        safesearch     = opts.get("safe_search")
        freshness      = opts.get("freshness")
        extra_snippets = opts.get("extra_snippets", False)
        result_filter  = opts.get("result_filter")

        # ── Build request body ────────────────────────────────────────
        body: dict = {"q": input, "count": count}

        if country        is not None:  body["country"]        = country
        if search_lang    is not None:  body["search_lang"]    = search_lang
        if safesearch     is not None:  body["safesearch"]     = safesearch
        if freshness      is not None:  body["freshness"]      = freshness
        if result_filter  is not None:  body["result_filter"]  = result_filter
        if extra_snippets:              body["extra_snippets"] = True

        headers = {
            "x-subscription-token": self._api_key,
            "Content-Type":         "application/json",
            "Accept":               "application/json",
        }

        # ── Send request ──────────────────────────────────────────────
        response = self._http.request(
            "POST",
            _ENDPOINT,
            body    = json.dumps(body).encode("utf-8"),
            headers = headers,
        )

        raw = response.data.decode("utf-8", errors="replace")

        if not (200 <= response.status < 300):
            raise RuntimeError(
                f"Brave Search API error [{response.status}]: {raw}"
            )

        data = json.loads(raw)

        # ── Parse results ─────────────────────────────────────────────
        web_results = (
            data.get("web", {}).get("results", [])
            or data.get("results", [])
        )

        results = []
        for item in web_results:
            entry: dict = {
                "title":       item.get("title", ""),
                "url":         item.get("url", ""),
                "description": item.get("description", ""),
            }
            if "page_age" in item:
                entry["age"] = item["page_age"]
            if extra_snippets and item.get("extra_snippets"):
                entry["extra_snippets"] = item["extra_snippets"]
            results.append(entry)

        # ── Format as readable text ───────────────────────────────────
        lines = [f'Search results for "{input}" ({len(results)} results):\n']
        for i, r in enumerate(results, 1):
            lines.append(f"[{i}] {r['title']}")
            lines.append(f"    URL: {r['url']}")
            if r["description"]:
                lines.append(f"    Summary: {r['description']}")
            if "age" in r:
                lines.append(f"    Age: {r['age']}")
            if "extra_snippets" in r:
                for snip in r["extra_snippets"]:
                    lines.append(f"    · {snip}")
            lines.append("")   # blank line between results

        return "\n".join(lines).rstrip()
