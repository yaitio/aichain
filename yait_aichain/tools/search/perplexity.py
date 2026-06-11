"""
tools.search.perplexity — searchPerplexity
==========================================

Wraps the Perplexity Search API (POST /search) behind the standard
:class:`~tools.search.Search` interface.

Endpoint  : POST https://api.perplexity.ai/search
Auth      : ``Authorization: Bearer <api_key>``
Docs      : https://docs.perplexity.ai/api-reference/search-post

Unlike BraveSearchTool (which returns ranked links), the Perplexity Search
API returns rich text **snippets** extracted directly from page content.
This makes it particularly useful when you want substantive excerpts without
a separate markitdown fetch step.

The tool returns a human-readable plain-text string::

    Search results for "query text" (5 results):

    [1] Page Title
        URL: https://example.com/page
        Date: 2025-11-03
        Snippet: Detailed excerpt from the page content covering the
                 specific topic you searched for…

    [2] Another Title
        URL: https://another.com/article
        Snippet: …

Environment variable
---------------------
``PERPLEXITY_API_KEY`` — API key obtained from https://www.perplexity.ai/settings/api
"""

from __future__ import annotations

import json
import os

import urllib3

from ._base import Search

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ENDPOINT = "https://api.perplexity.ai/search"
_ENV_KEY   = "PERPLEXITY_API_KEY"


# ---------------------------------------------------------------------------
# searchPerplexity
# ---------------------------------------------------------------------------

class searchPerplexity(Search):
    """
    Web search powered by the Perplexity Search API.

    Returns rich text snippets extracted directly from page content —
    useful when you want substantive excerpts without a separate URL-fetch
    step.

    Parameters
    ----------
    api_key : str | None, optional
        Perplexity API key (``pplx-...``).  When omitted the value of the
        ``PERPLEXITY_API_KEY`` environment variable is used.

    Raises
    ------
    ValueError
        At construction time if no API key is found in either the argument
        or the environment.

    Examples
    --------
    Basic search::

        from tools.search import searchPerplexity

        tool   = searchPerplexity()
        result = tool(input="nuclear fusion breakthroughs 2025",
                      options={"max_results": 5})

        if result:
            print(result.output)   # plain-text numbered list with snippets
        else:
            print("Error:", result.error)

    With date and domain filters::

        text = tool.run(
            input   = "GPT-5 benchmark results",
            options = {
                "max_results":  8,
                "recency":      "month",
                "domains":      ["openai.com", "arxiv.org"],
                "after_date":   "01/01/2025",
            },
        )
        print(text)
    """

    name        = "searchPerplexity"
    description = (
        "Search the web using the Perplexity Search API. "
        "Returns a numbered plain-text list with titles, URLs, and rich text "
        "snippets extracted directly from page content. "
        "Useful when you need substantive excerpts without a separate URL-fetch step."
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
                    "max_results": {
                        "type":        "integer",
                        "description": "Maximum number of results to return (1–20, default 10).",
                    },
                    "recency": {
                        "type":        "string",
                        "enum":        ["hour", "day", "week", "month", "year"],
                        "description": (
                            "Filter results by how recently they were published.  "
                            "Use 'day' for breaking news, 'week' for recent events, "
                            "'month' for the past month, 'year' for the past year."
                        ),
                    },
                    "domains": {
                        "type":        "array",
                        "items":       {"type": "string"},
                        "description": (
                            "Restrict results to specific domains "
                            "(e.g. [\"openai.com\", \"arxiv.org\"]).  "
                            "Up to 20 domains."
                        ),
                    },
                    "country": {
                        "type":        "string",
                        "description": (
                            "Two-letter ISO 3166-1 country code to localise results "
                            "(e.g. 'US', 'GB', 'DE')."
                        ),
                    },
                    "language": {
                        "type":        "string",
                        "description": (
                            "ISO 639-1 language code (or list of codes) for results "
                            "(e.g. 'en', 'fr').  Up to 20 languages."
                        ),
                    },
                    "after_date": {
                        "type":        "string",
                        "description": (
                            "Only return results published after this date.  "
                            "Format: MM/DD/YYYY  (e.g. '01/01/2025')."
                        ),
                    },
                    "before_date": {
                        "type":        "string",
                        "description": (
                            "Only return results published before this date.  "
                            "Format: MM/DD/YYYY  (e.g. '12/31/2025')."
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
                f"No Perplexity API key found.  Pass api_key= or set the "
                f"{_ENV_KEY} environment variable."
            )
        self._api_key = key
        self._http    = urllib3.PoolManager()

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def run(self, input: str, options: "dict | None" = None) -> str:
        """
        Execute a Perplexity Search query and return a formatted plain-text string.

        The result is formatted so that an LLM orchestrator can read titles,
        URLs, and substantive content snippets directly::

            Search results for "query text" (5 results):

            [1] Page Title
                URL: https://example.com/page
                Date: 2025-11-03
                Snippet: Detailed excerpt from the page…

        Parameters
        ----------
        input : str
            Search query (required).
        options : dict | None
            Optional configuration keys:
              ``max_results``  — number of results (1–20, default 10).
              ``recency``      — "hour" / "day" / "week" / "month" / "year".
              ``domains``      — list of domains to restrict results to.
              ``country``      — ISO 3166-1 alpha-2 country code.
              ``language``     — ISO 639-1 language code (str or list).
              ``after_date``   — only results after this date (MM/DD/YYYY).
              ``before_date``  — only results before this date (MM/DD/YYYY).

        Returns
        -------
        str
            Human-readable numbered list of results with titles, URLs, dates,
            and rich text snippets.

        Raises
        ------
        RuntimeError
            On any non-2xx HTTP response from the Perplexity API.
        """
        opts = options or {}

        max_results  = opts.get("max_results", 10)
        recency      = opts.get("recency")
        domains      = opts.get("domains")
        country      = opts.get("country")
        language     = opts.get("language")
        after_date   = opts.get("after_date")
        before_date  = opts.get("before_date")

        # ── Build request body ────────────────────────────────────────
        body: dict = {"query": input, "max_results": max_results}

        if recency    is not None:
            body["search_recency_filter"]     = recency
        if domains    is not None:
            body["search_domain_filter"]      = domains
        if country    is not None:
            body["country"]                   = country
        if language   is not None:
            # API expects a list; wrap a plain string automatically
            body["search_language_filter"]    = (
                language if isinstance(language, list) else [language]
            )
        if after_date  is not None:
            body["search_after_date_filter"]  = after_date
        if before_date is not None:
            body["search_before_date_filter"] = before_date

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type":  "application/json",
            "Accept":        "application/json",
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
                f"Perplexity Search API error [{response.status}]: {raw}"
            )

        data = json.loads(raw)

        # ── Parse results ─────────────────────────────────────────────
        results = data.get("results", [])

        # ── Format as readable text ───────────────────────────────────
        lines = [f'Search results for "{input}" ({len(results)} results):\n']
        for i, r in enumerate(results, 1):
            lines.append(f"[{i}] {r.get('title', '(no title)')}")
            lines.append(f"    URL: {r.get('url', '')}")
            if r.get("date"):
                lines.append(f"    Date: {r['date']}")
            if r.get("last_updated"):
                lines.append(f"    Updated: {r['last_updated']}")
            snippet = r.get("snippet", "")
            if snippet:
                wrapped = _wrap(snippet, width=100, indent="    ")
                lines.append(f"    Snippet: {wrapped}")
            lines.append("")   # blank line between results

        return "\n".join(lines).rstrip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wrap(text: str, width: int = 100, indent: str = "    ") -> str:
    """
    Soft-wrap *text* at *width* characters, indenting continuation lines.
    Returns the wrapped string (first line is NOT indented — the caller
    writes the label prefix).
    """
    words   = text.split()
    lines   = []
    current = []
    length  = 0

    for word in words:
        if length + len(word) + (1 if current else 0) > width:
            lines.append(" ".join(current))
            current = [word]
            length  = len(word)
        else:
            current.append(word)
            length += len(word) + (1 if len(current) > 1 else 0)

    if current:
        lines.append(" ".join(current))

    return ("\n" + indent + "         ").join(lines)
