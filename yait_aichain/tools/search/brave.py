"""
tools.search.brave — searchBrave
================================

The Brave Search API behind the common :class:`~tools.search.Search` format.

Endpoint  : POST https://api.search.brave.com/res/v1/web/search
Auth      : ``x-subscription-token`` header
Docs      : https://api-dashboard.search.brave.com/api-reference/web/search/post

Translated: ``recency`` → ``freshness`` (``pd``/``pw``/``pm``/``py``; Brave has
no hour, so ``hour`` is adapted to a day and said so), dates → a
``YYYY-MM-DDtoYYYY-MM-DD`` range, ``domains`` → ``site:`` in the query (Brave
has no site field), ``language`` → ``search_lang``. At most 20 results.

Brave's own options stay: ``safe_search``, ``extra_snippets`` (up to five more
passages per result), ``result_filter``, and ``freshness`` itself, which wins
over the standard time options when given.

Environment variable
--------------------
``BRAVE_SEARCH_API_KEY`` — from https://api-dashboard.search.brave.com
"""

from __future__ import annotations

import html
import json
import re
from datetime import date

from ._base import Search

_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
_FRESHNESS = {"hour": "pd", "day": "pd", "week": "pw", "month": "pm", "year": "py"}


def site_query(query: str, domains) -> str:
    """*query* restricted to *domains* with ``site:`` — for providers with no site field."""
    if not domains:
        return query
    sites = " OR ".join(f"site:{d}" for d in domains)
    return f"{query} ({sites})" if len(domains) > 1 else f"{query} {sites}"


class searchBrave(Search):
    """
    Web search powered by the Brave Search API — titles, URLs, the page's
    description and, with ``extra_snippets``, more passages from it.

    Examples
    --------
    ::

        from yait_aichain.tools import searchBrave

        tool = searchBrave()
        tool.run(input="Python asyncio tutorial", options={"max_results": 5, "recency": "month"})
        tool.run(input="AI agents", options={"domains": ["hbr.org"]})["results"]
    """

    name        = "searchBrave"
    description = ("Search the web with Brave. Returns titles, URLs, dates and a "
                   "description of each page.")
    max_results = 20
    _ENV_KEY    = "BRAVE_SEARCH_API_KEY"
    _PROVIDER   = "Brave Search"
    _LIST_PRICE   = 0.005
    _PRICE_SOURCE = "brave.com/search/api, $5 / 1 000 requests, read 2026-10-08"
    extra_options = {
        "safe_search":    {"type": "string", "enum": ["off", "moderate", "strict"],
                           "description": "Adult-content filter. Default 'moderate'."},
        "extra_snippets": {"type": "boolean",
                           "description": "Up to five more passages from each page. Default false."},
        "result_filter":  {"type": "string",
                           "description": "Comma-separated result types: discussions, faq, infobox, "
                                          "news, query, summarizer, videos, web, locations."},
        "freshness":      {"type": "string",
                           "description": "Brave's own time filter — 'pd', 'pw', 'pm', 'py' or "
                                          "'YYYY-MM-DDtoYYYY-MM-DD'; wins over recency and the dates."},
    }

    def _body(self, query: str, opts: dict, window: tuple, notices: list) -> dict:
        body = {"q": site_query(query, opts.get("domains")), "count": opts["max_results"]}
        start, end = window
        if opts.get("freshness"):
            body["freshness"] = opts["freshness"]
        elif opts.get("recency") and not end:
            if opts["recency"] == "hour":
                self._notice(notices, "recency 'hour' was widened to a day: Brave's finest window is a day")
            body["freshness"] = _FRESHNESS[opts["recency"]]
        elif start or end:
            body["freshness"] = f"{start or '1970-01-01'}to{end or date.today().isoformat()}"
        if opts.get("country"):
            body["country"] = opts["country"]
        if opts.get("language"):
            body["search_lang"] = opts["language"]
        if opts.get("safe_search"):
            body["safesearch"] = opts["safe_search"]
        if opts.get("result_filter"):
            body["result_filter"] = opts["result_filter"]
        if opts.get("extra_snippets"):
            body["extra_snippets"] = True
        return body

    def _search(self, query, opts, window, notices, cost):
        response = self._http.request(
            "POST", _ENDPOINT, body=json.dumps(self._body(query, opts, window, notices)).encode("utf-8"),
            headers={"x-subscription-token": self._api_key, "Content-Type": "application/json",
                     "Accept": "application/json"})
        raw = response.data.decode("utf-8", errors="replace")
        if not 200 <= response.status < 300:
            raise RuntimeError(f"Brave Search API error [{response.status}]: {raw}")
        data = json.loads(raw)
        items = (data.get("web") or {}).get("results") or data.get("results") or []
        out = []
        for it in items:
            article = it.get("article") or {}
            authors = article.get("author") or []
            author = ", ".join(a.get("name", "") for a in authors if isinstance(a, dict)) if isinstance(authors, list) else ""
            # article.date is the publication date the page declares; page_age is
            # when Brave saw the page — 2026-09-30 for an HBR article of 09-01.
            out.append(self.record(it.get("title"), it.get("url"), article.get("date") or it.get("page_age"),
                                   author, [_plain(it.get("description"))] +
                                   [_plain(s) for s in it.get("extra_snippets") or []]))
        return out


def _plain(s: "str | None") -> str:
    """Brave marks the query's words with <strong> and escapes entities."""
    return html.unescape(re.sub(r"<[^>]+>", "", s or ""))
