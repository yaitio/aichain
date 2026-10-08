"""
tools.search.firecrawl — searchFirecrawl
========================================

The Firecrawl Search API (v2) behind the common :class:`~tools.search.Search`
format. Firecrawl is a scraper first, so the same call that finds the pages can
return each page's content as Markdown — ``text_chars`` asks for it, as on Exa,
and it lands in each result's ``text``.

Endpoint  : POST https://api.firecrawl.dev/v2/search
Auth      : ``Authorization: Bearer fc-…``
Docs      : https://docs.firecrawl.dev/api-reference/endpoint/search — read
            2026-10-08.

Translated: ``max_results`` → ``limit`` (at most 100), ``recency`` →
``tbs=qdr:h|d|w|m|y``, dates → ``tbs=cdr:1,cd_min:…,cd_max:…``, ``country`` →
``country``, ``domains`` → ``includeDomains``. That field takes host names
only, so a domain with a path prefix travels as ``site:`` in the query instead,
and the caller is told. ``language`` has no field on this API and is declined
with a notice.

Firecrawl's own options stay: ``exclude_domains`` (it cannot be combined with
``domains``, and asking for both is refused), ``categories`` (``research``,
``pdf``, ``developer``, ``gov``), ``sources`` (``web``, ``news``), ``location``,
``safe`` and ``tbs`` (raw, wins over the standard time options).

**Its time filter is not enforced** — measured 2026-10-08. A ``cdr`` range of
28.09–04.10.2026 returned exactly the unfiltered results, and with
``hbr.org`` all five fell outside it (2025-11 to 2026-08); ``qdr:w`` changed
the results but still let through a page from April. The filter is sent, and
every search that uses it says so in ``notices``: check the dates. And they
are only there to check when the page is scraped (``text_chars``) or comes
from the ``news`` source — a web result carries no date of its own.

Environment variable
--------------------
``FIRECRAWL_API_KEY`` — from https://www.firecrawl.dev/app/api-keys
"""

from __future__ import annotations

import json

from ._base import Search
from .brave import site_query

_ENDPOINT = "https://api.firecrawl.dev/v2/search"
_QDR = {"hour": "h", "day": "d", "week": "w", "month": "m", "year": "y"}
_CATEGORIES = ["research", "pdf", "developer", "gov"]


def _mdy(iso: str) -> str:
    y, m, d = iso.split("-")
    return f"{m}/{d}/{y}"


class searchFirecrawl(Search):
    """
    Web search powered by Firecrawl — results with a description and, on
    request, each page's content as Markdown.

    Examples
    --------
    ::

        from yait_aichain.tools import searchFirecrawl

        tool = searchFirecrawl()
        tool.run(input="web scraping for LLMs", options={"recency": "month", "max_results": 5})

        # The pages themselves, cut to 5 000 characters each:
        tool.search(input="EU AI Act", options={"domains": ["europa.eu"], "text_chars": 5000})
    """

    name        = "searchFirecrawl"
    description = ("Search the web with Firecrawl. Returns titles, URLs and a description "
                   "of each page; can return each page's content as Markdown.")
    max_results = 100
    _ENV_KEY    = "FIRECRAWL_API_KEY"
    _PROVIDER   = "Firecrawl"
    declines    = {"language": "Firecrawl has no language filter"}
    extra_options = {
        "exclude_domains": {"type": "array", "items": {"type": "string"},
                            "description": "Leave out results from these host names. Not with domains."},
        "categories":      {"type": "array", "items": {"type": "string", "enum": _CATEGORIES},
                            "description": "Restrict to research sites, PDFs, the developer index or "
                                           "the government index ('gov' combines with nothing else)."},
        "sources":         {"type": "array", "items": {"type": "string", "enum": ["web", "news"]},
                            "description": "Where to search: 'web' (default) and/or 'news'."},
        "location":        {"type": "string", "description": "A place to search from, e.g. 'Berlin,Germany'."},
        "safe":            {"type": "boolean", "description": "Filter explicit content."},
        "tbs":             {"type": "string",
                            "description": "Firecrawl's raw time filter, e.g. 'qdr:w' or 'sbd:1,qdr:m'; "
                                           "wins over recency and the dates."},
        "text_chars":      {"type": "integer",
                            "description": "Also return each page's content as Markdown, cut to this many "
                                           "characters. Scraping costs extra credits."},
    }

    def _body(self, query: str, opts: dict, window: tuple, notices: list) -> dict:
        domains = list(opts.get("domains") or [])
        if domains and opts.get("exclude_domains"):
            raise ValueError("searchFirecrawl takes domains or exclude_domains, not both")
        hosts = [d for d in domains if "/" not in d.strip("/")]
        if domains and len(hosts) < len(domains):
            self._notice(notices, "Firecrawl's domain filter takes host names only; the domains "
                                  "with a path were sent as site: in the query")
            query = site_query(query, domains)
        body: dict = {"query": query, "limit": opts["max_results"]}
        if domains and len(hosts) == len(domains):
            body["includeDomains"] = hosts
        if opts.get("exclude_domains"):
            body["excludeDomains"] = list(opts["exclude_domains"])
        start, end = window
        if opts.get("tbs") or opts.get("recency") or start or end:
            self._notice(notices, "Firecrawl takes the time filter but did not enforce it when measured "
                                  "on 2026-10-08; check each result's date (dates come with text_chars or "
                                  "the news source)")
        if opts.get("tbs"):
            body["tbs"] = opts["tbs"]
        elif opts.get("recency") and not end:
            body["tbs"] = f"qdr:{_QDR[opts['recency']]}"
        elif start or end:
            body["tbs"] = "cdr:1" + (f",cd_min:{_mdy(start)}" if start else "") + (f",cd_max:{_mdy(end)}" if end else "")
        if opts.get("country"):
            body["country"] = opts["country"]
        if opts.get("location"):
            body["location"] = opts["location"]
        if opts.get("safe") is not None:
            body["safe"] = bool(opts["safe"])
        if opts.get("categories"):
            body["categories"] = [{"type": c} for c in opts["categories"]]
        if opts.get("sources"):
            body["sources"] = [{"type": s} for s in opts["sources"]]
        if opts.get("text_chars"):
            body["scrapeOptions"] = {"formats": [{"type": "markdown"}]}
        return body

    def _search(self, query, opts, window, notices):
        response = self._http.request(
            "POST", _ENDPOINT, body=json.dumps(self._body(query, opts, window, notices)).encode("utf-8"),
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json",
                     "Accept": "application/json"})
        raw = response.data.decode("utf-8", errors="replace")
        if not 200 <= response.status < 300:
            raise RuntimeError(f"Firecrawl API error [{response.status}]: {raw}")
        data = json.loads(raw)
        if data.get("success") is False:
            raise RuntimeError(f"Firecrawl API error: {data.get('error') or raw}")
        if data.get("warning"):
            self._notice(notices, f"Firecrawl: {data['warning']}")
        found = data.get("data") or {}
        cut = int(opts.get("text_chars") or 0)
        out = []
        for item in (found.get("web") or []) + (found.get("news") or []):
            meta = item.get("metadata") or {}
            when = item.get("date") or meta.get("publishedTime") or meta.get("article:published_time") or ""
            text = (item.get("markdown") or "")[:cut] if cut else ""
            out.append(self.record(item.get("title") or meta.get("title"), item.get("url") or meta.get("url"),
                                   when, meta.get("author") or "",
                                   [item.get("description") or item.get("snippet") or ""], text))
        return out
