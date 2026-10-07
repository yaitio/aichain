"""
tools.search.serp — searchSerp
==============================

SerpAPI behind the common :class:`~tools.search.Search` format — one key for
Google, Bing, DuckDuckGo, Baidu, Yandex and the other engines it scrapes.

Endpoint  : GET https://serpapi.com/search
Auth      : ``api_key`` query parameter
Docs      : https://serpapi.com/search-api

Translated for the Google engines: ``recency`` → ``tbs=qdr:h|d|w|m|y``, dates →
``tbs=cdr:1,cd_min:…,cd_max:…``, ``country`` → ``gl``, ``language`` → ``hl``.
``domains`` → ``site:`` in the query on every engine. On an engine that is not
Google the time, country and language filters are not applied, and the caller
is told. At most 100 results.

SerpAPI's own options stay: ``engine``, ``location``, ``safe``, ``start``,
``no_cache`` and ``time_filter`` (a raw ``tbs``, which wins over the standard
time options).

Result dates are what Google prints — ``Sep 30, 2026`` or ``3 days ago`` —
normalised to a day.

Environment variable
--------------------
``SERPAPI_API_KEY`` — from https://serpapi.com/manage-api-key
"""

from __future__ import annotations

import json
import urllib.parse

from ._base import Search
from .brave import site_query

_ENDPOINT = "https://serpapi.com/search"
_QDR = {"hour": "h", "day": "d", "week": "w", "month": "m", "year": "y"}


def _mdy(iso: str) -> str:
    y, m, d = iso.split("-")
    return f"{int(m)}/{int(d)}/{y}"


class searchSerp(Search):
    """
    Web search through SerpAPI — Google by default, or any engine it supports.

    Examples
    --------
    ::

        from yait_aichain.tools import searchSerp

        tool = searchSerp()
        tool.run(input="Python asyncio tutorial", options={"recency": "week", "country": "GB"})
        tool.run(input="人工智能", options={"engine": "baidu"})
    """

    name        = "searchSerp"
    description = ("Search the web through SerpAPI — Google by default, or Bing, "
                   "DuckDuckGo, Baidu, Yandex and others. Returns titles, URLs, dates and snippets.")
    max_results = 100
    _ENV_KEY    = "SERPAPI_API_KEY"
    _PROVIDER   = "SerpAPI"
    extra_options = {
        "engine":      {"type": "string",
                        "description": "The engine: 'google' (default), 'bing', 'duckduckgo', 'baidu', "
                                       "'yandex', 'google_news', 'google_scholar' and others."},
        "location":    {"type": "string", "description": "A place to search from, e.g. 'Austin, Texas'."},
        "safe":        {"type": "string", "enum": ["active", "off"], "description": "Google SafeSearch."},
        "start":       {"type": "integer", "description": "Offset of the first result, for paging."},
        "no_cache":    {"type": "boolean", "description": "Bypass SerpAPI's cache."},
        "time_filter": {"type": "string",
                        "description": "A raw Google 'tbs', e.g. 'qdr:w'; wins over recency and the dates."},
    }

    def _params(self, query: str, opts: dict, window: tuple, notices: list) -> dict:
        engine = opts.get("engine", "google")
        p = {"q": site_query(query, opts.get("domains")), "engine": engine,
             "num": opts["max_results"], "api_key": self._api_key}
        google = engine.startswith("google")
        start, end = window
        tbs = opts.get("time_filter")
        if not tbs and opts.get("recency") and not end:
            tbs = f"qdr:{_QDR[opts['recency']]}"
        elif not tbs and (start or end):
            tbs = "cdr:1" + (f",cd_min:{_mdy(start)}" if start else "") + (f",cd_max:{_mdy(end)}" if end else "")
        for key, field, value in (("time", "tbs", tbs), ("country", "gl", opts.get("country")),
                                  ("language", "hl", opts.get("language"))):
            if not value:
                continue
            if google:
                p[field] = value.lower() if field in ("gl", "hl") else value
            else:
                self._notice(notices, f"the {key} filter was not applied: SerpAPI takes it on the "
                                      f"Google engines, and this search used {engine}")
        if opts.get("location"):
            p["location"] = opts["location"]
        if opts.get("safe"):
            p["safe"] = opts["safe"]
        if opts.get("start") is not None:
            p["start"] = opts["start"]
        if opts.get("no_cache"):
            p["no_cache"] = "true"
        return p

    def _search(self, query, opts, window, notices):
        url = f"{_ENDPOINT}?{urllib.parse.urlencode(self._params(query, opts, window, notices))}"
        response = self._http.request("GET", url)
        raw = response.data.decode("utf-8", errors="replace")
        if not 200 <= response.status < 300:
            raise RuntimeError(f"SerpAPI error [{response.status}]: {raw[:400]}")
        data = json.loads(raw)
        if "error" in data:
            # "Google hasn't returned any results" is an answer, not a failure.
            if "hasn't returned any results" in str(data["error"]):
                return []
            raise RuntimeError(f"SerpAPI error: {data['error']}")
        items = data.get("organic_results") or data.get("news_results") or []
        return [self.record(it.get("title"), it.get("link"), it.get("date"),
                            (it.get("source") or "") if isinstance(it.get("source"), str) else "",
                            [it.get("snippet") or ""]) for it in items][:opts["max_results"]]
