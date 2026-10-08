"""
tools.search.parallel — searchParallel
======================================

The Parallel Search API behind the common :class:`~tools.search.Search`
format. Parallel is built for agents: it takes an **objective** in natural
language beside a few short keyword queries, and returns for each page the
excerpts that serve the objective.

Endpoint  : POST https://api.parallel.ai/v1/search
Auth      : ``x-api-key`` header
Docs      : https://docs.parallel.ai/api-reference/search/search — read
            2026-10-08.

The query an agent passes becomes both the ``objective`` and the one keyword
query. ``search_queries`` adds keyword queries of its own — Parallel advises
two or three of three to six words — and ``objective`` replaces the goal.

Translated: ``max_results`` → ``advanced_settings.max_results``, ``domains`` →
``source_policy.include_domains`` (path prefixes and bare extensions such as
``.edu`` work), ``start_date`` and ``recency`` → ``source_policy.after_date``
(a day, so ``hour`` is a day here and said so), ``country`` →
``advanced_settings.location``. Parallel filters from a start date only, so
``end_date`` is declined with a notice, and it has no language filter.

**Undated pages pass the date filter**, as on Exa — measured 2026-10-08: a
week on ``bcg.com/publications`` returned BCG pages with no date. And the
ceiling on results is 20, not documented and measured the same day.

Parallel's own options stay: ``objective``, ``search_queries``, ``mode``
(``turbo``, ``fast``, ``basic``, ``advanced`` — the default), ``exclude_domains``
(ignored by Parallel when ``domains`` is set, so the pair is refused),
``max_chars_total`` and ``max_age_seconds`` (fetch a page live when the cached
copy is older; slower).

Environment variable
--------------------
``PARALLEL_API_KEY`` — from https://platform.parallel.ai
"""

from __future__ import annotations

import json

from ._base import Search

_ENDPOINT = "https://api.parallel.ai/v1/search"


class searchParallel(Search):
    """
    Web search powered by Parallel — the excerpts of each page that serve the
    objective of the search.

    Examples
    --------
    ::

        from yait_aichain.tools import searchParallel

        tool = searchParallel()
        tool.run(input="What did the major consultancies publish about AI agents this week?",
                 options={"recency": "week", "domains": ["bcg.com/publications", "bain.com/insights"]})
        tool.search(input="Parallel Web Systems announcements",
                    options={"search_queries": ["Parallel Web Systems products",
                                                "Parallel Web Systems funding"]})
    """

    name        = "searchParallel"
    description = ("Search the web with Parallel, a search engine built for agents. Returns "
                   "titles, URLs, dates and the excerpts of each page that serve the goal of the search.")
    #: Not documented; asked for 100 on 2026-10-08, it returned 20.
    max_results = 20
    _ENV_KEY    = "PARALLEL_API_KEY"
    _PROVIDER   = "Parallel"
    declines    = {"end_date": "Parallel filters from a start date only",
                   "language": "Parallel has no language filter"}
    extra_options = {
        "objective":       {"type": "string",
                            "description": "The goal of the search in natural language; default: the query."},
        "search_queries":  {"type": "array", "items": {"type": "string"},
                            "description": "Short keyword queries, three to six words, to run beside the query."},
        "mode":            {"type": "string", "enum": ["turbo", "fast", "basic", "advanced"],
                            "description": "'advanced' (default) is the most thorough; 'turbo' the fastest."},
        "exclude_domains": {"type": "array", "items": {"type": "string"},
                            "description": "Leave out these domains or domain/path prefixes. Not with domains."},
        "max_chars_total": {"type": "integer",
                            "description": "Upper bound on characters of excerpts across all results."},
        "max_age_seconds": {"type": "integer",
                            "description": "Fetch a page live when the cached copy is older than this "
                                           "(600 or more). Slower."},
    }

    def _body(self, query: str, opts: dict, window: tuple, notices: list) -> dict:
        if opts.get("domains") and opts.get("exclude_domains"):
            raise ValueError("searchParallel takes domains or exclude_domains, not both — "
                             "Parallel ignores the exclusions when domains are set")
        queries = [query] + [q for q in opts.get("search_queries") or [] if q and q != query]
        body: dict = {"objective": opts.get("objective") or query, "search_queries": queries}
        if opts.get("mode"):
            body["mode"] = opts["mode"]
        if opts.get("max_chars_total"):
            body["max_chars_total"] = int(opts["max_chars_total"])
        settings: dict = {"max_results": opts["max_results"]}
        policy: dict = {}
        if opts.get("domains"):
            policy["include_domains"] = list(opts["domains"])
        if opts.get("exclude_domains"):
            policy["exclude_domains"] = list(opts["exclude_domains"])
        start, _ = window
        if start:
            if opts.get("recency") == "hour":
                self._notice(notices, "recency 'hour' was widened to a day: Parallel filters by date")
            policy["after_date"] = start
        if policy:
            settings["source_policy"] = policy
        if opts.get("country"):
            settings["location"] = str(opts["country"]).lower()
        if opts.get("max_age_seconds"):
            settings["fetch_policy"] = {"max_age_seconds": int(opts["max_age_seconds"])}
        body["advanced_settings"] = settings
        return body

    def _search(self, query, opts, window, notices):
        response = self._http.request(
            "POST", _ENDPOINT, body=json.dumps(self._body(query, opts, window, notices)).encode("utf-8"),
            headers={"x-api-key": self._api_key, "Content-Type": "application/json",
                     "Accept": "application/json"})
        raw = response.data.decode("utf-8", errors="replace")
        if not 200 <= response.status < 300:
            raise RuntimeError(f"Parallel API error [{response.status}]: {raw}")
        data = json.loads(raw)
        for w in data.get("warnings") or []:
            self._notice(notices, f"Parallel: {w.get('message') if isinstance(w, dict) else w}")
        return [self.record(r.get("title"), r.get("url"), r.get("publish_date"), "", r.get("excerpts") or [])
                for r in data.get("results") or []][:opts["max_results"]]
