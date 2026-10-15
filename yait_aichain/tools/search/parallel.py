"""Parallel Search (``POST https://api.parallel.ai/v1/search``, ``x-api-key``)
behind the common :class:`~tools.search.Search` format. The query is both the
``objective`` and the one keyword query; ``search_queries`` adds more.
``domains`` → ``source_policy.include_domains``, ``start_date``/``recency`` →
``after_date`` (a day); ``end_date`` and ``language`` are declined. Undated
pages pass the date filter. Key: ``PARALLEL_API_KEY``.
See docs/tools/search.md and https://docs.parallel.ai/api-reference/search/search.
"""

from __future__ import annotations


from ...clients._http import request_json
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
        tool.run(input="Parallel Web Systems announcements",
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
    #: Per search: $0.005 basic/advanced (the default), $0.001 turbo/fast, ten results included;
    #: each further result with its excerpts $0.001 — billed as sku_extract_excerpts, measured 2026-10-08.
    _LIST_PRICE   = 0.005
    _PRICE_SOURCE = "docs.parallel.ai/getting-started/pricing, read 2026-10-08"
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

    def _search(self, query, opts, window, notices, cost):
        data = request_json(self._http, "POST", _ENDPOINT, json=self._body(query, opts, window, notices),
                            headers={"x-api-key": self._api_key, "Accept": "application/json"},
                            what="Parallel API")
        if data.get("usage"):
            n = {u.get("name"): u.get("count", 0) for u in data["usage"] if isinstance(u, dict)}
            per = 0.001 if opts.get("mode") in ("turbo", "fast") else 0.005
            cost.update(usd=n.get("sku_search", 0) * per + n.get("sku_extract_excerpts", 0) * 0.001,
                        usage=n, basis="operations reported by Parallel, at the published rates")
        for w in data.get("warnings") or []:
            self._notice(notices, f"Parallel: {w.get('message') if isinstance(w, dict) else w}")
        return [self.record(r.get("title"), r.get("url"), r.get("publish_date"), "", r.get("excerpts") or [])
                for r in data.get("results") or []][:opts["max_results"]]
