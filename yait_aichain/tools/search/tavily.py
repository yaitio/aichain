"""Tavily Search (``POST https://api.tavily.com/search``, Bearer) behind the
common :class:`~tools.search.Search` format: up to three passages per page,
the page as Markdown with ``text_chars``. ``recency`` → ``time_range`` (finest
is a day), ``domains`` → ``include_domains``, ``country`` → the English name
Tavily takes, on the ``general`` topic only. Dates are Tavily's estimate;
undated pages pass unless ``filter_by_published_date``. Key: ``TAVILY_API_KEY``.
See docs/tools/search.md and
https://docs.tavily.com/documentation/api-reference/endpoint/search.
"""

from __future__ import annotations

import json
from pathlib import Path


from ...clients._http import request_json
from ._base import Search

_ENDPOINT = "https://api.tavily.com/search"
_TIME_RANGE = {"hour": "day", "day": "day", "week": "week", "month": "month", "year": "year"}

#: ISO 3166-1 alpha-2 → the name Tavily's ``country`` takes (its documented
#: list, 2026-10-08), as data beside this module. A name passed as such goes
#: through unchanged.
_COUNTRY = json.loads((Path(__file__).with_name("tavily_countries.json")).read_text())


class searchTavily(Search):
    """
    Web search powered by Tavily — each page with the passages chosen for the
    query, and on request the page itself.

    Examples
    --------
    ::

        from yait_aichain.tools import searchTavily

        tool = searchTavily()
        tool.run(input="EU AI Act enforcement", options={"recency": "month", "country": "DE"})
        tool.run(input="AI and leadership", options={
            "domains": ["hbr.org"], "start_date": "2026-09-28", "end_date": "2026-10-04",
            "filter_by_published_date": True})
    """

    name        = "searchTavily"
    description = ("Search the web with Tavily. Returns titles, URLs, dates and the passages "
                   "of each page chosen for the query.")
    max_results = 20
    _ENV_KEY    = "TAVILY_API_KEY"
    _PROVIDER   = "Tavily"
    _CREDIT       = 0.008                  # pay-as-you-go; monthly plans pay $0.005–0.0075
    _LIST_PRICE   = _CREDIT                # basic search: 1 credit
    _PRICE_SOURCE = "docs.tavily.com/documentation/api-credits, $0.008 / credit, read 2026-10-08"
    extra_options = {
        "topic":             {"type": "string", "enum": ["general", "news", "finance"],
                              "description": "What to search: 'general' (default), 'news' or 'finance'."},
        "search_depth":      {"type": "string", "enum": ["ultra-fast", "fast", "basic", "advanced"],
                              "description": "'basic' (default); 'advanced' is more relevant and costs two credits."},
        "chunks_per_source": {"type": "integer", "description": "Passages per page, 1 to 3 (default 3)."},
        "exclude_domains":   {"type": "array", "items": {"type": "string"},
                              "description": "Leave out results from these domains."},
        "exact_match":       {"type": "boolean",
                              "description": "Only pages containing the quoted phrases of the query exactly."},
        "safe_search":       {"type": "boolean", "description": "Filter adult content."},
        "filter_by_published_date": {"type": "boolean",
                              "description": "Drop pages outside the date window — and pages with no date."},
        "filter_by_language": {"type": "boolean",
                              "description": "Make language a filter rather than a ranking boost."},
        "text_chars":        {"type": "integer",
                              "description": "Also return each page as Markdown, cut to this many characters."},
    }

    def _body(self, query: str, opts: dict, window: tuple, notices: list) -> dict:
        # include_usage: without it the answer says nothing of the credits it cost.
        body: dict = {"query": query, "max_results": opts["max_results"], "include_published_date": True,
                      "include_usage": True}
        start, end = window
        if opts.get("recency") and not end:
            if opts["recency"] == "hour":
                self._notice(notices, "recency 'hour' was widened to a day: Tavily's finest window is a day")
            body["time_range"] = _TIME_RANGE[opts["recency"]]
        else:
            if start:
                body["start_date"] = start
            if end:
                body["end_date"] = end
        if opts.get("domains"):
            body["include_domains"] = list(opts["domains"])
        if opts.get("exclude_domains"):
            body["exclude_domains"] = list(opts["exclude_domains"])
        topic = opts.get("topic", "general")
        if opts.get("topic"):
            body["topic"] = topic
        if opts.get("country"):
            if topic != "general":
                self._notice(notices, f"country was not applied: Tavily takes it on the general topic, "
                                      f"and this search used {topic}")
            else:
                c = str(opts["country"])
                body["country"] = _COUNTRY.get(c.upper(), c.lower()) if len(c) == 2 else c.lower()
        if opts.get("language"):
            body["language"] = opts["language"]
        if opts.get("search_depth"):
            body["search_depth"] = opts["search_depth"]
        if opts.get("chunks_per_source"):
            body["chunks_per_source"] = int(opts["chunks_per_source"])
        flags = {"exact_match": opts.get("exact_match"), "safe_search": opts.get("safe_search"),
                 "filter_by_published_date": opts.get("filter_by_published_date"),
                 "filter_by_language": opts.get("filter_by_language")}
        body.update({k: bool(v) for k, v in flags.items() if v is not None})
        if opts.get("text_chars"):
            body["include_raw_content"] = "markdown"
        return body

    def _search(self, query, opts, window, notices, cost):
        data = request_json(self._http, "POST", _ENDPOINT, json=self._body(query, opts, window, notices),
                            headers={"Authorization": f"Bearer {self._api_key}", "Accept": "application/json"},
                            what="Tavily API")
        credits = (data.get("usage") or {}).get("credits")
        if credits is not None:
            cost.update(usd=credits * self._CREDIT, usage={"credits": credits},
                        basis="credits reported by Tavily, at the pay-as-you-go rate")
        cut = int(opts.get("text_chars") or 0)
        out = []
        for r in data.get("results") or []:
            # Passages arrive joined as "<chunk 1> [...] <chunk 2>".
            chunks = [c.strip() for c in (r.get("content") or "").split("[...]")]
            out.append(self.record(r.get("title"), r.get("url"), r.get("published_date"), "", chunks,
                                   (r.get("raw_content") or "")[:cut] if cut else ""))
        return out
