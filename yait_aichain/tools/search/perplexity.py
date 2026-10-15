"""Perplexity Search (``POST https://api.perplexity.ai/search``, Bearer) behind
the common :class:`~tools.search.Search` format; a substantive passage from
each page. ``recency`` → ``search_recency_filter``, dates → the MM/DD/YYYY
filters, ``domains`` → ``search_domain_filter`` (at most 20), ``language`` →
``search_language_filter``; its own ``after_date``/``before_date`` win.
Key: ``PERPLEXITY_API_KEY``. See docs/tools/search.md and
https://docs.perplexity.ai/api-reference/search-post.
"""

from __future__ import annotations


from ...clients._http import request_json
from ._base import Search

_ENDPOINT = "https://api.perplexity.ai/search"
_MAX_DOMAINS = 20


def _mdy(iso: str) -> str:
    y, m, d = iso.split("-")
    return f"{m}/{d}/{y}"


class searchPerplexity(Search):
    """
    Web search powered by the Perplexity Search API — each result with a
    substantive passage from the page.

    Examples
    --------
    ::

        from yait_aichain.tools import searchPerplexity

        tool = searchPerplexity()
        tool.run(input="EU AI Act enforcement", options={
            "domains": ["europa.eu"], "start_date": "2026-09-01", "max_results": 5})
    """

    name        = "searchPerplexity"
    description = ("Search the web with Perplexity. Returns titles, URLs, dates and a "
                   "substantive passage from each page.")
    max_results = 20
    _ENV_KEY    = "PERPLEXITY_API_KEY"
    _PROVIDER   = "Perplexity"
    _LIST_PRICE   = 0.005
    _PRICE_SOURCE = "docs.perplexity.ai, $5 / 1 000 requests, read 2026-10-08"
    extra_options = {
        "after_date":  {"type": "string", "description": "Perplexity's own MM/DD/YYYY lower bound; wins over start_date."},
        "before_date": {"type": "string", "description": "Perplexity's own MM/DD/YYYY upper bound; wins over end_date."},
    }

    def _body(self, query: str, opts: dict, window: tuple) -> dict:
        body = {"query": query, "max_results": opts["max_results"]}
        start, end = window
        domains = opts.get("domains")
        if domains:
            if len(domains) > _MAX_DOMAINS:
                raise ValueError(f"searchPerplexity filters at most {_MAX_DOMAINS} domains; got {len(domains)}")
            body["search_domain_filter"] = list(domains)
        if opts.get("recency") and not end:
            body["search_recency_filter"] = opts["recency"]
        else:
            if start:
                body["search_after_date_filter"] = _mdy(start)
            if end:
                body["search_before_date_filter"] = _mdy(end)
        if opts.get("after_date"):
            body.pop("search_recency_filter", None)
            body["search_after_date_filter"] = opts["after_date"]
        if opts.get("before_date"):
            body["search_before_date_filter"] = opts["before_date"]
        if opts.get("country"):
            body["country"] = opts["country"]
        if opts.get("language"):
            lang = opts["language"]
            body["search_language_filter"] = lang if isinstance(lang, list) else [lang]
        return body

    def _search(self, query, opts, window, notices, cost):
        data = request_json(self._http, "POST", _ENDPOINT, json=self._body(query, opts, window),
                            headers={"Authorization": f"Bearer {self._api_key}", "Accept": "application/json"},
                            what="Perplexity Search API")
        return [self.record(r.get("title"), r.get("url"), r.get("date"), "", [r.get("snippet") or ""])
                for r in data.get("results", [])]
