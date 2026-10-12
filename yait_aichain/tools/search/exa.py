"""
tools.search.exa — searchExa
============================

The Exa Search API behind the common :class:`~tools.search.Search` format.

Endpoint  : POST https://api.exa.ai/search
Auth      : ``x-api-key`` header
Docs      : https://exa.ai/docs/reference/search — the OpenAPI spec there is the
            source of truth; read 2026-10-02 and checked live the same day.

Exa is a search engine built for agents: the same request that finds the pages
can return query-relevant **highlights** from each one, so the agent reads the
part that answers it without fetching the page. Highlights are on by default
here; ``text_chars`` adds the page text itself, cut to a length, into each
result's ``text``.

**Highlights are capped, at 1 000 characters a result by default.** Exa sizes
them to relevance and left alone that size is most of the page — measured
2026-10-02: 5 000 to 8 000 characters for each of three results, about 20 000
characters for a three-result search, which an agent then carries in its
context for the rest of the run. ``highlights.maxCharacters`` bounds the whole
of a result's highlights, and the price does not move with it.
``highlight_chars`` sets the cap; ``0`` turns highlights off.

Translated: dates and ``recency`` → ``startPublishedDate`` /
``endPublishedDate``, ``domains`` → ``includeDomains`` (path prefixes work),
``country`` → ``userLocation``. ``language`` has no field on this API and is
declined with a notice. At most 100 results.

**Exa's date filter lets undated pages through** (measured 2026-10-06: 31 of
73 results in a one-week window carried no date), so a dated window is a
preference here, not a guarantee.

Environment variable
--------------------
``EXA_API_KEY`` — from https://dashboard.exa.ai/api-keys
"""

from __future__ import annotations


from ...clients._http import request_json
from ._base import Search

_ENDPOINT = "https://api.exa.ai/search"

#: Characters of highlights per result unless asked otherwise. Ten results at
#: this size are about 10 000 characters — a search an agent can afford to
#: keep in its context; uncapped, the same search was measured at about six
#: times that.
_HIGHLIGHT_CHARS = 1000


class searchExa(Search):
    """
    Web search powered by the Exa Search API — results with the passages that
    answer the query.

    Examples
    --------
    ::

        from yait_aichain.tools import searchExa

        tool = searchExa()
        tool.run(input="KV cache in LLM inference explained", options={"max_results": 5, "recency": "year"})

        # News only, from two sites, with up to 2 000 characters of each page:
        tool.run(input="EU AI Act enforcement", options={
            "category": "news", "domains": ["reuters.com", "ft.com"], "text_chars": 2000})
    """

    name        = "searchExa"
    description = ("Search the web with Exa, a search engine built for agents. Returns "
                   "titles, URLs, dates and the passages of each page most relevant to the query.")
    max_results = 100
    _ENV_KEY    = "EXA_API_KEY"
    _PROVIDER   = "Exa"
    #: Auto search, up to 10 results. Measured 2026-10-08: the answer's own
    #: costDollars was $0.007 a call with highlights and with page text alike.
    _LIST_PRICE   = 0.007
    _PRICE_SOURCE = "exa.ai/pricing, $7 / 1 000 requests up to 10 results, read 2026-10-08"
    declines    = {"language": "Exa has no language filter"}
    extra_options = {
        "exclude_domains": {"type": "array", "items": {"type": "string"},
                            "description": "Leave out results from these domains."},
        "category":        {"type": "string",
                            "enum": ["company", "publication", "news", "personal site", "financial report", "people"],
                            "description": "Focus on one kind of page."},
        "search_type":     {"type": "string",
                            "enum": ["instant", "fast", "auto", "deep-lite", "deep", "deep-reasoning"],
                            "description": "'auto' (default) balances quality and speed; 'instant'/'fast' "
                                           "trade quality for latency; the 'deep' modes search iteratively and cost more."},
        "highlight_chars": {"type": "integer",
                            "description": f"Most characters of highlights per result (default {_HIGHLIGHT_CHARS}); "
                                           "0 turns them off."},
        "text_chars":      {"type": "integer",
                            "description": "Also return each page's text, cut to this many characters. Off by "
                                           "default — highlights usually carry the answer."},
    }

    def _body(self, query: str, opts: dict, window: tuple) -> dict:
        body: dict = {"query": query, "type": opts.get("search_type", "auto"),
                      "numResults": opts["max_results"], "contents": {}}
        cap = opts.get("highlight_chars", _HIGHLIGHT_CHARS)
        if cap:
            body["contents"]["highlights"] = {"maxCharacters": int(cap)}
        if opts.get("text_chars"):
            body["contents"]["text"] = {"maxCharacters": int(opts["text_chars"])}
        if opts.get("domains"):
            body["includeDomains"] = list(opts["domains"])
        if opts.get("exclude_domains"):
            body["excludeDomains"] = list(opts["exclude_domains"])
        if opts.get("category"):
            body["category"] = opts["category"]
        if opts.get("country"):
            body["userLocation"] = opts["country"]
        start, end = window
        if start:
            body["startPublishedDate"] = f"{start}T00:00:00.000Z"
        if end:
            body["endPublishedDate"] = f"{end}T23:59:59.999Z"
        return body

    def _search(self, query, opts, window, notices, cost):
        data = request_json(self._http, "POST", _ENDPOINT, json=self._body(query, opts, window),
                            headers={"x-api-key": self._api_key, "Accept": "application/json"},
                            what="Exa Search API")
        spent = (data.get("costDollars") or {}).get("total")
        if spent is not None:
            cost.update(usd=float(spent), basis="reported by Exa", usage=data["costDollars"])
        return [self.record(r.get("title"), r.get("url"), r.get("publishedDate"), r.get("author") or "",
                            r.get("highlights") or [], r.get("text") or "")
                for r in data.get("results") or []]
