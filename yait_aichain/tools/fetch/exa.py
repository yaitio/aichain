"""
tools.fetch.exa — fetchExa
==========================

Exa Contents behind the common :class:`~tools.fetch.Fetch` record.

Endpoint  : POST https://api.exa.ai/contents
Auth      : ``x-api-key``
Docs      : https://exa.ai/docs/reference/get-contents — answer checked live
            2026-10-09.

Up to 100 URLs a call, from Exa's cache with a live crawl when a page is not
cached; ``max_age_hours`` sets how old a cached copy may be (0 fetches fresh, -1
cache only). Title, date and author come with the text. Measured 2026-10-08:
on articles behind a paywall Exa returns the public teaser, not the article.

**Price.** $1 per 1 000 pages of text, read from the answer.

Environment variable
--------------------
``EXA_API_KEY``
"""

from __future__ import annotations


from ...clients._http import request_json
from ._base import Fetch

_ENDPOINT = "https://api.exa.ai/contents"


class fetchExa(Fetch):
    """
    Web pages' text by Exa Contents — up to 100 URLs a call, with title, date and author.

    Examples
    --------
    ::

        from yait_aichain.tools import fetchExa

        fetchExa().fetch(["https://example.com/a", "https://example.com/b"], {"max_age_hours": 24})["pages"]
    """

    name        = "fetchExa"
    description = "Fetch a web page with Exa and return its text, with title, date and author."
    max_batch   = 100
    _ENV_KEY    = "EXA_API_KEY"
    _PROVIDER   = "Exa"
    _LIST_PRICE   = 0.001
    _PRICE_SOURCE = "exa.ai/pricing, contents $1 / 1 000 pages, read 2026-10-08"
    extra_options = {
        "max_age_hours": {"type": "integer",
                          "description": "Use a cached copy younger than this (0 fetches fresh, -1 cache only; at most 720)."},
    }

    def _fetch(self, urls, opts, notices, cost):
        body: dict = {"urls": urls, "text": {"maxCharacters": opts["max_chars"]}}
        if opts.get("max_age_hours") is not None:
            body["maxAgeHours"] = int(opts["max_age_hours"])
        data = request_json(self._http, "POST", _ENDPOINT, json=body,
                            headers={"x-api-key": self._api_key, "Accept": "application/json"},
                            what="Exa API")
        spent = (data.get("costDollars") or {}).get("total")
        if spent is not None:
            cost.update(usd=float(spent), basis="reported by Exa", usage=data["costDollars"])
        out = {}
        for r in data.get("results") or []:
            u = r.get("url") or r.get("id") or ""
            out[u] = self.page(u, final_url=u, title=r.get("title") or "", published=r.get("publishedDate") or "",
                               author=r.get("author") or "", text=r.get("text") or "")
        for s in data.get("statuses") or []:
            if s.get("status") != "success" and s.get("id"):
                err = s.get("error") or {}
                out[s["id"]] = self.page(s["id"], error=f"{err.get('tag', 'failed')}"
                                                        + (f" (HTTP {err['httpStatusCode']})" if err.get("httpStatusCode") else ""))
        return out
