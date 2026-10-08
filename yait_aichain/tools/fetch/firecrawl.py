"""
tools.fetch.firecrawl — fetchFirecrawl
======================================

Firecrawl Scrape behind the common :class:`~tools.fetch.Fetch` record.

Endpoint  : POST https://api.firecrawl.dev/v2/scrape
Auth      : ``Authorization: Bearer fc-…``
Docs      : https://docs.firecrawl.dev/api-reference/endpoint/scrape — answer
            checked live 2026-10-09.

One URL a call. ``only_main_content`` (default true) asks Firecrawl to drop
navigation, headers and footers — measured 2026-10-08 on 34 articles, half of
what came back was still page furniture. Of the five fetch tools this one
reached the most: text from sites behind bot protection (McKinsey, INSEAD) and
from articles readers see only with a subscription (HBR) — how, the
measurement did not establish.

**Price.** A page is one credit, read from the answer; a PDF is billed per PDF
page. On low plans the scrape endpoint takes about ten requests a minute: a 429
is retried after the wait the answer asks for.

Environment variable
--------------------
``FIRECRAWL_API_KEY``
"""

from __future__ import annotations

import json

from ._base import Fetch

_ENDPOINT = "https://api.firecrawl.dev/v2/scrape"


class fetchFirecrawl(Fetch):
    """
    A web page's text by Firecrawl Scrape — the main content, as Markdown, with
    the page's metadata.

    Examples
    --------
    ::

        from yait_aichain.tools import fetchFirecrawl

        fetchFirecrawl().fetch(["https://example.com/a", "https://example.com/b"])["pages"]
    """

    name        = "fetchFirecrawl"
    description = ("Fetch a web page with Firecrawl and return its main content as Markdown, "
                   "with title, date and author where the page declares them.")
    max_batch   = 1
    _ENV_KEY    = "FIRECRAWL_API_KEY"
    _PROVIDER   = "Firecrawl"
    _CREDIT       = 19 / 5000           # Hobby plan; a larger plan pays less
    _LIST_PRICE   = _CREDIT
    _PRICE_SOURCE = "firecrawl.dev/pricing, 1 credit a page, Hobby $19 / 5 000 credits, read 2026-10-08"
    extra_options = {
        "only_main_content": {"type": "boolean",
                              "description": "Drop navigation, headers and footers (default true)."},
    }

    def _fetch(self, urls, opts, notices, cost):
        u = urls[0]
        body = {"url": u, "formats": ["markdown"], "onlyMainContent": opts.get("only_main_content", True) is not False}
        response = self._http.request(
            "POST", _ENDPOINT, body=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json",
                     "Accept": "application/json"})
        raw = response.data.decode("utf-8", errors="replace")
        if response.status == 402:
            raise RuntimeError(f"Firecrawl API error [{response.status}]: {raw}")   # out of credits: the caller's to fix
        try:
            data = json.loads(raw)
        except ValueError:
            return {u: self.page(u, error=f"HTTP {response.status}: {raw[:200]}")}
        if not 200 <= response.status < 300 or data.get("success") is False:
            return {u: self.page(u, error=f"HTTP {response.status}: {data.get('error') or raw[:200]}")}
        d = data.get("data") or {}; meta = d.get("metadata") or {}
        credits = meta.get("creditsUsed", data.get("creditsUsed"))
        if credits is not None:
            cost.update(usd=credits * self._CREDIT, usage={"credits": credits},
                        basis="credits reported by Firecrawl, at the Hobby plan's rate")
        status = meta.get("statusCode")
        if isinstance(status, int) and status >= 400:
            # Firecrawl returns an error page and charges for it; it is not the page asked for.
            return {u: self.page(u, final_url=meta.get("url", ""), error=f"the page answered HTTP {status}")}
        published = (meta.get("publishedTime") or meta.get("article:published_time")
                     or meta.get("datePublished") or meta.get("og:article:published_time") or "")
        author = meta.get("author") or meta.get("article:author") or ""
        return {u: self.page(u, final_url=meta.get("url") or meta.get("sourceURL") or "", title=meta.get("title", ""),
                             published=published, author=author if isinstance(author, str) else "",
                             text=d.get("markdown") or "")}
