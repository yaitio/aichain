"""Tavily Extract (``POST https://api.tavily.com/extract``, Bearer) behind the
common :class:`~tools.fetch.Fetch` record. Up to 20 URLs a call; a URL in
``failed_results`` is a page with ``error`` here. Text and title only, no date
or author. One credit per five successful URLs, read from the answer, so a
call with fewer than five reports 0. Key: ``TAVILY_API_KEY``.
See docs/tools/fetch.md and
https://docs.tavily.com/documentation/api-reference/endpoint/extract.
"""

from __future__ import annotations


from ...clients._http import request_json
from ._base import Fetch

_ENDPOINT = "https://api.tavily.com/extract"


class fetchTavily(Fetch):
    """
    Web pages' text by Tavily Extract — up to 20 URLs a call, as Markdown.

    Examples
    --------
    ::

        from yait_aichain.tools import fetchTavily

        fetchTavily().run(["https://example.com/a", "https://example.com/b"])["pages"]
    """

    name        = "fetchTavily"
    description = "Fetch a web page with Tavily and return its text as Markdown."
    max_batch   = 20
    _ENV_KEY    = "TAVILY_API_KEY"
    _PROVIDER   = "Tavily"
    _CREDIT       = 0.008
    _LIST_PRICE   = _CREDIT / 5
    _PRICE_SOURCE = "docs.tavily.com/documentation/api-credits, 1 credit / 5 URLs, $0.008 / credit, read 2026-10-08"
    extra_options = {
        "extract_depth": {"type": "string", "enum": ["basic", "advanced"],
                          "description": "'basic' (default); 'advanced' reaches tables and embedded content at twice the price."},
    }

    def _fetch(self, urls, opts, notices, cost):
        body = {"urls": urls, "format": "markdown", "include_usage": True,
                "extract_depth": opts.get("extract_depth") or "basic"}
        data = request_json(self._http, "POST", _ENDPOINT, json=body,
                            headers={"Authorization": f"Bearer {self._api_key}", "Accept": "application/json"},
                            what="Tavily API")
        credits = (data.get("usage") or {}).get("credits")
        if credits is not None:
            cost.update(usd=credits * self._CREDIT, usage={"credits": credits},
                        basis="credits reported by Tavily, at the pay-as-you-go rate (billed per five URLs)")
        out = {}
        for r in data.get("results") or []:
            out[self._match(r.get("url", ""), urls)] = self.page(r.get("url", ""), final_url=r.get("url", ""),
                                                                 title=r.get("title", ""), text=r.get("raw_content") or "")
        for f in data.get("failed_results") or []:
            u = self._match(f.get("url", ""), urls)
            out[u] = self.page(u, error=f.get("error") or "failed")
        for u, p in out.items():
            p["url"] = u
        return out

    @staticmethod
    def _match(returned: str, asked: list) -> str:
        """Tavily echoes the URL, sometimes normalised; map it back to the one asked for."""
        if returned in asked:
            return returned
        strip = lambda s: s.rstrip("/").split("#")[0]
        return next((a for a in asked if strip(a) == strip(returned)), returned)
