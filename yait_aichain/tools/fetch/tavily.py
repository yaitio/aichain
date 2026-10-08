"""
tools.fetch.tavily — fetchTavily
================================

Tavily Extract behind the common :class:`~tools.fetch.Fetch` record.

Endpoint  : POST https://api.tavily.com/extract
Auth      : ``Authorization: Bearer tvly-…``
Docs      : https://docs.tavily.com/documentation/api-reference/endpoint/extract
            — answer checked live 2026-10-09.

Up to 20 URLs a call; a URL that fails comes back in ``failed_results`` and is
a page with ``error`` here. Tavily returns the text and a title and **no date
or author** — those fields stay empty. ``extract_depth: "advanced"`` reaches
more (tables, embedded content) at twice the price.

**Price.** One credit per five successful URLs (basic), read from the answer —
so a call with fewer than five reports 0 until the count reaches five.

Environment variable
--------------------
``TAVILY_API_KEY``
"""

from __future__ import annotations

import json

from ._base import Fetch

_ENDPOINT = "https://api.tavily.com/extract"


class fetchTavily(Fetch):
    """
    Web pages' text by Tavily Extract — up to 20 URLs a call, as Markdown.

    Examples
    --------
    ::

        from yait_aichain.tools import fetchTavily

        fetchTavily().fetch(["https://example.com/a", "https://example.com/b"])["pages"]
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
        response = self._http.request(
            "POST", _ENDPOINT, body=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json",
                     "Accept": "application/json"})
        raw = response.data.decode("utf-8", errors="replace")
        if not 200 <= response.status < 300:
            raise RuntimeError(f"Tavily API error [{response.status}]: {raw}")
        data = json.loads(raw)
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
