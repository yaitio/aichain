"""
tools.fetch.parallel — fetchParallel
====================================

Parallel Extract behind the common :class:`~tools.fetch.Fetch` record.

Endpoint  : POST https://api.parallel.ai/v1/extract
Auth      : ``x-api-key``
Docs      : https://docs.parallel.ai/api-reference/extract/extract — answer
            checked live 2026-10-09.

Up to 20 URLs a call; the tool asks for each page's full content (Parallel's
default is excerpts only) and returns it with the title and date. A URL that
fails comes back in ``errors`` and is a page with ``error`` here.
``max_age_seconds`` (600 or more) fetches a page live when the cached copy is
older — slower.

**Price.** $1 per 1 000 URLs, from the operation counts in the answer.

Environment variable
--------------------
``PARALLEL_API_KEY``
"""

from __future__ import annotations


from ...clients._http import request_json
from ._base import Fetch

_ENDPOINT = "https://api.parallel.ai/v1/extract"
#: Parallel warns when Extract is called without an objective, because its
#: excerpts are chosen by it. This tool asks for full content and reads no
#: excerpts, so the warning says nothing about what it returns.
_NO_OBJECTIVE = "Neither objective nor search_queries were provided"


class fetchParallel(Fetch):
    """
    Web pages' full content by Parallel Extract — up to 20 URLs a call, with title and date.

    Examples
    --------
    ::

        from yait_aichain.tools import fetchParallel

        fetchParallel().run(["https://example.com/a", "https://example.com/b"])["pages"]
    """

    name        = "fetchParallel"
    description = "Fetch a web page with Parallel and return its full content as Markdown, with title and date."
    max_batch   = 20
    _ENV_KEY    = "PARALLEL_API_KEY"
    _PROVIDER   = "Parallel"
    _LIST_PRICE   = 0.001
    _PRICE_SOURCE = "docs.parallel.ai/getting-started/pricing, extract $1 / 1 000 URLs, read 2026-10-08"
    extra_options = {
        "max_age_seconds": {"type": "integer",
                            "description": "Fetch a page live when the cached copy is older than this (600 or more). Slower."},
    }

    def _fetch(self, urls, opts, notices, cost):
        settings: dict = {"full_content": {"max_chars_per_result": opts["max_chars"]}}
        if opts.get("max_age_seconds"):
            settings["fetch_policy"] = {"max_age_seconds": int(opts["max_age_seconds"])}
        data = request_json(self._http, "POST", _ENDPOINT, json={"urls": urls, "advanced_settings": settings},
                            headers={"x-api-key": self._api_key, "Accept": "application/json"},
                            what="Parallel API")
        if data.get("usage"):
            n = {u.get("name"): u.get("count", 0) for u in data["usage"] if isinstance(u, dict)}
            cost.update(usd=sum(v for k, v in n.items() if str(k).startswith("sku_extract")) * 0.001,
                        usage=n, basis="operations reported by Parallel, at the published rate")
        for w in data.get("warnings") or []:
            msg = w.get("message") if isinstance(w, dict) else str(w)
            if msg and _NO_OBJECTIVE not in msg:
                self._notice(notices, f"Parallel: {msg}")
        out = {}
        for r in data.get("results") or []:
            u = r.get("url", "")
            out[u] = self.page(u, final_url=u, title=r.get("title") or "", published=r.get("publish_date") or "",
                               text=r.get("full_content") or "\n\n".join(r.get("excerpts") or []))
        for e in data.get("errors") or []:
            u = e.get("url", "")
            out[u] = self.page(u, error=f"{e.get('error_type', 'failed')}"
                                       + (f" (HTTP {e['http_status_code']})" if e.get("http_status_code") else ""))
        return out
