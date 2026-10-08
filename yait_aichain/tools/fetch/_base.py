"""
tools.fetch._base
=================

``Fetch`` — the text of a web page from its URL, one record for every provider.

A search finds the link; a fetch brings back the page. Four providers sell
this as an API — Firecrawl Scrape, Tavily Extract, Exa Contents, Parallel
Extract — and one more runs locally and free: MarkItDown, the same library as
``convertToMD``, installed only with ``pip install "yait-aichain[fetch]"``.

They differ in what they return (Tavily has no metadata at all), how many
URLs a call takes (one to a hundred), how they report what a call cost, and in
what they can reach: measured on 40 articles on 2026-10-08, Firecrawl returned
text from sites behind bot protection and from articles readers see only with a
subscription, Exa and Tavily mostly the public teaser, and a local fetch what
the site serves an ordinary client. This class holds the record; each provider
translates.

Pages — one record
------------------
``fetch()`` returns ``{"pages", "notices", "cost", "cost_basis", "usage"}``;
each page is::

    {"url": str,           # as asked
     "final_url": str,     # where it ended, after redirects, when the provider says
     "title": str, "published": "YYYY-MM-DD" or "", "author": str,
     "text": str,          # the page as Markdown, cut to max_chars
     "error": str}         # "" when the page came back

A page that failed is still a record, with ``error`` set and no text: one bad
URL does not fail the call. ``published`` is what the page or the provider
declares, normalised to a day, or empty.

``run()`` — what an agent calls — takes one URL and returns the page as text,
or raises with the reason when it could not be fetched.

Standard option
---------------
``max_chars`` — most characters of text per page, default 20 000; cut at the
provider where it takes a limit, here otherwise.
"""

from __future__ import annotations

import os
import warnings

from ...clients._base import make_http
from .._base import Tool
from ..search._base import SEARCH_RETRIES, SEARCH_TIMEOUT, day

DEFAULT_MAX_CHARS = 20_000

STANDARD_OPTIONS = {
    "max_chars": {"type": "integer",
                  "description": f"Most characters of text per page (default {DEFAULT_MAX_CHARS:,}).".replace(",", " ")},
}


class Fetch(Tool):
    """
    Base of the fetch tools: one record of a page, a provider per subclass.

    A subclass sets ``name``, ``description``, ``max_batch`` (URLs a call
    takes), ``extra_options``, ``_ENV_KEY``, ``_PROVIDER``, ``_LIST_PRICE``
    (USD a page at the published rate) and ``_PRICE_SOURCE``, and implements
    ``_fetch(urls, opts, notices, cost) -> {url: page}`` returning records built
    with ``self.page(...)`` and, where the answer reports what the call cost,
    filling ``cost`` with ``usd``, ``basis`` and ``usage``.
    """

    name        = "fetch"
    description = "Fetch a web page by its URL and return its text as Markdown."
    # It reads the web and changes nothing anywhere.
    risk        = "read"
    max_batch: int = 1
    extra_options: dict = {}
    _ENV_KEY: "str | tuple" = ()
    _PROVIDER: str = ""
    _LIST_PRICE: float = 0.0
    _PRICE_SOURCE: str = ""

    parameters = {
        "type": "object",
        "properties": {
            "url":     {"type": "string", "description": "The page's URL, http or https."},
            "options": {"type": "object", "description": "Optional settings.", "properties": dict(STANDARD_OPTIONS)},
        },
        "required": ["url"],
    }

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        props = dict(STANDARD_OPTIONS); props.update(cls.extra_options)
        cls.parameters = {
            "type": "object",
            "properties": {
                "url":     {"type": "string", "description": "The page's URL, http or https."},
                "options": {"type": "object", "description": "Optional settings.", "properties": props},
            },
            "required": ["url"],
        }

    def __init__(self, api_key: "str | None" = None) -> None:
        key = api_key or (os.environ.get(self._ENV_KEY) if isinstance(self._ENV_KEY, str) else None)
        if not key:
            raise ValueError(f"No {self._PROVIDER} API key found. Pass api_key= or set the "
                             f"{self._ENV_KEY} environment variable.")
        self._api_key = key
        self._http = make_http(timeout=SEARCH_TIMEOUT, retries=SEARCH_RETRIES)

    # ------------------------------------------------------------------

    def fetch(self, urls, options: "dict | None" = None) -> dict:
        """Fetch one URL or a list of them; every URL comes back as a page record."""
        wanted = [urls] if isinstance(urls, str) else list(urls or [])
        if not wanted or not all(isinstance(u, str) and u.strip() for u in wanted):
            raise ValueError(f"{type(self).__name__} needs a URL or a list of URLs")
        wanted = list(dict.fromkeys(u.strip() for u in wanted))
        for u in wanted:
            if not u.lower().startswith(("http://", "https://")):
                raise ValueError(f"{type(self).__name__} fetches http and https URLs; got {u!r}")
        opts = dict(options or {})
        opts["max_chars"] = int(opts.get("max_chars") or DEFAULT_MAX_CHARS)
        notices: list = []
        total = {"usd": 0.0, "usage": {}, "basis": ""}
        got: dict = {}
        for i in range(0, len(wanted), self.max_batch):
            cost: dict = {}
            got.update(self._fetch(wanted[i:i + self.max_batch], opts, notices, cost))
            if "usd" not in cost:
                cost.update(usd=self._LIST_PRICE * len(wanted[i:i + self.max_batch]),
                            basis=f"published rate ({self._PRICE_SOURCE})")
            total["usd"] += cost["usd"]; total["basis"] = cost["basis"]
            for k, v in (cost.get("usage") or {}).items():
                total["usage"][k] = total["usage"].get(k, 0) + v if isinstance(v, (int, float)) else v
        pages = []
        for u in wanted:
            p = got.get(u) or self.page(u, error=f"{self._PROVIDER} returned nothing for this URL")
            if p["text"] and len(p["text"]) > opts["max_chars"]:
                p["text"] = p["text"][:opts["max_chars"]]
            pages.append(p)
        return {"pages": pages, "notices": notices, "cost": round(total["usd"], 6),
                "cost_basis": total["basis"], "usage": total["usage"]}

    def run(self, url: str, options: "dict | None" = None) -> str:  # type: ignore[override]
        """Fetch *url*; the page as text a model reads. Raises with the reason when it failed."""
        found = self.fetch(url, options)
        p = found["pages"][0]
        if p["error"]:
            raise RuntimeError(f"{self.name} could not fetch {url}: {p['error']}")
        return self.format(p, found["notices"])

    # ------------------------------------------------------------------

    @staticmethod
    def format(p: dict, notices: list = ()) -> str:
        head = [f"Title: {p['title'] or '(untitled)'}", f"URL: {p['final_url'] or p['url']}"]
        meta = ([f"Published: {p['published']}"] if p["published"] else []) + \
               ([f"Author: {p['author']}"] if p["author"] else [])
        if meta:
            head.append(" · ".join(meta))
        out = "\n".join(head) + "\n\n" + p["text"].strip()
        for n in notices:
            out += f"\n\nNote: {n}"
        return out

    @staticmethod
    def page(url, final_url="", title="", published="", author="", text="", error="") -> dict:
        """One page in the common shape — every key present."""
        return {"url": url, "final_url": final_url or "", "title": (title or "").strip(),
                "published": day(published), "author": (author or "").strip() if isinstance(author, str) else "",
                "text": text or "", "error": error or ""}

    def _notice(self, notices: list, text: str) -> None:
        notices.append(text)
        warnings.warn(f"{self.name}: {text}", stacklevel=4)

    def _fetch(self, urls: list, opts: dict, notices: list, cost: dict) -> dict:  # pragma: no cover
        raise NotImplementedError(f"{type(self).__name__} must implement _fetch()")
