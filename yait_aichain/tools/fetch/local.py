"""
tools.fetch.local — fetchLocal
==============================

A page fetched from this machine and converted to Markdown here — no
provider, no key, no charge, and nothing to install for HTML.

HTML is converted with the standard library (``tools.fetch._html``); text,
Markdown, JSON and XML are returned as they are; a PDF needs ``pypdf``
(``pip install "yait-aichain[fetch]"``) and says so when it is missing. Until
4.0 this was MarkItDown, which brought ``magika`` and ``onnxruntime`` — tens
of megabytes and a model loaded at import — to turn tags into text.

It reads what the site serves an ordinary client — measured 2026-10-08 on 34
articles, 21 came back in full; the rest were behind bot protection or a
paywall. The whole page is converted, so navigation and footers come with
the article. Title, date and author are read from the page's ``<head>``
(JSON-LD first, then meta tags).

**robots.txt is respected by default.** Before a page is fetched the site's
``robots.txt`` is read once per host and the page is skipped — a record with
``error`` set, no request made — when it disallows any of ``robots_agents``:
``*`` and the Claude crawlers by default, since a page an agent reads is
usually read for a model. Per RFC 9309 a robots.txt that answers 4xx allows
everything and one that answers 5xx disallows everything; a site that does not
answer at all is reported as unreachable. ``respect_robots=False`` turns the check off; that is the
caller's decision about the sites they read.

No User-Agent is invented: the request goes out as ``python-urllib3`` unless
``headers=`` says otherwise.
"""

from __future__ import annotations

import codecs
import io
import time

from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import urllib3

from ..._require import require
from ...clients._base import make_http
from .._security import assert_safe_url, safe_request
from ._html import html_to_markdown
from ._page import DEFAULT_TIMEOUT, _ACCEPT, page_metadata
from ._base import Fetch

#: ``*`` plus the names Anthropic's crawlers and user-directed fetcher answer to.
ROBOTS_AGENTS = ("*", "ClaudeBot", "Claude-User", "anthropic-ai")



#: What fetchLocal converts: a page's types. A zip, an office file or an
#: image is refused with its type named.
_PAGE_TYPES = ("text/", "html", "xml", "json", "application/pdf")
#: The most of one page fetchLocal reads.
MAX_PAGE_BYTES = 10 << 20


#: How long a robots.txt read is trusted, and how many hosts are remembered.
ROBOTS_TTL = 3600.0
ROBOTS_CACHE_SIZE = 256

class fetchLocal(Fetch):
    """
    A web page's text fetched from this machine and converted to Markdown — free.

    Parameters
    ----------
    timeout : (connect, read) seconds, default (10, 30)
    headers : dict, sent with every request
    respect_robots : bool, default True — skip pages robots.txt disallows
    robots_agents : names checked against robots.txt, default ``ROBOTS_AGENTS``

    Examples
    --------
    ::

        from yait_aichain.tools import fetchLocal

        fetchLocal().run("https://example.com")["pages"][0]["text"]
    """

    name        = "fetchLocal"
    description = ("Fetch a web page from this machine and return its text as Markdown, "
                   "with title, date and author where the page declares them.")
    max_batch   = 1
    _PROVIDER   = "local"
    _LIST_PRICE   = 0.0
    _PRICE_SOURCE = "runs locally"

    def __init__(self, timeout=DEFAULT_TIMEOUT, headers: dict | None = None,
                 respect_robots: bool = True, robots_agents=ROBOTS_AGENTS, *,
                 name: str | None = None, description: str | None = None) -> None:
        if isinstance(timeout, list):            # as YAML gives it back after Chain.save
            timeout = tuple(timeout)
        connect, read = timeout if isinstance(timeout, tuple) else (timeout, timeout)
        self._http = make_http(timeout=urllib3.Timeout(connect=connect, read=read),
                               retries=urllib3.Retry(total=2, backoff_factor=0.5, redirect=False))
        self.respect_robots = respect_robots
        # One name is one agent, not its letters.
        self.robots_agents = (robots_agents,) if isinstance(robots_agents, str) else tuple(robots_agents)
        self._robots: dict = {}
        self._timeout, self._headers = timeout, dict(headers or {})
        self._named(name, description)

    def _serialise_init_args(self) -> dict:
        """What `Chain.save` writes. Headers are not: they carry cookies and tokens."""
        if self._headers:
            import warnings
            warnings.warn(f"{self.name}: headers are not saved — they can carry credentials; "
                          "pass them again after Chain.load", RuntimeWarning, stacklevel=3)
        out = super()._serialise_init_args()
        out.update(timeout=list(self._timeout) if isinstance(self._timeout, tuple) else self._timeout,
                   respect_robots=self.respect_robots, robots_agents=list(self.robots_agents))
        return out

    def _get(self, url: str, limit: int | None = None):
        """GET *url*, every hop checked: ``(status, final_url, content_type, body)``.
        The body is read up to *limit* bytes and no further; past it, ``body``
        is ``None``."""
        response = safe_request(self._http, "GET", url, preload_content=False, cross_host=True,
                                headers={"Accept": _ACCEPT, **self._headers})
        try:
            final = getattr(response, "final_url", url)
            ctype = (response.headers.get("content-type") or "").lower()
            if response.status >= 400:
                return response.status, final, ctype, b""
            # Pages, not archives: a converter that opens a zip it is handed
            # unpacked a 149 KB answer to 950 MB. Text, markup, JSON and PDF
            # are what a page is; anything else is said, not converted.
            if ctype and not any(t in ctype for t in _PAGE_TYPES):
                return response.status, final, ctype, b""
            chunks, size = [], 0
            for piece in response.stream(64 * 1024):
                size += len(piece)
                if limit is not None and size > limit:
                    return response.status, final, ctype, None
                chunks.append(piece)
            return response.status, final, ctype, b"".join(chunks)
        finally:
            response.release_conn()

    def _fetch(self, urls, opts, notices, cost):
        u = urls[0]
        cost.update(usd=0.0, basis="runs locally")
        try:
            assert_safe_url(u)
            if self.respect_robots:
                refused = self._refused_by_robots(u)
                if refused:
                    return {u: self.page(u, error=refused)}
            status, final, ctype, body = self._get(u, MAX_PAGE_BYTES)
            if status >= 400:
                return {u: self.page(u, final_url=final, error=f"HTTP {status}")}
            if ctype and not any(t in ctype for t in _PAGE_TYPES):
                return {u: self.page(u, final_url=final,
                                     error=f"not a page ({ctype.split(';')[0]}); not converted")}
            if body is None:
                return {u: self.page(u, final_url=final,
                                     error=f"over {MAX_PAGE_BYTES:,} bytes; not fetched")}
            meta: dict = {}
            if "pdf" in ctype or body[:5] == b"%PDF-":
                text = _pdf_text(body)
            else:
                charset = _charset(ctype)
                html = body.decode(charset, errors="replace")
                if "html" in ctype or (not ctype and "<html" in html[:2048].lower()):
                    meta = page_metadata(html)
                    text = html_to_markdown(html)
                else:
                    text = html
        except Exception as exc:                       # one page failing is a record, not a crash
            return {u: self.page(u, error=f"{type(exc).__name__}: {exc}")}
        return {u: self.page(u, final_url=final, title=meta.get("title", ""),
                             published=meta.get("published", ""), author=meta.get("author", ""), text=text)}

    def _refused_by_robots(self, url: str) -> str:
        """Why robots.txt refuses *url*, or "" when it allows it."""
        parts = urlsplit(url)
        root = f"{parts.scheme}://{parts.netloc}"
        cached = self._robots.get(root)
        if cached is None or time.monotonic() - cached[0] > ROBOTS_TTL:
            parser = RobotFileParser()
            try:
                status, _, ctype, body = self._get(root + "/robots.txt", 512 * 1024)
                if 400 <= status < 500:
                    parser.allow_all = True
                elif status >= 500:
                    parser.disallow_all = True
                else:
                    parser.parse((body or b"").decode(_charset(ctype), errors="replace").splitlines())
            except Exception as exc:
                # The site did not answer at all: say so, rather than blame
                # robots.txt — and do not remember it, where one failed read
                # kept the host unreachable until the process restarted.
                return f"the site could not be reached ({type(exc).__name__}: {exc})"
            if not parser.disallow_all:            # a 5xx is asked again next time, too
                if len(self._robots) >= ROBOTS_CACHE_SIZE:
                    self._robots.pop(next(iter(self._robots)))
                self._robots[root] = (time.monotonic(), parser)
        else:
            parser = cached[1]
        if isinstance(parser, str):
            return parser
        if parser.disallow_all:
            return "robots.txt answered with a server error, so the site is treated as disallowing (RFC 9309)"
        denied = [a for a in self.robots_agents if not parser.can_fetch(a, url)]
        return f"robots.txt disallows {', '.join(denied)}" if denied else ""


def _charset(ctype: str) -> str:
    """The charset a Content-Type names, if Python knows it; UTF-8 otherwise —
    a charset Python had never heard of lost the page."""
    for part in ctype.split(";"):
        key, _, value = part.strip().partition("=")
        if key == "charset" and value:
            try:
                return codecs.lookup(value.strip('"')).name
            except LookupError:
                break
    return "utf-8"


def _pdf_text(body: bytes) -> str:
    """A PDF's text, page by page, with ``pypdf`` — the one format here the
    standard library cannot read."""
    pypdf = require("pypdf", package="pypdf", who="fetchLocal reading a PDF", extra="fetch")
    reader = pypdf.PdfReader(io.BytesIO(body))
    return "\n\n".join((page.extract_text() or "").strip() for page in reader.pages).strip()
