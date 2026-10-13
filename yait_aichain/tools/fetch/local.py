"""
tools.fetch.local — fetchLocal
==============================

A page fetched from this machine and converted by MarkItDown — no provider,
no key, no charge.

Not installed by default: ``pip install "yait-aichain[fetch]"`` brings
MarkItDown, and constructing the tool without it raises with that line.

It reads what the site serves an ordinary client — measured 2026-10-08 on 34
articles, 21 came back in full; the rest were behind bot protection or a
paywall. MarkItDown converts the whole page, so navigation and footers come
with the article. Title, date and author are read from the page's ``<head>``
(JSON-LD first, then meta tags), the part MarkItDown drops.

**robots.txt is respected by default.** Before a page is fetched the site's
``robots.txt`` is read once per host and the page is skipped — a record with
``error`` set, no request made — when it disallows any of ``robots_agents``:
``*`` and the Claude crawlers by default, since a page an agent reads is
usually read for a model. Per RFC 9309 a robots.txt that answers 4xx allows
everything and one that answers 5xx disallows everything; a site that does not
answer at all is reported as unreachable. ``respect_robots=False`` turns the check off; that is the
caller's decision about the sites they read.

No User-Agent is invented: the request goes out as ``python-requests`` unless
``headers=`` says otherwise.
"""

from __future__ import annotations

import codecs
import time

from ..._require import require

from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

from .._security import assert_safe_url, safe_get
from ._page import DEFAULT_TIMEOUT, _session, page_metadata
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
    A web page's text fetched from this machine and converted by MarkItDown — free.

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
        MarkItDown = require("markitdown", package="markitdown", who="fetchLocal", extra="fetch").MarkItDown
        if isinstance(timeout, list):            # as YAML gives it back after Chain.save
            timeout = tuple(timeout)
        self._session = _session(timeout, headers)
        self._md = MarkItDown(requests_session=self._session)
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

    def _fetch(self, urls, opts, notices, cost):
        u = urls[0]
        cost.update(usd=0.0, basis="runs locally")
        try:
            assert_safe_url(u)
            if self.respect_robots:
                refused = self._refused_by_robots(u)
                if refused:
                    return {u: self.page(u, error=refused)}
            response = safe_get(self._session, u, stream=True)
            if response.status_code >= 400:
                return {u: self.page(u, final_url=response.url, error=f"HTTP {response.status_code}")}
            ctype = response.headers.get("content-type", "").lower()
            # Pages, not archives: MarkItDown opens a zip it is handed, and a
            # 149 KB answer unpacked to 950 MB. Text, markup, JSON and PDF are
            # what a page is; anything else is said, not converted.
            if ctype and not any(t in ctype for t in _PAGE_TYPES):
                response.close()
                return {u: self.page(u, final_url=response.url,
                                     error=f"not a page ({ctype.split(';')[0]}); not converted")}
            # Read up to the budget and no further — `stream=True` meant
            # nothing while `.content` read the whole body after it.
            body, size = [], 0
            for piece in response.iter_content(64 * 1024):
                size += len(piece)
                if size > MAX_PAGE_BYTES:
                    response.close()
                    return {u: self.page(u, final_url=response.url,
                                         error=f"over {MAX_PAGE_BYTES:,} bytes; not fetched")}
                body.append(piece)
            response._content = b"".join(body)      # what MarkItDown and `.content` read
            meta: dict = {}
            if "html" in ctype:
                charset = response.encoding if "charset=" in ctype else "utf-8"
                try:
                    codecs.lookup(charset or "utf-8")
                except LookupError:
                    charset = "utf-8"            # a charset Python has never heard of lost the page
                meta = page_metadata(response.content.decode(charset or "utf-8", errors="replace"))
            text = self._md.convert_response(response).text_content
        except Exception as exc:                       # one page failing is a record, not a crash
            return {u: self.page(u, error=f"{type(exc).__name__}: {exc}")}
        return {u: self.page(u, final_url=response.url, title=meta.get("title", ""),
                             published=meta.get("published", ""), author=meta.get("author", ""), text=text)}

    def _refused_by_robots(self, url: str) -> str:
        """Why robots.txt refuses *url*, or "" when it allows it."""
        parts = urlsplit(url)
        root = f"{parts.scheme}://{parts.netloc}"
        cached = self._robots.get(root)
        if cached is None or time.monotonic() - cached[0] > ROBOTS_TTL:
            parser = RobotFileParser()
            try:
                r = safe_get(self._session, root + "/robots.txt")
                if 400 <= r.status_code < 500:
                    parser.allow_all = True
                elif r.status_code >= 500:
                    parser.disallow_all = True
                else:
                    parser.parse(r.text.splitlines())
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
