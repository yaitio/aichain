"""What ``fetchLocal`` reads from a page's ``<head>``: its title, dates and
author, which a conversion of the body leaves out. See docs/tools/fetch.md.
"""

from __future__ import annotations

import json
from html.parser import HTMLParser


#: ``(connect, read)`` seconds for a URL: without one a silent server is a
#: call that never ends.
DEFAULT_TIMEOUT = (10.0, 30.0)

#: The ``Accept`` header a page fetch sends: Markdown first, where a site serves it.
_ACCEPT = "text/markdown, text/html;q=0.9, text/plain;q=0.8, */*;q=0.1"


# ----------------------------------------------------------------------
# Page metadata
# ----------------------------------------------------------------------

#: Meta names that carry a publication date, best first. Publishers disagree
#: on the word; these are the ones in common use (Open Graph, Dublin Core,
#: Google Scholar, Parse.ly, Sailthru, and the bare ``date``).
_PUBLISHED = ("article:published_time", "datepublished", "citation_publication_date",
              "dc.date", "dc.date.issued", "dcterms.date", "dcterms.issued",
              "parsely-pub-date", "sailthru.date", "publish_date", "publication_date",
              "pubdate", "date")
_MODIFIED  = ("article:modified_time", "datemodified", "og:updated_time", "dcterms.modified")
_AUTHOR    = ("author", "article:author", "citation_author", "dc.creator", "parsely-author")
_TITLE     = ("og:title", "twitter:title", "citation_title", "dc.title")
_DESC      = ("description", "og:description", "twitter:description")


class _HeadReader(HTMLParser):
    """Collects the head a body conversion leaves out: ``<meta>``, ``<title>``, the
    canonical link and JSON-LD blocks."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.meta: dict[str, str] = {}
        self.title = ""
        self.canonical = ""
        self.ld: list = []
        self._in = None
        self._buf: list[str] = []

    def handle_starttag(self, tag, attrs):
        a = {k.lower(): (v or "") for k, v in attrs}
        if tag == "meta":
            key = (a.get("property") or a.get("name") or a.get("itemprop") or "").strip().lower()
            if key and a.get("content", "").strip() and key not in self.meta:
                self.meta[key] = a["content"].strip()
        elif tag == "link" and "canonical" in a.get("rel", "").lower().split():
            self.canonical = self.canonical or a.get("href", "").strip()
        elif tag == "title" and not self.title:
            self._in, self._buf = "title", []
        elif tag == "script" and a.get("type", "").lower() == "application/ld+json":
            self._in, self._buf = "ld", []

    def handle_data(self, data):
        if self._in:
            self._buf.append(data)

    def handle_endtag(self, tag):
        if self._in == "title" and tag == "title":
            self.title = " ".join("".join(self._buf).split())
            self._in = None
        elif self._in == "ld" and tag == "script":
            try:
                self.ld.append(json.loads("".join(self._buf)))
            except ValueError:
                pass
            self._in = None


def _ld_objects(node):
    """Every dict in a JSON-LD tree, ``@graph`` included."""
    if isinstance(node, list):
        for item in node:
            yield from _ld_objects(item)
    elif isinstance(node, dict):
        yield node
        for value in node.values():
            if isinstance(value, (list, dict)):
                yield from _ld_objects(value)


def _ld_name(value) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return str(value.get("name") or "").strip()
    if isinstance(value, list):
        return ", ".join(n for n in (_ld_name(v) for v in value) if n)
    return ""


def page_metadata(html: str) -> dict:
    """
    Title, publication and modification dates, author, description, site name
    and canonical URL of an HTML page — the ``<head>`` a body conversion leaves out.

    JSON-LD is read first, because a publisher who writes it means it; then
    the meta tags. Values are returned as the page states them, dates
    included, and a field the page does not carry is left out rather than
    guessed.
    """
    reader = _HeadReader()
    try:
        reader.feed(html)
    except Exception:  # a malformed page still yields what was read before it
        pass
    meta, out = reader.meta, {}

    article = None
    for obj in _ld_objects(reader.ld):
        if obj.get("datePublished") or obj.get("headline"):
            kind = obj.get("@type")
            kinds = kind if isinstance(kind, list) else [kind]
            if any(isinstance(k, str) and ("Article" in k or "Posting" in k or k == "Report")
                   for k in kinds):
                article = obj
                break
            article = article or obj
    if article:
        for key, field in (("title", "headline"), ("published", "datePublished"),
                           ("modified", "dateModified"), ("description", "description")):
            if isinstance(article.get(field), str) and article[field].strip():
                out[key] = article[field].strip()
        if _ld_name(article.get("author")):
            out["author"] = _ld_name(article.get("author"))

    def first(keys):
        return next((meta[k] for k in keys if meta.get(k)), "")

    for key, keys in (("title", _TITLE), ("published", _PUBLISHED), ("modified", _MODIFIED),
                      ("author", _AUTHOR), ("description", _DESC)):
        if key not in out and first(keys):
            out[key] = first(keys)
    if "title" not in out and reader.title:
        out["title"] = reader.title
    if meta.get("og:site_name"):
        out["site"] = meta["og:site_name"]
    if reader.canonical:
        out["canonical"] = reader.canonical
    return out
