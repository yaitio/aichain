"""
tools.convert.to_md — convertToMD
===================================

Converts files and URLs to Markdown using Microsoft's MarkItDown library
(https://github.com/microsoft/markitdown).

Supported input formats
-----------------------
  Documents   PDF, DOCX, PPTX, XLSX, XLS, ODP, ODT, ODS
  Web         HTML pages, URLs
  Text        Plain text, CSV, JSON, XML, YAML, RST, EPUB
  Code        Any source file (returned as a fenced code block)
  Images      JPEG, PNG, GIF, BMP, TIFF, WEBP
                → pass an LLM client at construction for AI descriptions
  Audio       WAV, MP3, M4A, FLAC, OGG
                → pass an LLM client at construction for transcription
  Archives    ZIP (converts each contained file recursively)

Optional LLM integration
-------------------------
MarkItDown can use an LLM to generate textual descriptions of images and
transcribe audio.  Pass a client and model name at construction::

    import openai
    from yait_aichain.tools.convert import convertToMD

    tool = convertToMD(
        llm_client=openai.OpenAI(),
        llm_model="gpt-4o",
    )
    result = tool(input="diagram.png")

Without an LLM client, images produce only basic metadata and audio files
are skipped.

Installation
------------
    pip install markitdown

Environment variable
---------------------
None required.  Pass API keys to the LLM client you supply, not here.
"""

from __future__ import annotations

import json
import os
from html.parser import HTMLParser
from typing import Any

from .._base import Tool
from .._security import assert_safe_url, confine_output_path, has_url_scheme

#: ``(connect, read)`` seconds for a URL. MarkItDown calls ``requests`` with
#: no timeout at all, so without one a silent server is a call that never ends.
DEFAULT_TIMEOUT = (10.0, 30.0)

#: The ``Accept`` header MarkItDown sets on the session it builds itself —
#: kept, since passing our own session would otherwise drop it.
_ACCEPT = "text/markdown, text/html;q=0.9, text/plain;q=0.8, */*;q=0.1"


def _session(timeout, headers=None):
    """A ``requests`` session that applies *timeout* to every request it makes
    unless the caller gave one, and sends *headers* on each."""
    import requests

    class _TimeoutSession(requests.Session):
        def request(self, *args, **kwargs):
            if timeout is not None:
                kwargs.setdefault("timeout", timeout)
            return super().request(*args, **kwargs)

    session = _TimeoutSession()
    session.headers.update({"Accept": _ACCEPT})
    session.headers.update(headers or {})
    return session


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
    """Collects what MarkItDown throws away: ``<meta>``, ``<title>``, the
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
    and canonical URL of an HTML page — the ``<head>`` MarkItDown drops.

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


def _front_matter(metadata: dict) -> str:
    """The metadata as a front-matter block; values JSON-quoted so a colon or
    a quote in a title cannot break it."""
    lines = [f"{k}: {json.dumps(v, ensure_ascii=False)}" for k, v in metadata.items()]
    return "---\n" + "\n".join(lines) + "\n---\n\n"


class convertToMD(Tool):
    """
    Convert a file or URL to Markdown text.

    Parameters
    ----------
    llm_client : Any | None, optional
        An OpenAI-compatible client instance used by MarkItDown to describe
        images and transcribe audio.  When ``None`` (default) those media
        types receive only basic metadata.

    llm_model : str | None, optional
        Model name passed to *llm_client*.  Required when *llm_client* is
        supplied.  Typical value: ``"gpt-4o"``.

    enable_builtins : bool | None, optional
        Passed directly to ``MarkItDown(enable_builtins=...)``.
        ``None`` (default) uses MarkItDown's own default.

    enable_plugins : bool | None, optional
        Passed directly to ``MarkItDown(enable_plugins=...)``.
        ``None`` (default) uses MarkItDown's own default.

    timeout : float | tuple[float, float] | None, optional
        Network timeout for a URL, in seconds — one number for both, or
        ``(connect, read)``. The read timeout bounds the wait between bytes,
        not the whole download. Default ``(10, 30)``; ``None`` waits forever,
        which MarkItDown alone does: measured 2026-10-06, a site that holds
        the connection open and never answers hung a conversion for hours.

    headers : dict | None, optional
        HTTP headers sent with every request for a URL, on top of the
        ``Accept`` header MarkItDown asks for. The usual one is
        ``User-Agent``: measured 2026-10-07, blogs.lse.ac.uk answers 403 to
        the ``python-requests`` agent on every path and 200 to a browser's.

    After every call, ``last_metadata`` holds what the page's ``<head>`` says
    about it — ``title``, ``published``, ``modified``, ``author``,
    ``description``, ``site``, ``canonical`` and the final ``url`` — for an
    HTML page fetched by URL or read from a local ``.html`` file, and ``{}``
    for anything else. MarkItDown drops the head, and with it the
    publication date, which on some sites appears nowhere in the body
    (library.hbs.edu, checked 2026-10-07). ``options={"metadata": True}``
    also puts it at the top of the returned Markdown as front matter.

    Examples
    --------
    Basic usage (no LLM)::

        from yait_aichain.tools.convert import convertToMD

        tool = convertToMD()

        # Call-style — returns ToolResult, never raises
        result = tool(input="report.pdf")
        if result:
            print(result.output)          # Markdown string
        else:
            print("Error:", result.error)

        # Direct run — returns str, raises on error
        markdown = tool.run(input="slides.pptx")

        # Save to file via options
        tool.run(input="data.xlsx", options={"output_path": "data.md"})

    With LLM for image descriptions::

        import openai
        tool = convertToMD(
            llm_client=openai.OpenAI(),
            llm_model="gpt-4o",
        )
        result = tool(input="architecture_diagram.png")
    """

    name = "convertToMD"
    description = (
        "Convert a file or URL to Markdown text.  Supports PDF, DOCX, PPTX, "
        "XLSX, HTML, plain text, CSV, JSON, XML, images, audio, ZIP archives, "
        "and more."
    )
    parameters = {
        "type": "object",
        "properties": {
            "input": {
                "type": "string",
                "description": "Absolute or relative file path, or a URL, to convert to Markdown.",
            },
            "options": {
                "type": "object",
                "description": "Conversion options.",
                "properties": {
                    "output_path": {
                        "type": "string",
                        "description": "Optional file path to write the result. Parent dirs created automatically.",
                    },
                    "metadata": {
                        "type": "boolean",
                        "description": ("For a web page, start the result with its title, "
                                        "publication date, author and canonical URL as front "
                                        "matter. Default false."),
                    },
                },
            },
        },
        "required": ["input"],
    }

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        llm_client:      Any        = None,
        llm_model:       str | None = None,
        enable_builtins: bool | None = None,
        enable_plugins:  bool | None = None,
        timeout:         "float | tuple[float, float] | None" = DEFAULT_TIMEOUT,
        headers:         "dict | None" = None,
    ) -> None:
        self._llm_client      = llm_client
        self._llm_model       = llm_model
        self._enable_builtins = enable_builtins
        self._enable_plugins  = enable_plugins
        self._timeout         = timeout
        self._headers         = dict(headers or {})
        self._md              = None   # lazily initialised
        self._session         = None
        self.last_metadata: dict = {}

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(
        self,
        input:   str,
        options: dict | None = None,
    ) -> str:
        """
        Convert *input* to Markdown and optionally save it to a file.

        Parameters
        ----------
        input : str
            File path (absolute or relative) or URL to convert.
        options : dict | None, optional
            Conversion options:
              ``output_path`` — destination file path for the Markdown output.
              Parent directories are created if they do not exist.  When
              omitted the result is only returned, not written to disk.
              ``metadata`` — when true, the page's metadata (see the class
              docstring) is put at the top as front matter.

        Returns
        -------
        str
            The full Markdown content of the converted document.

        Raises
        ------
        ImportError
            If the ``markitdown`` package is not installed.
        FileNotFoundError
            If *input* is a local path that does not exist.
        """
        opts        = options or {}
        output_path = opts.get("output_path")

        # If the input is a URL, only allow public http(s) targets — this blocks
        # SSRF to internal/metadata hosts and file:// local-file disclosure.
        is_url = has_url_scheme(input)
        if is_url:
            assert_safe_url(input)

        md = self._get_converter()
        self.last_metadata = {}
        if is_url and input.strip().lower().startswith(("http:", "https:")):
            # Fetched here rather than by MarkItDown so the <head> it drops can
            # be read from the same response — one request, not two.
            response = self._session.get(input.strip(), stream=True)
            response.raise_for_status()
            ctype = response.headers.get("content-type", "").lower()
            if "html" in ctype:
                # requests assumes Latin-1 for text/* without a charset; the
                # web is UTF-8 unless it says otherwise.
                charset = response.encoding if "charset=" in ctype else "utf-8"
                self.last_metadata = page_metadata(
                    response.content.decode(charset or "utf-8", errors="replace"))
                self.last_metadata["url"] = response.url
            result = md.convert_response(response)
        else:
            if (not is_url and os.path.splitext(input)[1].lower() in (".html", ".htm")
                    and os.path.isfile(input)):
                with open(input, encoding="utf-8", errors="replace") as fh:
                    self.last_metadata = page_metadata(fh.read())
            result = md.convert(input)
        text = result.text_content
        if opts.get("metadata") and self.last_metadata:
            text = _front_matter(self.last_metadata) + text

        if output_path:
            output_path = confine_output_path(output_path)
            parent = os.path.dirname(output_path)
            os.makedirs(parent, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as fh:
                fh.write(text)

        return text

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_converter(self):
        """
        Return the shared ``MarkItDown`` instance, creating it on first use.

        Raises
        ------
        ImportError
            If ``markitdown`` is not installed.
        """
        if self._md is not None:
            return self._md

        try:
            from markitdown import MarkItDown
        except ImportError:
            raise ImportError(
                "markitdown is required for convertToMD.  "
                "Install it with:  pip install markitdown"
            )

        kwargs: dict = {}
        if self._enable_builtins is not None:
            kwargs["enable_builtins"] = self._enable_builtins
        if self._enable_plugins is not None:
            kwargs["enable_plugins"] = self._enable_plugins
        if self._llm_client is not None:
            kwargs["llm_client"] = self._llm_client
        if self._llm_model is not None:
            kwargs["llm_model"] = self._llm_model
        self._session = _session(self._timeout, self._headers)
        kwargs["requests_session"] = self._session

        self._md = MarkItDown(**kwargs)
        return self._md

    def __repr__(self) -> str:
        extras = []
        if self._llm_client is not None:
            extras.append(f"llm_model={self._llm_model!r}")
        extra_str = ", " + ", ".join(extras) if extras else ""
        return f"convertToMD(name={self.name!r}{extra_str})"
