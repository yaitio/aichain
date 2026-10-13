"""
tools.fetch._html — an HTML page as Markdown, with the standard library.

``fetchLocal`` converted pages with MarkItDown, which brought ``magika`` and
``onnxruntime`` — tens of megabytes and a model loaded at import — to turn
tags into text. A page a model reads needs headings, paragraphs, lists,
links, emphasis, code and tables; ``html.parser`` reads all of them.

What is not kept: scripts, styles, forms, SVG and anything hidden by the
``hidden`` attribute. Navigation and footers are kept, as they were.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

#: Elements whose content is not text a reader sees.
_SKIP = {"script", "style", "noscript", "template", "svg", "head", "iframe", "canvas",
         "form", "button", "select", "option", "textarea", "object", "embed"}
#: Elements that end a line.
_BLOCK = {"p", "div", "section", "article", "main", "header", "footer", "nav", "aside",
          "figure", "figcaption", "address", "dl", "dt", "dd", "hr", "body", "html"}
_VOID = {"br", "hr", "img", "meta", "link", "input", "source", "wbr", "area", "base", "col"}


class _Markdown(HTMLParser):

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self._skip = 0
        self._pre = 0
        self._lists: list[list] = []          # [kind, counter]
        self._link: list[str | None] = []
        self._table: list[list[str]] | None = None
        self._cell: list[str] | None = None
        self._quote = 0

    # ── writing ───────────────────────────────────────────────────────
    def _write(self, text: str) -> None:
        if self._cell is not None:
            self._cell.append(text)
        else:
            self.out.append(text)

    def _break(self, blank: bool = False) -> None:
        if self._cell is not None:
            self._cell.append(" ")
            return
        self.out.append("\n\n" if blank else "\n")
        if self._quote:
            self.out.append("> " * self._quote)

    # ── tags ──────────────────────────────────────────────────────────
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if self._skip or tag in _SKIP or "hidden" in a:
            if tag not in _VOID:
                self._skip += 1
            return
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._break(blank=True)
            self._write("#" * int(tag[1]) + " ")
        elif tag in _BLOCK:
            self._break(blank=tag in ("p", "section", "article"))
            if tag == "hr":
                self._write("---")
                self._break(blank=True)
        elif tag == "br":
            self._break()
        elif tag == "blockquote":
            self._quote += 1
            self._break(blank=True)
        elif tag in ("ul", "ol"):
            self._lists.append([tag, 0])
            self._break()
        elif tag == "li":
            self._break()
            depth = max(0, len(self._lists) - 1)
            if self._lists and self._lists[-1][0] == "ol":
                self._lists[-1][1] += 1
                self._write("   " * depth + f"{self._lists[-1][1]}. ")
            else:
                self._write("   " * depth + "- ")
        elif tag == "pre":
            self._pre += 1
            self._break(blank=True)
            self._write("```\n")
        elif tag == "code" and not self._pre:
            self._write("`")
        elif tag in ("strong", "b"):
            self._write("**")
        elif tag in ("em", "i"):
            self._write("*")
        elif tag == "a":
            href = a.get("href") or ""
            self._link.append(href if href and not href.startswith(("javascript:", "#")) else None)
            if self._link[-1]:
                self._write("[")
        elif tag == "img":
            alt = (a.get("alt") or "").strip()
            if alt:
                self._write(f"[image: {alt}]")
        elif tag == "table":
            self._table = []
            self._break(blank=True)
        elif tag == "tr" and self._table is not None:
            self._table.append([])
        elif tag in ("td", "th") and self._table is not None:
            self._cell = []

    def handle_endtag(self, tag):
        if self._skip:
            if tag not in _VOID:
                self._skip -= 1
            return
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._break(blank=True)
        elif tag in _BLOCK and tag != "hr":
            self._break(blank=tag in ("p", "section", "article"))
        elif tag == "blockquote":
            self._quote = max(0, self._quote - 1)
            self._break(blank=True)
        elif tag in ("ul", "ol"):
            if self._lists:
                self._lists.pop()
            self._break()
        elif tag == "pre":
            self._pre = max(0, self._pre - 1)
            self._write("\n```")
            self._break(blank=True)
        elif tag == "code" and not self._pre:
            self._write("`")
        elif tag in ("strong", "b"):
            self._write("**")
        elif tag in ("em", "i"):
            self._write("*")
        elif tag == "a" and self._link:
            href = self._link.pop()
            if href:
                self._write(f"]({href})")
        elif tag in ("td", "th") and self._table is not None and self._cell is not None:
            if not self._table:
                self._table.append([])
            self._table[-1].append(" ".join("".join(self._cell).split()).replace("|", "\\|"))
            self._cell = None
        elif tag == "table" and self._table is not None:
            rows = [r for r in self._table if r]
            self._table = None
            if rows:
                width = max(len(r) for r in rows)
                rows = [r + [""] * (width - len(r)) for r in rows]
                lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
                lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
                self._write("\n".join(lines))
            self._break(blank=True)

    def handle_data(self, data):
        if self._skip:
            return
        if self._pre:
            self._write(data)
            return
        text = re.sub(r"\s+", " ", data)
        if text.strip() or (self.out and not self.out[-1].endswith((" ", "\n"))):
            self._write(text)


def html_to_markdown(html: str) -> str:
    """*html* as Markdown: blocks on their own lines, at most one blank line
    between them, and no trailing spaces."""
    parser = _Markdown()
    parser.feed(html)
    parser.close()
    text = "".join(parser.out)
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()
