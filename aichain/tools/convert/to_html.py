"""
tools.convert.to_html — convertToHTML
=======================================

Converts Markdown text to HTML, LaTeX, or normalised Markdown using
the ``mistletoe`` library (https://github.com/miyuchina/mistletoe).

Supported output formats
------------------------
  html      Standard HTML fragment (no <html>/<body> wrapper by default).
  latex     LaTeX document body, ready for inclusion in a .tex file.
  markdown  Normalised / reformatted Markdown — useful for round-trip
            clean-up or automated document modifications.

Installation
------------
    pip install mistletoe

Environment variable
--------------------
None required.
"""

from __future__ import annotations

import os

from .._base import Tool


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SUPPORTED_FORMATS = frozenset({"html", "latex", "markdown"})


# ---------------------------------------------------------------------------
# convertToHTML
# ---------------------------------------------------------------------------

class convertToHTML(Tool):
    """
    Convert Markdown text to HTML, LaTeX, or normalised Markdown.

    Uses ``mistletoe``'s built-in renderer system — each output format
    has a dedicated, fully-featured renderer rather than a regex-based
    post-processor.

    Parameters
    ----------
    None.  The tool is stateless; all inputs are provided to :meth:`run`.

    Examples
    --------
    Convert to HTML::

        from tools.convert import convertToHTML

        tool = convertToHTML()

        # Call-style — returns ToolResult, never raises
        result = tool(input="# Hello\\n\\nWorld", options={"format": "html"})
        if result:
            print(result.output)   # '<h1>Hello</h1>\\n<p>World</p>\\n'
        else:
            print("Error:", result.error)

        # Direct run — returns str, raises on error
        html = tool.run(input="# Hello", options={"format": "html"})

    Convert to LaTeX and save::

        tool.run(
            input="# Introduction\\n\\nSome **bold** text.",
            options={"format": "latex", "output_path": "intro.tex"},
        )

    Normalise / reformat Markdown::

        clean_md = tool.run(input=messy_markdown, options={"format": "markdown"})
    """

    name        = "convertToHTML"
    description = (
        "Convert Markdown text to HTML, LaTeX, or normalised Markdown. "
        "Supported formats: html, latex, markdown."
    )
    parameters  = {
        "type": "object",
        "properties": {
            "input": {
                "type": "string",
                "description": "Markdown text to convert.",
            },
            "options": {
                "type": "object",
                "description": "Conversion options.",
                "properties": {
                    "format": {
                        "type": "string",
                        "enum": ["html", "latex", "markdown"],
                        "description": "Output format. Default: 'html'.",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Optional file path to write the result.",
                    },
                },
            },
        },
        "required": ["input"],
    }

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(
        self,
        input:   str,
        options: dict | None = None,
    ) -> str:
        """
        Convert *input* from Markdown to the requested format.

        Parameters
        ----------
        input : str
            Markdown source text to convert.
        options : dict | None, optional
            Conversion options:
              ``format``      — target format, one of ``"html"``, ``"latex"``,
                               or ``"markdown"``.  Default: ``"html"``.
              ``output_path`` — if provided, the converted text is written to
                               this path.  Parent directories are created
                               automatically.

        Returns
        -------
        str
            The converted document as a string.

        Raises
        ------
        ImportError
            If the ``mistletoe`` package is not installed.
        ValueError
            If *format* is not one of the supported values.
        """
        opts        = options or {}
        fmt         = opts.get("format", "html")
        output_path = opts.get("output_path")

        fmt = fmt.lower().strip()
        if fmt not in _SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format {fmt!r}.  "
                f"Choose from: {sorted(_SUPPORTED_FORMATS)}"
            )

        result = self._convert(input, fmt)

        if output_path:
            parent = os.path.dirname(os.path.abspath(output_path))
            os.makedirs(parent, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as fh:
                fh.write(result)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _convert(text: str, fmt: str) -> str:
        """Dispatch to the appropriate mistletoe renderer."""
        import sys
        import os

        # The old tool file was named 'mistletoe.py', which shadowed the
        # installed library when tools/ was on sys.path.  This file is now
        # named 'to_html.py' so the shadowing issue no longer exists, but we
        # keep this defensive sys.path manipulation in case any caller has
        # added a directory containing a file named 'mistletoe.py' to the path.
        _this_dir = os.path.normcase(os.path.abspath(os.path.dirname(__file__)))
        _saved    = sys.path[:]
        sys.path  = [
            p for p in sys.path
            if os.path.normcase(os.path.abspath(p)) != _this_dir
        ]
        try:
            try:
                from mistletoe import Document
            except ImportError:
                raise ImportError(
                    "mistletoe is required for convertToHTML.  "
                    "Install it with:  pip install mistletoe"
                )

            if fmt == "html":
                from mistletoe.html_renderer import HtmlRenderer
                with HtmlRenderer() as renderer:
                    return renderer.render(Document(text))

            if fmt == "latex":
                from mistletoe.latex_renderer import LaTeXRenderer
                with LaTeXRenderer() as renderer:
                    return renderer.render(Document(text))

            # fmt == "markdown"
            from mistletoe.markdown_renderer import MarkdownRenderer
            with MarkdownRenderer() as renderer:
                return renderer.render(Document(text))
        finally:
            sys.path[:] = _saved

    def __repr__(self) -> str:
        return f"convertToHTML(name={self.name!r})"
