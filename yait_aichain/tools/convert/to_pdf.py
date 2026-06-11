"""
tools.convert.to_pdf — convertToPDF
=====================================

Renders HTML to PDF using the ``weasyprint`` library
(https://weasyprint.org/).

Accepts HTML either as a string or as a path to an HTML file.
Output is either written to a file (when ``output_path`` is provided in
options) or returned as raw ``bytes``.

Installation
------------
    pip install weasyprint

Environment variable
--------------------
None required.
"""

from __future__ import annotations

import os

from .._base import Tool


# ---------------------------------------------------------------------------
# convertToPDF
# ---------------------------------------------------------------------------

class convertToPDF(Tool):
    """
    Render an HTML document to PDF.

    Accepts HTML content as a string or a path to an existing HTML file.
    Output can be saved to a ``.pdf`` file or returned as raw bytes for
    programmatic use.

    Parameters
    ----------
    None.  The tool is stateless; all inputs are provided to :meth:`run`.

    Examples
    --------
    From an HTML string, save to file::

        from tools.convert import convertToPDF

        tool = convertToPDF()

        # Call-style — returns ToolResult, never raises
        result = tool(
            input="<h1>Hello</h1><p>World</p>",
            options={"output_path": "hello.pdf"},
        )
        if result:
            print("Saved to:", result.output)   # path string
        else:
            print("Error:", result.error)

        # Direct run — returns path string, raises on error
        path = tool.run(
            input="<h1>Hi</h1>",
            options={"output_path": "out.pdf"},
        )

    From an HTML file::

        tool.run(input="report.html", options={"output_path": "report.pdf"})

    Return raw bytes (no file written)::

        pdf_bytes = tool.run(input="<p>Hello</p>")
        with open("manual.pdf", "wb") as fh:
            fh.write(pdf_bytes)

    Chained with convertToHTML::

        from tools.convert import convertToHTML, convertToPDF

        html = convertToHTML().run(input=markdown_text, options={"format": "html"})
        convertToPDF().run(input=html, options={"output_path": "document.pdf"})
    """

    name        = "convertToPDF"
    description = (
        "Render an HTML document to PDF.  Accepts HTML as a string or as "
        "a path to an HTML file.  Returns the saved file path when "
        "output_path is provided, otherwise returns raw PDF bytes."
    )
    parameters  = {
        "type": "object",
        "properties": {
            "input": {
                "type": "string",
                "description": "HTML content as a string, or path to an existing HTML file.",
            },
            "options": {
                "type": "object",
                "description": "Rendering options.",
                "properties": {
                    "output_path": {
                        "type": "string",
                        "description": "Destination file path for the PDF. Returns raw bytes when omitted.",
                    },
                    "base_url": {
                        "type": "string",
                        "description": "Base URL for resolving relative asset references in the HTML.",
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
    ) -> "str | bytes":
        """
        Render *input* HTML to PDF.

        Parameters
        ----------
        input : str
            HTML content as a string, or a file path to an existing HTML
            file.  The tool detects which form is provided automatically:
            if *input* is a readable file path it is treated as a file;
            otherwise it is treated as an HTML string.
        options : dict | None, optional
            Rendering options:
              ``output_path`` — if provided, the PDF is written to this path
                               and the path string is returned.  Parent
                               directories are created if needed.  When
                               omitted the raw PDF bytes are returned.
              ``base_url``    — base URL or directory for resolving relative
                               asset references (CSS, images, fonts) inside
                               the HTML.

        Returns
        -------
        str
            The absolute *output_path* when a destination file was given.
        bytes
            Raw PDF bytes when *output_path* is ``None``.

        Raises
        ------
        ImportError
            If the ``weasyprint`` package is not installed.
        FileNotFoundError
            If *input* looks like a file path but the file does not exist.
        """
        import sys
        import os

        opts        = options or {}
        output_path = opts.get("output_path")
        base_url    = opts.get("base_url")

        # The old tool file was named 'weasyprint.py', which shadowed the
        # installed library when tools/ was on sys.path.  This file is now
        # named 'to_pdf.py' so the shadowing issue no longer exists, but we
        # keep this defensive sys.path manipulation in case any caller has
        # added a directory containing a file named 'weasyprint.py' to the path.
        _this_dir = os.path.normcase(os.path.abspath(os.path.dirname(__file__)))
        _saved    = sys.path[:]
        sys.path  = [
            p for p in sys.path
            if os.path.normcase(os.path.abspath(p)) != _this_dir
        ]
        try:
            from weasyprint import HTML
        except ImportError:
            sys.path[:] = _saved
            raise ImportError(
                "weasyprint is required for convertToPDF.  "
                "Install it with:  pip install weasyprint"
            )

        try:
            # Determine whether input is a file path or an HTML string
            html_obj = self._build_html(HTML, input, base_url)

            if output_path:
                parent = os.path.dirname(os.path.abspath(output_path))
                os.makedirs(parent, exist_ok=True)
                html_obj.write_pdf(output_path)
                return os.path.abspath(output_path)

            return html_obj.write_pdf()
        finally:
            sys.path[:] = _saved

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_html(html_cls, source: str, base_url: "str | None"):
        """
        Return a ``weasyprint.HTML`` instance from *source*.

        If *source* is an existing file path, use ``filename=``.
        Otherwise treat it as an HTML string and use ``string=``.
        """
        kwargs = {}
        if base_url is not None:
            kwargs["base_url"] = base_url

        # Treat as file path if it resolves to an existing file
        if os.path.isfile(source):
            return html_cls(filename=source, **kwargs)

        # Treat as HTML string
        return html_cls(string=source, **kwargs)

    def __repr__(self) -> str:
        return f"convertToPDF(name={self.name!r})"
