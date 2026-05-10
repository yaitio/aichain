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
    from tools.convert import convertToMD

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

import os
from typing import Any

from .._base import Tool


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

    Examples
    --------
    Basic usage (no LLM)::

        from tools.convert import convertToMD

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
    ) -> None:
        self._llm_client      = llm_client
        self._llm_model       = llm_model
        self._enable_builtins = enable_builtins
        self._enable_plugins  = enable_plugins
        self._md              = None   # lazily initialised

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

        md     = self._get_converter()
        result = md.convert(input)
        text   = result.text_content

        if output_path:
            parent = os.path.dirname(os.path.abspath(output_path))
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

        self._md = MarkItDown(**kwargs)
        return self._md

    def __repr__(self) -> str:
        extras = []
        if self._llm_client is not None:
            extras.append(f"llm_model={self._llm_model!r}")
        extra_str = ", " + ", ".join(extras) if extras else ""
        return f"convertToMD(name={self.name!r}{extra_str})"
