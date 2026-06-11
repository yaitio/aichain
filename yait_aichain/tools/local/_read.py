"""
tools.local._read — LocalReadTool
===================================

Read a file's contents, optionally restricted to a line range.
All paths are sandboxed to the configured root directory.
"""

from __future__ import annotations

import os

from .._base import Tool
from ._base  import SandboxedTool


class LocalReadTool(Tool, SandboxedTool):
    """
    Read a file's contents.

    Parameters
    ----------
    root_dir : str | None
        Sandbox root. Defaults to ``os.getcwd()``.

    ``run()`` options
    -----------------
    ``start_line`` int — first line to return, 1-indexed (default: 1)
    ``end_line``   int — last line to return inclusive (default: all)
    ``encoding``   str — file encoding (default ``"utf-8"``)

    Returns
    -------
    str
        File contents, prefixed with a header showing path and line range.
    """

    name        = "local_read"
    description = (
        "Read the contents of a local file. "
        "Supports reading a specific line range for large files."
    )
    parameters  = {
        "type": "object",
        "properties": {
            "input": {
                "type":        "string",
                "description": "Path to the file to read (relative to sandbox root).",
            },
            "options": {
                "type":        "object",
                "description": "Read options.",
                "properties": {
                    "start_line": {
                        "type":        "integer",
                        "description": "First line to return, 1-indexed (default 1).",
                    },
                    "end_line": {
                        "type":        "integer",
                        "description": "Last line to return, inclusive (default: all).",
                    },
                    "encoding": {
                        "type":        "string",
                        "description": "File encoding (default utf-8).",
                    },
                },
            },
        },
        "required": ["input"],
    }

    def __init__(self, root_dir: str | None = None) -> None:
        Tool.__init__(self)
        SandboxedTool.__init__(self, root_dir)

    def run(self, input: str, options: dict | None = None) -> str:
        opts       = options or {}
        start_line = int(opts.get("start_line", 1))
        end_line   = opts.get("end_line")
        encoding   = opts.get("encoding", "utf-8")

        path = self._resolve(input)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {input!r}")

        with open(path, encoding=encoding, errors="replace") as fh:
            lines = fh.readlines()

        total = len(lines)
        start = max(0, start_line - 1)
        end   = (int(end_line) if end_line is not None else total)
        chunk = lines[start:end]

        header = (
            f"── {self._rel(path)}  "
            f"(lines {start + 1}–{start + len(chunk)} of {total}) "
            f"──\n"
        )
        return header + "".join(chunk)


def localRead(root_dir: str | None = None) -> LocalReadTool:
    """Return a :class:`LocalReadTool` sandboxed to *root_dir*."""
    return LocalReadTool(root_dir)
