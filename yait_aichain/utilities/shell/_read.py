"""
utilities.shell._read — LocalReadTool
=====================================

Read a file's contents, optionally restricted to a line range.
All paths are sandboxed to the configured root directory.
"""

from __future__ import annotations

import os

from ...tools._base import Tool
from ._base  import SandboxedTool



#: The most one read returns.
MAX_READ_BYTES = 200_000

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
    risk        = "read"   # looks, changes nothing
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

    def run(self, input: str, options: dict | None = None) -> str:
        opts       = options or {}
        start_line = int(opts.get("start_line", 1))
        end_line   = opts.get("end_line")
        encoding   = opts.get("encoding", "utf-8")

        path = self._resolve(input)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {input!r}")

        # Streamed, and bounded: a 2 GB log read whole for forty lines was the
        # memory of the process, and a model asking for "all" of it was its
        # context. Lines past MAX_READ_BYTES are counted, not kept.
        start = max(0, start_line - 1)
        end   = int(end_line) if end_line is not None else None
        chunk, kept, total, clipped = [], 0, 0, False
        with open(path, encoding=encoding, errors="replace") as fh:
            for i, line in enumerate(fh):
                total = i + 1
                if i < start or (end is not None and i >= end):
                    continue
                if kept + len(line) > MAX_READ_BYTES:
                    clipped = True
                    continue
                chunk.append(line)
                kept += len(line)
        if clipped:
            chunk.append(f"\n… [stopped at {MAX_READ_BYTES:,} characters; ask for a narrower "
                         "start_line/end_line] …\n")

        header = (
            f"── {self._rel(path)}  "
            f"(lines {start + 1}–{start + len(chunk)} of {total}) "
            f"──\n"
        )
        return header + "".join(chunk)


