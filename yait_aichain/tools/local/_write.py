"""
tools.local._write — LocalWriteTool
=====================================

Create or overwrite a file within the sandbox.
Parent directories are created automatically.
All paths are sandboxed to the configured root directory.
"""

from __future__ import annotations

import os

from .._base import Tool
from ._base  import SandboxedTool


class LocalWriteTool(Tool, SandboxedTool):
    """
    Write content to a file.

    Parameters
    ----------
    root_dir : str | None
        Sandbox root. Defaults to ``os.getcwd()``.

    ``run()`` input / options
    -------------------------
    input   : str  — content to write
    options:
      ``path``     str  — destination file path (required)
      ``mode``     str  — ``"w"`` (overwrite, default) or ``"a"`` (append)
      ``encoding`` str  — file encoding (default ``"utf-8"``)

    Returns
    -------
    dict
        ``{"path": "...", "bytes": N, "mode": "w|a"}``
    """

    name        = "local_write"
    description = (
        "Write content to a local file. Creates the file (and any parent "
        "directories) if they do not exist. Can overwrite or append."
    )
    parameters  = {
        "type": "object",
        "properties": {
            "input": {
                "type":        "string",
                "description": "Content to write to the file.",
            },
            "options": {
                "type":        "object",
                "description": "Write options.",
                "properties": {
                    "path": {
                        "type":        "string",
                        "description": "Destination file path (relative to sandbox root).",
                    },
                    "mode": {
                        "type":        "string",
                        "enum":        ["w", "a"],
                        "description": "'w' to overwrite (default), 'a' to append.",
                    },
                    "encoding": {
                        "type":        "string",
                        "description": "File encoding (default utf-8).",
                    },
                },
                "required": ["path"],
            },
        },
        "required": ["input"],
    }

    def __init__(self, root_dir: str | None = None) -> None:
        Tool.__init__(self)
        SandboxedTool.__init__(self, root_dir)

    def run(self, input: str, options: dict | None = None) -> dict:
        opts     = options or {}
        rel_path = opts.get("path")
        if not rel_path:
            raise ValueError("options['path'] is required for local_write.")
        mode     = opts.get("mode", "w")
        encoding = opts.get("encoding", "utf-8")

        path = self._resolve(rel_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, mode=mode, encoding=encoding) as fh:
            fh.write(input)

        n_bytes = os.path.getsize(path)
        return {
            "path":  self._rel(path),
            "bytes": n_bytes,
            "mode":  mode,
        }


def localWrite(root_dir: str | None = None) -> LocalWriteTool:
    """Return a :class:`LocalWriteTool` sandboxed to *root_dir*."""
    return LocalWriteTool(root_dir)
