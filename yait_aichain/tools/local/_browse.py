"""
tools.local._browse — LocalBrowseTool
=======================================

List the contents of a directory as a formatted tree.

Supports depth control and glob-style filename filtering.
All paths are sandboxed to the configured root directory.
"""

from __future__ import annotations

import fnmatch
import os

from .._base import Tool
from ._base  import SandboxedTool


class LocalBrowseTool(Tool, SandboxedTool):
    """
    List directory contents as a tree.

    Parameters
    ----------
    root_dir : str | None
        Sandbox root. Defaults to ``os.getcwd()``.

    ``run()`` options
    -----------------
    ``depth``       int  — how many levels to descend (default 2)
    ``pattern``     str  — glob filter, e.g. ``"*.py"`` (default: all files)
    ``show_hidden`` bool — include dotfiles (default False)

    Returns
    -------
    str
        Directory tree as a formatted string.
    """

    name        = "local_browse"
    description = (
        "List the contents of a local directory as a tree. "
        "Use this to explore the filesystem and understand project structure."
    )
    parameters  = {
        "type": "object",
        "properties": {
            "input": {
                "type":        "string",
                "description": "Directory path to list. Relative to sandbox root. Default: root itself.",
            },
            "options": {
                "type":        "object",
                "description": "Browse options.",
                "properties": {
                    "depth": {
                        "type":        "integer",
                        "description": "How many directory levels to show (default 2).",
                    },
                    "pattern": {
                        "type":        "string",
                        "description": "Glob pattern to filter files, e.g. '*.py'.",
                    },
                    "show_hidden": {
                        "type":        "boolean",
                        "description": "Include hidden files and directories (default false).",
                    },
                },
            },
        },
        "required": ["input"],
    }

    def __init__(self, root_dir: str | None = None) -> None:
        Tool.__init__(self)
        SandboxedTool.__init__(self, root_dir)

    def run(self, input: str = ".", options: dict | None = None) -> str:
        opts        = options or {}
        max_depth   = int(opts.get("depth", 2))
        pattern     = opts.get("pattern", "*")
        show_hidden = bool(opts.get("show_hidden", False))

        target = self._resolve(input or ".")
        if not os.path.isdir(target):
            raise ValueError(f"Not a directory: {input!r}")

        lines = [f"📁 {self._rel(target)}/"]
        self._tree(target, "", max_depth, 0, pattern, show_hidden, lines)
        lines.append(f"\nRoot: {self.root}")
        return "\n".join(lines)

    def _tree(
        self,
        path:        str,
        prefix:      str,
        max_depth:   int,
        depth:       int,
        pattern:     str,
        show_hidden: bool,
        lines:       list[str],
    ) -> None:
        if depth >= max_depth:
            return

        try:
            entries = sorted(os.scandir(path), key=lambda e: (not e.is_dir(), e.name))
        except PermissionError:
            lines.append(f"{prefix}  [permission denied]")
            return

        visible = [
            e for e in entries
            if (show_hidden or not e.name.startswith("."))
            and (e.is_dir() or fnmatch.fnmatch(e.name, pattern))
        ]

        for i, entry in enumerate(visible):
            is_last    = (i == len(visible) - 1)
            connector  = "└── " if is_last else "├── "
            extension  = "    " if is_last else "│   "

            if entry.is_dir(follow_symlinks=False):
                lines.append(f"{prefix}{connector}📁 {entry.name}/")
                self._tree(
                    entry.path, prefix + extension,
                    max_depth, depth + 1, pattern, show_hidden, lines,
                )
            else:
                size = self._fmt_size(entry.stat().st_size)
                lines.append(f"{prefix}{connector}{entry.name}  ({size})")

    @staticmethod
    def _fmt_size(n: int) -> str:
        for unit in ("B", "KB", "MB", "GB"):
            if n < 1024:
                return f"{n:.0f} {unit}"
            n /= 1024
        return f"{n:.1f} GB"


def localBrowse(root_dir: str | None = None) -> LocalBrowseTool:
    """Return a :class:`LocalBrowseTool` sandboxed to *root_dir*."""
    return LocalBrowseTool(root_dir)
