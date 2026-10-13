"""
utilities.shell._browse — LocalBrowseTool
=========================================

List the contents of a directory as a formatted tree.

Supports depth control and glob-style filename filtering.
All paths are sandboxed to the configured root directory.
"""

from __future__ import annotations

import fnmatch
import os

from ...tools._base import Tool
from ._base  import SandboxedTool



#: The most entries one listing returns.
MAX_ENTRIES = 2_000

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
    risk        = "read"   # looks, changes nothing
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

    def run(self, input: str = ".", options: dict | None = None) -> str:
        opts        = options or {}
        max_depth   = int(opts.get("depth", 2))
        pattern     = opts.get("pattern", "*")
        show_hidden = bool(opts.get("show_hidden", False))

        target = self._resolve(input or ".")
        if not os.path.isdir(target):
            raise ValueError(f"Not a directory: {input!r}")

        lines = [f"📁 {self._rel(target)}/"]

        def walk(path: str, prefix: str, depth: int) -> None:
            if depth >= max_depth:
                return
            try:
                entries = sorted(os.scandir(path), key=lambda e: (not e.is_dir(), e.name))
            except PermissionError:
                lines.append(f"{prefix}  [permission denied]")
                return
            visible = [e for e in entries
                       if (show_hidden or not e.name.startswith("."))
                       and (e.is_dir() or fnmatch.fnmatch(e.name, pattern))]
            for i, entry in enumerate(visible):
                if len(lines) > MAX_ENTRIES:
                    lines.append(f"{prefix}… [{len(visible) - i} more; stopped at {MAX_ENTRIES} entries]")
                    return
                last = i == len(visible) - 1
                branch = "└── " if last else "├── "
                if entry.is_dir(follow_symlinks=False):
                    lines.append(f"{prefix}{branch}📁 {entry.name}/")
                    walk(entry.path, prefix + ("    " if last else "│   "), depth + 1)
                else:
                    # Not followed: a dangling link raised and ended the listing.
                    try:
                        size = self._fmt_size(entry.stat(follow_symlinks=False).st_size)
                    except OSError:
                        size = "unreadable"
                    lines.append(f"{prefix}{branch}{entry.name}  ({size})")

        walk(target, "", 0)
        if len(lines) > MAX_ENTRIES + 1:
            lines = lines[:MAX_ENTRIES + 1] + [f"… [stopped at {MAX_ENTRIES} entries; narrow depth or pattern]"]
        # Not the host's absolute path: where the folder lives is not the
        # model's business, and every path it gives is relative to it anyway.
        lines.append("\nRoot: .")
        return "\n".join(lines)

    @staticmethod
    def _fmt_size(n: int) -> str:
        for unit in ("B", "KB", "MB", "GB"):
            if n < 1024:
                return f"{n:.0f} {unit}"
            n /= 1024
        return f"{n:.1f} GB"


