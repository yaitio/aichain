"""
tools.local._base
=================

Shared sandbox enforcement for all local filesystem tools.

Every local tool receives a ``root_dir`` at construction time.
All path operations are validated against it — any attempt to
navigate above the root raises ``PermissionError``.

Path resolution rules
---------------------
1. Relative paths are resolved relative to ``root_dir``.
2. Absolute paths must start with ``root_dir`` (after symlink resolution).
3. Path traversal (``../../``) is caught after ``os.path.realpath()``.

Default ``root_dir``
---------------------
When omitted, ``root_dir`` defaults to ``os.getcwd()`` — the current
working directory at the time the tool is instantiated.
"""

from __future__ import annotations

import os


class SandboxedTool:
    """
    Base class for local filesystem tools that enforce a root directory sandbox.

    Parameters
    ----------
    root_dir : str | None
        Absolute or relative path to the sandbox root.
        Defaults to ``os.getcwd()``.

    Raises
    ------
    PermissionError
        When any resolved path escapes the sandbox root.
    """

    def __init__(self, root_dir: str | None = None) -> None:
        self.root = os.path.realpath(root_dir or os.getcwd())

    def _resolve(self, path: str) -> str:
        """
        Resolve *path* relative to the sandbox root and validate it.

        Returns
        -------
        str
            Absolute resolved path, guaranteed to be inside the sandbox.

        Raises
        ------
        PermissionError
            When the resolved path escapes the sandbox root.
        """
        if os.path.isabs(path):
            resolved = os.path.realpath(path)
        else:
            resolved = os.path.realpath(os.path.join(self.root, path))

        if not resolved.startswith(self.root):
            raise PermissionError(
                f"Path {path!r} resolves to {resolved!r} which is outside "
                f"the sandbox root {self.root!r}."
            )
        return resolved

    def _rel(self, abs_path: str) -> str:
        """Return *abs_path* relative to the sandbox root for display."""
        try:
            return os.path.relpath(abs_path, self.root)
        except ValueError:
            return abs_path
