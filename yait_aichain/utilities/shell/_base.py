"""The sandbox the Shell tools share: every path is resolved against the root
(symlinks and ``..`` followed by ``realpath``) and one that escapes it raises
``PermissionError``. See docs/utilities/index.md.
"""

from __future__ import annotations

import os

from ...tools._security import within


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

    def _serialise_init_args(self) -> dict:
        """The folder, saved so a reloaded tool does not move to the loader's
        working directory."""
        return {"root_dir": self.root}

    def _resolve(self, path: str) -> str:
        """*path* resolved against the root, absolute and inside it, or
        ``PermissionError``."""
        if os.path.isabs(path):
            resolved = os.path.realpath(path)
        else:
            resolved = os.path.realpath(os.path.join(self.root, path))

        if not within(self.root, resolved):
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
