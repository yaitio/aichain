"""
tools.local._run — LocalRunTool
=================================

Execute a shell command or Python file. The working directory is set to the
sandbox root (or a sub-directory of it).

.. danger::
    This tool runs **arbitrary commands** through the shell (``shell=True``)
    with the full privileges of the host process. ``root_dir`` only sets the
    *starting directory* — it does NOT confine what the command can read,
    write, execute, or reach over the network (a command can use absolute
    paths, ``..``, ``curl | sh``, etc.). Treat ``LocalRunTool`` as
    arbitrary-code-execution: only expose it to trusted input, and never wire
    it to an agent acting on untrusted instructions without an OS-level
    sandbox (container, seccomp, separate user).
"""

from __future__ import annotations

import os
import subprocess
import sys

from .._base import Tool
from ._base  import SandboxedTool


_DEFAULT_TIMEOUT = 30   # seconds


class LocalRunTool(Tool, SandboxedTool):
    """
    Execute a shell command or Python file.

    Parameters
    ----------
    root_dir : str | None
        Sandbox root and default working directory. Defaults to ``os.getcwd()``.

    ``run()`` input / options
    -------------------------
    input   : str  — shell command or path to a Python file to run
    options:
      ``timeout``  int  — max seconds before the process is killed (default 30)
      ``cwd``      str  — working directory, must be inside sandbox (default: root)
      ``python``   bool — if True, always prepend ``sys.executable`` (default: auto-detect)

    Returns
    -------
    dict
        ``{"stdout": "...", "stderr": "...", "returncode": 0, "command": "..."}``

    Notes
    -----
    If the command is a ``.py`` file path, it is automatically executed
    with the current Python interpreter (``sys.executable``).
    """

    name        = "local_run"
    description = (
        "Execute a shell command or Python file within the sandbox directory. "
        "Returns stdout, stderr, and the return code. "
        "Use this to run scripts, tests, or any CLI tool."
    )
    parameters  = {
        "type": "object",
        "properties": {
            "input": {
                "type":        "string",
                "description": (
                    "Shell command to execute, e.g. 'ls -la' or 'python script.py'. "
                    "If a .py file path is given, it runs with the current Python."
                ),
            },
            "options": {
                "type":        "object",
                "description": "Execution options.",
                "properties": {
                    "timeout": {
                        "type":        "integer",
                        "description": f"Max seconds before the process is killed (default {_DEFAULT_TIMEOUT}).",
                    },
                    "cwd": {
                        "type":        "string",
                        "description": "Working directory (relative to sandbox root, default: root).",
                    },
                    "python": {
                        "type":        "boolean",
                        "description": "Force execution with the current Python interpreter.",
                    },
                },
            },
        },
        "required": ["input"],
    }

    def __init__(self, root_dir: str | None = None) -> None:
        Tool.__init__(self)
        SandboxedTool.__init__(self, root_dir)

    def run(self, input: str, options: dict | None = None) -> dict:
        opts    = options or {}
        timeout = int(opts.get("timeout", _DEFAULT_TIMEOUT))
        cwd_rel = opts.get("cwd", ".")
        force_python = bool(opts.get("python", False))

        # Resolve and validate working directory
        cwd = self._resolve(cwd_rel)
        if not os.path.isdir(cwd):
            raise ValueError(f"Working directory not found: {cwd_rel!r}")

        # Build command
        command = input.strip()
        if force_python or command.endswith(".py"):
            # If it looks like a bare .py path (no spaces), validate it
            parts = command.split()
            if len(parts) == 1 and parts[0].endswith(".py"):
                self._resolve(parts[0])   # sandbox check
            command = f"{sys.executable} {command}"

        # Execute
        try:
            result = subprocess.run(
                command,
                shell      = True,
                cwd        = cwd,
                capture_output = True,
                text       = True,
                timeout    = timeout,
            )
        except subprocess.TimeoutExpired:
            return {
                "stdout":     "",
                "stderr":     f"[timeout] Process killed after {timeout}s.",
                "returncode": -1,
                "command":    command,
            }

        return {
            "stdout":     result.stdout,
            "stderr":     result.stderr,
            "returncode": result.returncode,
            "command":    command,
        }


def localRun(root_dir: str | None = None) -> LocalRunTool:
    """Return a :class:`LocalRunTool` sandboxed to *root_dir*."""
    return LocalRunTool(root_dir)
