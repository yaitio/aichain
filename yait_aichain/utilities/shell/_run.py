"""
utilities.shell._run — LocalRunTool
===================================

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
import re
import shlex
import signal
import subprocess
import sys

from ...tools._base import Tool
from ._base  import SandboxedTool
from ...tools._security import clip

#: The most of stdout, and of stderr, a run returns.
MAX_OUTPUT_CHARS = 50_000


_DEFAULT_TIMEOUT = 30   # seconds
#: The longest a model may ask a command to run. Unbounded, `10**9` was an
#: OverflowError and a large one a process that outlived the agent.
MAX_TIMEOUT = 600

#: Environment names a child does not inherit: the provider keys in this
#: process's environment were the child's, and `env` printed them to the model.
_SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH", re.IGNORECASE)


def _child_env() -> dict:
    return {k: v for k, v in os.environ.items()
            if not _SECRET_NAME.search(k) and not k.startswith("AWS_")}


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
    risk        = "privileged"   # runs any command with the host process's rights — the folder is only where it starts
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

    def run(self, input: str, options: dict | None = None) -> dict:
        opts    = options or {}
        try:
            timeout = int(opts.get("timeout", _DEFAULT_TIMEOUT))
        except (TypeError, ValueError):
            raise ValueError(f"timeout is a whole number of seconds, not {opts.get('timeout')!r}") from None
        timeout = max(1, min(timeout, MAX_TIMEOUT))
        cwd_rel = opts.get("cwd", ".")
        force_python = bool(opts.get("python", False))

        # Resolve and validate working directory
        cwd = self._resolve(cwd_rel)
        if not os.path.isdir(cwd):
            raise ValueError(f"Working directory not found: {cwd_rel!r}")

        # Build command
        command = input.strip()
        # The interpreter goes in front only when asked, or when the command
        # *starts* with a script: "python x.py", "pytest t.py" and "cat m.py"
        # all end in .py and were rewritten to "<python> python x.py".
        first = command.split(maxsplit=1)[0] if command else ""
        if force_python or first.endswith(".py"):
            command = f"{shlex.quote(sys.executable)} {command}"

        # Its own session, so a timeout kills the whole group — `sleep 99 &`
        # and every stage of a pipeline outlived the shell that was killed —
        # no stdin to wait on, and an environment without this process's keys.
        proc = subprocess.Popen(
            command, shell=True, cwd=cwd, env=_child_env(),
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True)
        timed_out = False
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (AttributeError, ProcessLookupError, PermissionError):
                proc.kill()
            out, err = proc.communicate()       # what it printed before, kept
        # Decoded leniently: one invalid byte raised after the command's
        # effects had happened, and the result of a command that ran was lost.
        stdout = out.decode("utf-8", errors="replace")
        stderr = err.decode("utf-8", errors="replace")
        if timed_out:
            stderr += f"\n[timeout] Process group killed after {timeout}s."

        # Head and tail of each: a command that prints a megabyte is a
        # model's whole context, and the end is usually where the error is.
        return {
            "stdout":     clip(stdout, MAX_OUTPUT_CHARS),
            "stderr":     clip(stderr, MAX_OUTPUT_CHARS),
            "returncode": -1 if timed_out else proc.returncode,
            "command":    command,
        }


