"""
tools.local
===========

Local filesystem tools — browse, read, write, and execute files
within a sandboxed directory.

All four tools share the same ``root_dir`` sandbox. Any path that
resolves above the root raises ``PermissionError``.

Public API
----------
``LocalTools(root_dir)``
    Convenience factory — returns all four tools pre-configured
    with the same sandbox root.  Pass the result directly to an Agent.

``localBrowse(root_dir)``  → LocalBrowseTool
    List directory contents as a tree.

``localRead(root_dir)``    → LocalReadTool
    Read file contents (with optional line range).

``localWrite(root_dir)``   → LocalWriteTool
    Create or overwrite a file.

``localRun(root_dir)``     → LocalRunTool
    Execute a shell command or Python file.

``root_dir`` defaults to ``os.getcwd()`` when omitted.

Examples
--------
Give an agent full local filesystem access in the current directory::

    from yait_aichain import Agent, Model
    from yait_aichain.agent import step_count
    from yait_aichain.utilities import Shell

    agent = Agent(Model("claude-haiku-5-5"), tools=Shell().tools,    # all four, confined to cwd
                  stop_when=[step_count(20)])
    result = agent.run("Write a Python script that prints the Fibonacci sequence, save it as fib.py, and run it.")

Pin to a specific directory::

    sh = Shell("/Users/me/projects/myapp")
    agent = Agent(Model("claude-haiku-5-5"), tools=[sh.browse, sh.read],   # read-only
                  stop_when=[step_count(20)])
    result = agent.run("Find all TODO comments in the codebase and summarise them.")

Use tools individually::

    from yait_aichain.tools.local import localBrowse, localRead, localWrite, localRun

    browse = localBrowse("/my/project")
    print(browse.run("src", {"depth": 3, "pattern": "*.py"}))

    read = localRead("/my/project")
    print(read.run("src/main.py", {"start_line": 1, "end_line": 50}))

    write = localWrite("/my/project")
    write.run("# hello world\\n", {"path": "hello.py"})

    run = localRun("/my/project")
    result = run.run("hello.py", {"timeout": 10})
    print(result["stdout"])
"""

from ._browse import LocalBrowseTool, localBrowse
from ._read   import LocalReadTool,   localRead
from ._write  import LocalWriteTool,  localWrite
from ._run    import LocalRunTool,    localRun

__all__ = [
    "LocalTools",
    "localBrowse", "LocalBrowseTool",
    "localRead",   "LocalReadTool",
    "localWrite",  "LocalWriteTool",
    "localRun",    "LocalRunTool",
]


def LocalTools(root_dir: str | None = None) -> list:
    """
    Return all four local tools pre-configured with the same sandbox root.

    Parameters
    ----------
    root_dir : str | None
        Sandbox root directory.  Defaults to ``os.getcwd()``.

    Returns
    -------
    list
        ``[LocalBrowseTool, LocalReadTool, LocalWriteTool, LocalRunTool]``

    Example
    -------
    ::

        from yait_aichain import Agent, Model
        from yait_aichain.tools import LocalTools

        agent = Agent(Model("claude-haiku-5-5"), tools=LocalTools("/my/project"))
        # the same as tools=Shell("/my/project").tools, which new code uses
    """
    return [
        localBrowse(root_dir),
        localRead(root_dir),
        localWrite(root_dir),
        localRun(root_dir),
    ]


# ── The utility (3.6.0) ──────────────────────────────────────────────────────
from ._utility import Shell
