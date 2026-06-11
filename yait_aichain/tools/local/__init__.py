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

    from tools.local import LocalTools
    from agent import Agent
    from models import Model

    agent = Agent(
        orchestrator = Model("claude-sonnet-4-6"),
        tools        = LocalTools(),          # all 4 tools, sandboxed to cwd
        max_steps    = 20,
    )
    result = agent.run("Write a Python script that prints the Fibonacci sequence, save it as fib.py, and run it.")

Pin to a specific directory::

    tools = LocalTools("/Users/me/projects/myapp")

    agent = Agent(
        orchestrator = Model("claude-sonnet-4-6"),
        tools        = tools,
        max_steps    = 20,
    )
    result = agent.run("Find all TODO comments in the codebase and summarise them.")

Use tools individually::

    from tools.local import localBrowse, localRead, localWrite, localRun

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

        from tools.local import LocalTools
        from agent import Agent
        from models import Model

        agent = Agent(
            orchestrator = Model("claude-sonnet-4-6"),
            tools        = LocalTools("/my/project"),
            max_steps    = 20,
        )
    """
    return [
        localBrowse(root_dir),
        localRead(root_dir),
        localWrite(root_dir),
        localRun(root_dir),
    ]
