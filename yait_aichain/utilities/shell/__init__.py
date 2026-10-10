"""
utilities.shell
===============

``Shell(folder)`` — a folder as a utility: browse, read, write and execute
files within it.

Its four tools share the same folder. Any path that resolves above it
raises ``PermissionError``.

Tools
-----
``sh.browse``  → LocalBrowseTool
    List directory contents as a tree.

``sh.read``    → LocalReadTool
    Read file contents (with optional line range).

``sh.write``   → LocalWriteTool
    Create or overwrite a file.

``sh.run``     → LocalRunTool
    Execute a shell command or Python file.

``sh.tools`` gives all four. The folder defaults to ``os.getcwd()`` when
omitted.

Examples
--------
Give an agent full local filesystem access in the current directory::

    from yait_aichain import Agent, Model
    from yait_aichain.agent import step_count
    from yait_aichain.utilities import Shell

    agent = Agent(Model("claude-haiku-5-5"), tools=Shell().tools,    # all four, confined to cwd
                  stop_when=[step_count(20)])
    result = agent.run("Write a Python script that prints the Fibonacci sequence, save it as fib.py, and run it.")

Pin to a specific directory, read-only::

    sh = Shell("/Users/me/projects/myapp")
    agent = Agent(Model("claude-haiku-5-5"), tools=[sh.browse, sh.read],
                  stop_when=[step_count(20)])
    result = agent.run("Find all TODO comments in the codebase and summarise them.")

Use the tools from code::

    sh = Shell("/my/project")
    print(sh.browse.run("src", {"depth": 3, "pattern": "*.py"}))
    print(sh.read.run("src/main.py", {"start_line": 1, "end_line": 50}))
    sh.write.run("# hello world\\n", {"path": "hello.py"})
    result = sh.run.run("hello.py", {"timeout": 10})
    print(result["stdout"])
"""

from ._browse import LocalBrowseTool
from ._read import LocalReadTool
from ._write import LocalWriteTool
from ._run import LocalRunTool

__all__ = ["Shell", "LocalBrowseTool", "LocalReadTool", "LocalWriteTool", "LocalRunTool"]




# ── The utility ────────────────────────────────────────────────────────────
from ._utility import Shell
