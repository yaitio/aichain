"""utilities.shell._utility — ``Shell``, a folder as a utility."""

from __future__ import annotations

from .._base import Utility


class Shell(Utility):
    """
    A folder: ``browse``, ``read``, ``write`` and ``run`` are its tools, each
    confined to it, each given to an agent on its own.

    ::

        sh = Shell("projects/demo")
        sh.read.run("README.md", {"start_line": 1, "end_line": 40})
        Agent(model, tools=[sh.browse, sh.read])        # reads, and cannot write or run anything
        Agent(model, tools=sh.tools)                    # all four
        Agent(model, tools=[*Shell("a", prefix="a_").tools, *Shell("b", prefix="b_").tools])

    """

    TOOLS = ("browse", "read", "write", "run")

    def __init__(self, folder: str | None = None, *, prefix: str = "") -> None:
        from . import LocalBrowseTool, LocalReadTool, LocalRunTool, LocalWriteTool
        from ...tools._callable import named
        self.browse = LocalBrowseTool(folder)
        self.read = LocalReadTool(folder)
        self.write = LocalWriteTool(folder)
        self.run = LocalRunTool(folder)
        self.folder = self.read.root
        # Two folders in one agent: `Shell(a, prefix="src_")` names its tools
        # src_local_read and so on, where the two sets collided.
        if prefix:
            for t in (self.browse, self.read, self.write, self.run):
                named(t, prefix + t.name)
