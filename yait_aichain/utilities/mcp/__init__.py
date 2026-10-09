"""
utilities.mcp
=============

``MCP`` — an MCP server as a utility. Its tools are the ones the server
declares, each with the name, description and schema the server gives it.

::

    deepwiki = MCP("https://mcp.deepwiki.com/mcp")
    deepwiki.names                                    # ['ask_wiki_question', ...]
    deepwiki.tool("ask_wiki_question").run(repoName="yaitio/aichain", question="…")
    deepwiki.call("read_wiki_structure", repoName="yaitio/aichain")   # the same, for code
    Agent(model, tools=deepwiki.tools)

``MCPTool`` and ``MCPTools(server)`` — one tool, or every tool of a server —
are what it is built on, and still work.
"""

from __future__ import annotations

from .._base import Utility
from ._tools import DEFAULT_MCP_SETUP_TIMEOUT, MCPTool, MCPTools


class MCP(Utility):
    """
    An MCP server. ``tools`` and ``tool(name)`` give its tools; ``call(name,
    **arguments)`` runs one from code. The server is asked for its tools on
    first use, not here, so making the utility costs nothing.
    """

    def __init__(self, server, *, filter: "list | None" = None, headers: "dict | None" = None,
                 env: "dict | None" = None, cwd: "str | None" = None, transport: "str | None" = None,
                 timeout: "float | None" = DEFAULT_MCP_SETUP_TIMEOUT) -> None:
        self.server = server
        self._settings = dict(filter=filter, headers=headers, env=env, cwd=cwd, transport=transport,
                              timeout=timeout)
        self._tools: "list | None" = None

    @property
    def tools(self) -> list:
        if self._tools is None:
            self._tools = MCPTools(self.server, **self._settings)
        return list(self._tools)

    @property
    def names(self) -> list:
        """The names of the server's tools."""
        return [t.name for t in self.tools]

    @property
    def TOOLS(self) -> tuple:                         # for the agent's refusal message
        return tuple(self.names) if self._tools is not None else ()

    def tool(self, name: str) -> MCPTool:
        """The server's tool *name*, or a KeyError naming the ones it has."""
        for t in self.tools:
            if t.name == name:
                return t
        raise KeyError(f"the server has no tool {name!r}; it has {self.names}")

    def call(self, name: str, **arguments):
        """Run the server's tool *name* with *arguments*; its result, or the error it raised."""
        return self.tool(name).run(**arguments)

    def __repr__(self) -> str:
        return f"MCP({self.server!r})"


__all__ = ["MCP", "MCPTool", "MCPTools"]
