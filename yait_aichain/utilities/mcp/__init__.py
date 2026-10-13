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

Each tool is an ``MCPTool``. Two settings: ``connect_timeout`` bounds reaching
the server and listing its tools (60 s), ``call_timeout`` one call of a tool
(300 s); ``None`` waits indefinitely.

Each tool's risk class — what a permission policy reads — comes from the
server's annotations, and is ``privileged`` where the server says nothing:
another server's tool is unknown here, and the unknown goes to a person.
``risk={"tool_name": "read"}`` puts your word over the server's::

    MCP("https://mcp.deepwiki.com/mcp", risk={"ask_wiki_question": "read"})
"""

from __future__ import annotations

from .._base import Utility
from ._tools import DEFAULT_MCP_CALL_TIMEOUT, DEFAULT_MCP_SETUP_TIMEOUT, MCPTool, _redacted, _Server, discover, risk_from_annotations


class MCP(Utility):
    """
    An MCP server. ``tools`` and ``tool(name)`` give its tools; ``call(name,
    **arguments)`` runs one from code. The server is asked for its tools on
    first use, not here, so making the utility costs nothing.
    """

    def __init__(self, server, *, filter: list | None = None, headers: dict | None = None,
                 env: dict | None = None, cwd: str | None = None, transport: str | None = None,
                 connect_timeout: float | None = DEFAULT_MCP_SETUP_TIMEOUT,
                 call_timeout: float | None = DEFAULT_MCP_CALL_TIMEOUT, risk: dict | None = None) -> None:
        # Two timeouts, two names: reaching the server and listing its tools,
        # and one call of a tool.
        self.server = server
        self._conn = _Server(server, headers=headers, env=env, cwd=cwd, transport=transport,
                             call_timeout=call_timeout)
        if isinstance(filter, str) or not all(isinstance(n, str) for n in filter or ()):
            raise TypeError(f"MCP filter is a list of tool names, not {filter!r}")
        self._filter = None if filter is None else list(filter)
        self._connect_timeout = connect_timeout
        self._tools: list | None = None
        from ...tools._permissions import RISK_CLASSES
        bad = {n: r for n, r in (risk or {}).items() if r not in RISK_CLASSES}
        if bad:
            raise ValueError(f"MCP risk: {bad} — each must be one of {sorted(RISK_CLASSES)}")
        self._risk = dict(risk or {})

    @property
    def tools(self) -> list:
        if self._tools is None:
            tools = discover(self._conn, filter=self._filter, timeout=self._connect_timeout)
            missing = sorted(set(self._risk) - {t.name for t in tools})
            if missing:
                raise KeyError(f"risk= names {missing}, which the server does not have; "
                               f"it has {[t.name for t in tools]}")
            for t in tools:
                t.risk = self._risk.get(t.name, t.risk)
            self._tools = tools
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

    def call(self, name: str, /, **arguments):
        """Run the server's tool *name* with *arguments*; its result, or the error it raised."""
        return self.tool(name).run(**arguments)

    def __repr__(self) -> str:
        return f"MCP({_redacted(self.server)!r})"


__all__ = ["MCP", "MCPTool", "risk_from_annotations"]
