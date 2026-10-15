"""``MCP``: an MCP server as a utility, its tools the ones the server declares,
their risk classes from the server's annotations (``privileged`` where it says
nothing) unless ``risk=`` says otherwise. See docs/utilities/index.md.
"""

from __future__ import annotations

from .._base import Utility
from ._tools import DEFAULT_MCP_CALL_TIMEOUT, DEFAULT_MCP_SETUP_TIMEOUT, MCPTool, _redacted, _Server, discover, risk_from_annotations


def _remote(tool) -> str:
    return getattr(tool, "_remote", None) or tool.name


class MCP(Utility):
    """
    An MCP server. ``tools`` and ``tool(name)`` give its tools; ``call(name,
    **arguments)`` runs one from code. The server is asked for its tools on
    first use, not here, so making the utility costs nothing.
    """

    def __init__(self, server, *, filter: list | None = None, headers: dict | None = None,
                 env: dict | None = None, cwd: str | None = None, transport: str | None = None,
                 connect_timeout: float | None = DEFAULT_MCP_SETUP_TIMEOUT,
                 call_timeout: float | None = DEFAULT_MCP_CALL_TIMEOUT, risk: dict | None = None,
                 prefix: str = "", trust_annotations: bool = True) -> None:
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
        self._prefix = prefix
        self._trust = trust_annotations

    @property
    def tools(self) -> list:
        if self._tools is None:
            tools = discover(self._conn, filter=self._filter, timeout=self._connect_timeout)
            missing = sorted(set(self._risk) - {_remote(t) for t in tools})
            if missing:
                raise KeyError(f"risk= names {missing}, which the server does not have; "
                               f"it has {[_remote(t) for t in tools]}")
            for t in tools:
                # The server's own word lowers nothing when it is not trusted:
                # every tool is privileged unless `risk=` says otherwise.
                base = t.risk if self._trust else "privileged"
                t.risk = self._risk.get(_remote(t), base)
                if self._prefix:
                    t.name = self._prefix + t.name
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
            if name in (t.name, _remote(t)):
                return t
        raise KeyError(f"the server has no tool {name!r}; it has {self.names}")

    def call(self, name: str, /, **arguments):
        """Run the server's tool *name* with *arguments*; its result, or the error it raised."""
        return self.tool(name).run(**arguments)

    def __repr__(self) -> str:
        return f"MCP({_redacted(self.server)!r})"


__all__ = ["MCP", "MCPTool", "risk_from_annotations"]
