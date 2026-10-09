"""
utilities._base — what a utility is.

A utility holds a connection or a place — a database, a server, an API, a
folder. Code calls its methods directly; the actions an agent may have are
tools, held as attributes and called with ``.run`` like every tool. Only a tool
has ``.run``, so the code shows which is which:

    kb.upsert(chunks)                                  # a method: code and chain
    kb.query.run("how does caching work?", {"n": 5})   # a tool: an agent can take it

An agent given a utility or one of its methods refuses it by name and lists
the utility's tools (see ``agent._agent._check_tools``).
"""

from __future__ import annotations


class Utility:
    """Base of the utilities. ``TOOLS`` names the attributes that are tools."""

    TOOLS: tuple = ()

    @property
    def tools(self) -> list:
        """The utility's tools, to hand an agent at once: ``Agent(tools=kb.tools)``."""
        return [getattr(self, name) for name in self.TOOLS]

    def __repr__(self) -> str:
        return f"{type(self).__name__}(tools={list(self.TOOLS)})"
