"""What a utility is: an object whose methods code calls and whose tools — the
attributes ``TOOLS`` names, called with ``.run`` — an agent may be given. An
agent given a method is refused by name. See docs/utilities/index.md.
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
