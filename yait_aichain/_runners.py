"""
What kind of runner a step or a pool item is — one answer for Chain and Pool.

By ``isinstance``, imported late: the primitives import Chain and Pool, so a
module-level import here would be a cycle. Matching a class name would mistake
a user's own ``Agent`` or another library's ``Tool`` for ours.
"""

from __future__ import annotations


def is_tool(runner) -> bool:
    from .tools._base import Tool
    return isinstance(runner, Tool)


def is_agent(runner) -> bool:
    from .agent import Agent
    return isinstance(runner, Agent)
