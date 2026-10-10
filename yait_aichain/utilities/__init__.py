"""
utilities
=========

What holds a connection or a place, and gives an agent the tools it should
have. Code calls a utility's methods; its tools are called with ``.run`` and
handed to an agent one by one — see ``design/tool-convention.md``.

Vectors     a vector database — create, upsert, fetch, delete, count; ``query`` its tool.
MCP         an MCP server — its tools are the ones it declares.
REST        an HTTP API — base URL, auth and headers once; each endpoint a tool.
Shell       a folder — browse, read, write and run, each a tool confined to it.
"""

from ._base   import Utility
from .vectors import Vectors
from .mcp     import MCP
from .rest    import REST
from .shell   import Shell

__all__ = ["Utility", "Vectors", "MCP", "REST", "Shell"]
