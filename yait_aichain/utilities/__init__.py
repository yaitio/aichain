"""Utilities: what holds a connection or a place — a vector database, an MCP
server, an HTTP API, a folder — and gives an agent the tools it should have.
Code calls a utility's methods; its tools are called with ``.run``.
See docs/utilities/index.md.
"""

from ._base   import Utility
from .vectors import Vectors
from .mcp     import MCP
from .rest    import REST
from .shell   import Shell

__all__ = ["Utility", "Vectors", "MCP", "REST", "Shell"]
