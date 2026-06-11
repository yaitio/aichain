"""
tools.mcp — MCPTool, MCPTools
==============================

Integrate any MCP (Model Context Protocol) server as a first-class
aichain tool.  One ``MCPTool`` instance wraps one callable operation on
an MCP server behind the standard ``run(input, options=None)`` interface.
``MCPTools()`` discovers every tool a server exposes and returns them as
a ready-to-use list.

Both primitives work anywhere a Tool is accepted: Chain steps, Agent
tool lists, or direct ``run()`` calls.  No async boilerplate — the
library manages a background event loop internally.

Supported transports
---------------------
+--------------------------------------------------+---------------------+
| ``server`` value                                 | Transport           |
+==================================================+=====================+
| ``"./server.py"``, ``"python server.py ..."``    | STDIO subprocess    |
| ``{"command": "npx", "args": [...]}``            | STDIO subprocess    |
| ``"https://example.com/mcp"``                    | Streamable HTTP     |
| ``"http://localhost:8000/mcp"``                  | Streamable HTTP     |
| ``"https://example.com/sse"`` (URL ends /sse)    | SSE legacy          |
| ``{"url": "https://..."}``                       | HTTP (dict form)    |
+--------------------------------------------------+---------------------+

Quick examples
--------------
::

    from tools import MCPTools, MCPTool

    # ── Discover all tools on a server ─────────────────────────────────
    tools = MCPTools("https://api.example.com/mcp")
    # → [MCPTool("search"), MCPTool("summarise"), ...]

    # ── Use in an Agent ────────────────────────────────────────────────
    from agent import Agent
    from models import Model

    agent = Agent(
        model = Model("claude-sonnet-4-6"),
        tools = MCPTools("https://api.example.com/mcp"),
    )

    # ── Use a single tool in a Chain ───────────────────────────────────
    tools = MCPTools("https://api.example.com/mcp")
    search = next(t for t in tools if t.name == "search")
    # Then use `search` as a Chain step

    # ── Direct call ────────────────────────────────────────────────────
    result = search.run(input={"query": "latest AI research"})

    # ── HTTP with auth header ──────────────────────────────────────────
    tools = MCPTools(
        "https://api.example.com/mcp",
        headers={"Authorization": "Bearer sk-..."},
    )

    # ── STDIO subprocess ───────────────────────────────────────────────
    tools = MCPTools({"command": "python", "args": ["./my_server.py"]})

    # ── npx MCP server with env vars ───────────────────────────────────
    tools = MCPTools(
        {"command": "npx", "args": ["@modelcontextprotocol/server-filesystem", "/data"]},
        env={"READ_ONLY": "true"},
    )

    # ── Filter to specific tools ───────────────────────────────────────
    tools = MCPTools(
        "https://api.example.com/mcp",
        filter=["search", "summarise"],
    )

    # ── Build a single tool directly (skip discovery) ──────────────────
    tool = MCPTool(
        name        = "search_web",
        description = "Search the web for recent information.",
        server      = "https://api.example.com/mcp",
        headers     = {"Authorization": "Bearer sk-..."},
    )
    result = tool.run(input={"query": "MCP protocol"})

Installation
------------
::

    pip install fastmcp
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from ._base import Tool


# ---------------------------------------------------------------------------
# Background event loop — shared singleton
# ---------------------------------------------------------------------------

class _AsyncBridge:
    """
    A persistent daemon thread running a dedicated asyncio event loop.

    All async MCP operations are submitted here via :meth:`run`, which
    blocks the calling thread until the coroutine completes.

    This approach avoids two common problems with ``asyncio.run()``:

    * Calling from inside a running event loop (Jupyter, FastAPI) raises
      ``RuntimeError: This event loop is already running``.
    * Creating a new event loop per call tears down and rebuilds STDIO
      subprocess connections unnecessarily.

    A single background loop handles all concurrent MCPTool calls safely.
    """

    _instance: "_AsyncBridge | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target  = self._run_forever,
            daemon  = True,
            name    = "aichain-mcp-loop",
        )
        self._thread.start()

    def _run_forever(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coro: Any) -> Any:
        """Submit *coro* to the background loop and block until it completes."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()   # re-raises exceptions from the coroutine

    @classmethod
    def get(cls) -> "_AsyncBridge":
        """Return (or create) the shared singleton bridge."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance


def _run_sync(coro: Any) -> Any:
    """Run *coro* synchronously from any context (sync or async)."""
    return _AsyncBridge.get().run(coro)


# ---------------------------------------------------------------------------
# Transport helpers
# ---------------------------------------------------------------------------

def _is_sse(url: str, transport_hint: "str | None") -> bool:
    """Return True when the SSE transport should be preferred."""
    if transport_hint == "sse":
        return True
    if transport_hint in ("http", "streamable"):
        return False
    return url.rstrip("/").endswith("/sse")


def _build_client(
    server:    "str | dict | Any",
    *,
    headers:   "dict | None" = None,
    env:       "dict | None" = None,
    cwd:       "str | None"  = None,
    transport: "str | None"  = None,
) -> Any:
    """
    Construct a ``fastmcp.Client`` from *server*.

    Parameters
    ----------
    server : str | dict | object
        See the module docstring for accepted forms.
    headers : dict | None
        HTTP headers for HTTP / SSE transports.
    env : dict | None
        Extra environment variables for STDIO transports.
    cwd : str | None
        Working directory for STDIO subprocess.
    transport : str | None
        Force transport: ``"http"``, ``"sse"``, or ``"stdio"``.

    Raises
    ------
    ImportError
        When ``fastmcp`` is not installed.
    ValueError
        When the server spec is unrecognisable.
    """
    try:
        import fastmcp  # noqa: F401
        from fastmcp import Client
        from fastmcp.client.transports import (
            StdioTransport,
            StreamableHttpTransport,
            SSETransport,
        )
    except ImportError:
        raise ImportError(
            "fastmcp is required for MCPTool.  "
            "Install it with:  pip install fastmcp"
        )

    # ── Dict spec ─────────────────────────────────────────────────────────
    if isinstance(server, dict):
        if "url" in server:
            url      = server["url"]
            merged_h = {**(headers or {}), **server.get("headers", {})}
            hint     = transport or server.get("transport")
            if _is_sse(url, hint):
                return Client(SSETransport(url=url, headers=merged_h or None))
            return Client(StreamableHttpTransport(url=url, headers=merged_h or None))

        # stdio dict: {"command": "...", "args": [...], "env": {...}, "cwd": "..."}
        cmd     = server.get("command")
        if not cmd:
            raise ValueError(
                "MCPTool server dict must have either 'url' or 'command'."
            )
        args    = server.get("args", [])
        m_env   = {**(env or {}), **server.get("env", {})}
        m_cwd   = cwd or server.get("cwd")
        return Client(StdioTransport(
            command = cmd,
            args    = args,
            env     = m_env or None,
            cwd     = m_cwd,
        ))

    # ── String: URL or path / command ─────────────────────────────────────
    if isinstance(server, str):
        is_url = server.startswith(("http://", "https://"))
        if is_url or transport in ("http", "sse"):
            if _is_sse(server, transport):
                return Client(SSETransport(url=server, headers=headers))
            return Client(StreamableHttpTransport(url=server, headers=headers))

        # Local path or "python ./server.py --arg" style string → STDIO
        import shlex
        parts = shlex.split(server)
        return Client(StdioTransport(
            command = parts[0],
            args    = parts[1:],
            env     = env or None,
            cwd     = cwd,
        ))

    # ── In-process FastMCP server object ──────────────────────────────────
    return Client(server)


# ---------------------------------------------------------------------------
# Result normalisation
# ---------------------------------------------------------------------------

def _extract_result(result: Any) -> Any:
    """
    Normalise a ``CallToolResult`` to a plain Python value.

    Priority:

    1. ``result.data`` — fully deserialised value (fastmcp ≥ 2.x).
    2. Single text content block → plain ``str``.
    3. Multiple / mixed content blocks → ``list[dict]``.
    4. Empty result → ``None``.
    """
    # Prefer .data (covers primitives, dicts, lists, Pydantic models, etc.)
    data = getattr(result, "data", _MISSING)
    if data is not _MISSING and data is not None:
        return data

    content = getattr(result, "content", None)
    if not content:
        return None

    blocks: list[dict] = []
    for item in content:
        itype = getattr(item, "type", None)
        if itype == "text":
            blocks.append({"type": "text", "text": item.text})
        elif itype == "image":
            blocks.append({
                "type":      "image",
                "data":      item.data,
                "mime_type": getattr(item, "mimeType", "image/png"),
            })
        elif itype == "resource":
            blocks.append({
                "type":     "resource",
                "resource": item.resource,
            })
        else:
            blocks.append({"type": str(itype), "raw": str(item)})

    if len(blocks) == 1 and blocks[0]["type"] == "text":
        return blocks[0]["text"]

    return blocks if blocks else None


_MISSING = object()  # sentinel for "attribute not present"


# ---------------------------------------------------------------------------
# MCPTool
# ---------------------------------------------------------------------------

class MCPTool(Tool):
    """
    A single tool from an MCP server, exposed as a standard aichain Tool.

    ``MCPTool`` opens a connection to the MCP server, calls the specified
    tool, and closes the connection on every ``run()`` call.  The most
    convenient way to create instances is via the :func:`MCPTools`
    factory, which discovers all tools on a server at once.  Build an
    ``MCPTool`` directly when you already know the tool name and want to
    skip the discovery round-trip.

    Parameters
    ----------
    name : str
        Tool name as registered on the MCP server.
    description : str
        Human-readable description (shown to agents choosing tools).
    server : str | dict | object
        MCP server specification.  See :func:`MCPTools` for all accepted
        forms.
    parameters : dict | None, optional
        JSON Schema describing the tool's input arguments.  Used by
        Chains and Agents to match accumulated variables to tool inputs.
        When omitted a permissive schema is used that accepts any keyword
        arguments.
    headers : dict | None, optional
        HTTP headers for HTTP / SSE transports.
    env : dict | None, optional
        Extra environment variables for STDIO subprocess servers.
    cwd : str | None, optional
        Working directory for STDIO subprocess servers.
    transport : str | None, optional
        Force transport: ``"http"``, ``"sse"``, or ``"stdio"``.
        Auto-detected from *server* when omitted.

    Examples
    --------
    ::

        from tools import MCPTool

        # ── HTTP server ────────────────────────────────────────────────
        tool = MCPTool(
            name        = "get_weather",
            description = "Get current weather for a city.",
            server      = "https://weather.example.com/mcp",
        )
        result = tool.run(input={"city": "Tokyo"})

        # ── STDIO subprocess ───────────────────────────────────────────
        tool = MCPTool(
            name        = "list_files",
            description = "List files in a directory.",
            server      = {"command": "python", "args": ["./fs_server.py"]},
        )
        result = tool.run(input={"path": "/tmp"})

        # ── Use in a Chain step ────────────────────────────────────────
        from chain import Chain
        chain = Chain(steps=[
            (tool, "files"),
        ])
        chain.run(variables={"path": "/home"})
    """

    def __init__(
        self,
        name:        str,
        description: str,
        server:      "str | dict | Any",
        *,
        parameters:  "dict | None" = None,
        headers:     "dict | None" = None,
        env:         "dict | None" = None,
        cwd:         "str | None"  = None,
        transport:   "str | None"  = None,
    ) -> None:
        self.name        = name
        self.description = description
        self._server     = server
        self._headers    = headers
        self._env        = env
        self._cwd        = cwd
        self._transport  = transport

        # Exact schema when available (from MCPTools discovery); otherwise
        # a permissive fallback so Chain/Agent can still pass kwargs.
        self.parameters = parameters or {
            "type":                 "object",
            "properties":          {},
            "additionalProperties": True,
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        input:   "dict | str | None" = None,
        options: "dict | None"        = None,
        **kwargs: Any,
    ) -> Any:
        """
        Call the MCP tool and return its result.

        Parameters
        ----------
        input : dict | str | None, optional
            Tool arguments as a dict.  A bare string is treated as a
            single ``"input"`` field (compatibility shim for plain-string
            Chain steps).
        options : dict | None, optional
            Runtime overrides:
              ``timeout`` (float) — per-call timeout in seconds.
        **kwargs
            Individual argument values.  Merged with *input*; kwargs win
            on conflict.  This is the form used by Chains — accumulated
            variables whose names match declared parameter properties are
            passed here automatically.

        Returns
        -------
        str | dict | list | Any
            The tool result.  ``str`` for plain text, ``dict``/``list``
            for structured data, ``None`` for empty responses.

        Raises
        ------
        RuntimeError
            On connection failure or when the server reports a tool error.
        ImportError
            When ``fastmcp`` is not installed.
        """
        arguments: dict = {}
        if isinstance(input, dict):
            arguments.update(input)
        elif isinstance(input, str) and input:
            arguments["input"] = input
        arguments.update(kwargs)

        opts    = options or {}
        timeout = opts.get("timeout")

        return _run_sync(self._acall(arguments, timeout=timeout))

    def __call__(
        self,
        input:   "dict | str | None" = None,
        options: "dict | None"        = None,
        **kwargs: Any,
    ) -> "ToolResult":
        """
        Safe wrapper around :meth:`run` — never raises.

        Keyword arguments are forwarded to :meth:`run` exactly as with
        the direct call form, so both ``tool({"k": v})`` and
        ``tool(k=v)`` work identically.

        Returns
        -------
        ToolResult
            ``success=True`` + raw output on success; ``success=False``
            + error string on any exception.
        """
        from ._base import ToolResult  # local import avoids circular at module level

        try:
            output = self.run(input, options, **kwargs)
            return ToolResult(success=True, output=output)
        except Exception as exc:
            return ToolResult(success=False, output=None, error=str(exc))

    # ------------------------------------------------------------------
    # Async implementation
    # ------------------------------------------------------------------

    async def _acall(self, arguments: dict, *, timeout: "float | None") -> Any:
        client = _build_client(
            self._server,
            headers   = self._headers,
            env       = self._env,
            cwd       = self._cwd,
            transport = self._transport,
        )
        async with client:
            call_kwargs: dict = {"raise_on_error": True}
            if timeout is not None:
                call_kwargs["timeout"] = timeout
            result = await client.call_tool(self.name, arguments, **call_kwargs)
        return _extract_result(result)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        srv = (
            self._server[:47] + "..."
            if isinstance(self._server, str) and len(self._server) > 50
            else str(self._server)
        )
        return f"MCPTool(name={self.name!r}, server={srv!r})"


# ---------------------------------------------------------------------------
# MCPTools factory
# ---------------------------------------------------------------------------

def MCPTools(
    server:    "str | dict | Any",
    *,
    filter:    "list[str] | None" = None,
    headers:   "dict | None"      = None,
    env:       "dict | None"      = None,
    cwd:       "str | None"       = None,
    transport: "str | None"       = None,
) -> "list[MCPTool]":
    """
    Connect to an MCP server, discover all exposed tools, and return one
    :class:`MCPTool` per tool.

    The discovery connection is opened and closed within this call.  Each
    returned ``MCPTool`` opens its own connection independently when
    ``run()`` is called later.

    Parameters
    ----------
    server : str | dict | object
        MCP server specification.

        **String forms (transport auto-detected):**

        * ``"./server.py"`` or ``"python ./server.py --flag"``
          → STDIO subprocess.
        * ``"https://api.example.com/mcp"``
          → Streamable HTTP.
        * ``"https://api.example.com/sse"`` (URL ends with ``/sse``)
          → SSE (legacy).

        **Dict forms (explicit):**

        * ``{"command": "python", "args": ["server.py"]}``
          → STDIO.
        * ``{"command": "npx", "args": ["@scope/server", "/path"], "env": {...}}``
          → STDIO with env vars.
        * ``{"url": "https://api.example.com/mcp"}``
          → HTTP.
        * ``{"url": "https://...", "headers": {"Authorization": "Bearer ..."}}``
          → HTTP with auth.

    filter : list[str] | None, optional
        When provided, only return tools whose names appear in this list.
        All other tools are silently skipped.
    headers : dict | None, optional
        HTTP headers for HTTP / SSE transports.  Merged with any headers
        inside a dict-style *server* spec.
    env : dict | None, optional
        Additional environment variables for STDIO transports.  Merged
        with the current process environment and any ``env`` inside a
        dict-style *server* spec.
    cwd : str | None, optional
        Working directory for STDIO subprocess servers.
    transport : str | None, optional
        Force a specific transport: ``"http"``, ``"sse"``, or ``"stdio"``.
        Auto-detected from *server* when omitted.

    Returns
    -------
    list[MCPTool]
        One :class:`MCPTool` per tool exposed by the server (or filtered
        subset), ordered as returned by the server.

    Raises
    ------
    ImportError
        When ``fastmcp`` is not installed (``pip install fastmcp``).
    RuntimeError
        On connection failure.

    Examples
    --------
    ::

        from tools import MCPTools
        from agent import Agent
        from models import Model

        # ── HTTP server ────────────────────────────────────────────────
        tools = MCPTools("https://api.example.com/mcp")
        for t in tools:
            print(f"{t.name}: {t.description}")

        # ── HTTP with auth ─────────────────────────────────────────────
        tools = MCPTools(
            "https://api.example.com/mcp",
            headers={"Authorization": "Bearer sk-..."},
        )

        # ── STDIO subprocess ───────────────────────────────────────────
        tools = MCPTools({"command": "python", "args": ["./server.py"]})

        # ── npx server ────────────────────────────────────────────────
        tools = MCPTools(
            {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-brave-search"]},
            env={"BRAVE_API_KEY": "..."},
        )

        # ── Only specific tools ────────────────────────────────────────
        tools = MCPTools(
            "https://api.example.com/mcp",
            filter=["search_web", "get_weather"],
        )

        # ── Register in an Agent ───────────────────────────────────────
        agent = Agent(
            model = Model("claude-sonnet-4-6"),
            tools = MCPTools("https://api.example.com/mcp"),
        )
    """
    return _run_sync(
        _discover_tools(
            server    = server,
            filter    = filter,
            headers   = headers,
            env       = env,
            cwd       = cwd,
            transport = transport,
        )
    )


async def _discover_tools(
    server:    "str | dict | Any",
    *,
    filter:    "list[str] | None",
    headers:   "dict | None",
    env:       "dict | None",
    cwd:       "str | None",
    transport: "str | None",
) -> "list[MCPTool]":
    """Async implementation — open connection, fetch tool list, close."""
    client = _build_client(
        server,
        headers   = headers,
        env       = env,
        cwd       = cwd,
        transport = transport,
    )
    async with client:
        mcp_tools = await client.list_tools()

    result: list[MCPTool] = []
    for t in mcp_tools:
        if filter is not None and t.name not in filter:
            continue
        result.append(MCPTool(
            name        = t.name,
            description = getattr(t, "description", "") or "",
            server      = server,
            parameters  = getattr(t, "inputSchema", None),
            headers     = headers,
            env         = env,
            cwd         = cwd,
            transport   = transport,
        ))
    return result
