"""
utilities.mcp._tools — MCPTool and discovery
=============================================

One ``MCPTool`` wraps one operation on an MCP (Model Context Protocol)
server behind the standard ``run(input, options=None)`` interface.
``discover()`` is the internal function :class:`~yait_aichain.utilities.MCP`
uses to list a server's tools; the way in is ``MCP(server)``.

A tool works anywhere a Tool is accepted: Chain steps, Agent tool lists,
or direct ``run()`` calls.  No async boilerplate — the library manages a
background event loop internally.

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

    from yait_aichain.utilities import MCP

    server = MCP("https://api.example.com/mcp",
                 headers={"Authorization": "Bearer sk-..."})
    server.names                                   # ['search', 'summarise', ...]
    server.tool("search").run({"query": "latest AI research"})

    # STDIO subprocess, only two of its tools
    fs = MCP({"command": "npx", "args": ["@modelcontextprotocol/server-filesystem", "/data"]},
             filter=["read_file", "list_directory"])
    Agent(Model("claude-haiku-5-5"), tools=fs.tools)

    # One tool built directly, skipping discovery
    tool = MCPTool(
        name        = "search_web",
        description = "Search the web for recent information.",
        server      = "https://api.example.com/mcp",
    )
    tool.run({"query": "MCP protocol"})

Installation
------------
::

    pip install fastmcp
"""

from __future__ import annotations

from ..._require import require

import asyncio
import os
import shlex
import sys
import threading
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from ...tools._base import Tool, ToolResult


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

    _instance: _AsyncBridge | None = None
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

    def run(self, coro: Any, timeout: float | None = None) -> Any:
        """
        Submit *coro* to the background loop and block until it completes.

        ``timeout`` caps the total wall-clock wait.  Without it, a hung MCP
        server (e.g. a STDIO subprocess that spawned but never answers the
        protocol handshake) would block the calling thread forever — the
        daemon loop thread is not interruptible from the caller's side.
        """
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout)  # re-raises coroutine exceptions
        except FuturesTimeoutError:
            future.cancel()
            raise TimeoutError(
                f"MCP operation did not complete within {timeout:.0f}s "
                "(server hung during connection or call). Pass a larger "
                "call_timeout=… if the operation is legitimately slow."
            ) from None

    @classmethod
    def get(cls) -> _AsyncBridge:
        """Return (or create) the shared singleton bridge."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance


# Wall-clock cap for the connection/handshake phase (and tool discovery).
# Generous — covers slow STDIO subprocess startup — but finite, so a dead
# server fails with TimeoutError instead of hanging the pipeline forever.
DEFAULT_MCP_SETUP_TIMEOUT = 60.0


def _run_sync(coro: Any, timeout: float | None = None) -> Any:
    """Run *coro* synchronously from any context (sync or async)."""
    return _AsyncBridge.get().run(coro, timeout=timeout)


# ---------------------------------------------------------------------------
# The server: where it is and how to reach it, settled once
# ---------------------------------------------------------------------------

_TRANSPORTS = ("http", "streamable", "sse", "stdio")


def _is_sse(url: str, transport_hint: str | None) -> bool:
    """Return True when the SSE transport should be preferred."""
    if transport_hint == "sse":
        return True
    if transport_hint in ("http", "streamable"):
        return False
    return urlsplit(url).path.rstrip("/").endswith("/sse")


def _redacted(spec: Any) -> str:
    """The server for a repr or an error: a URL without its query, a dict
    without the values of its headers and env, which is where keys live."""
    if isinstance(spec, str):
        return urlsplit(spec)._replace(query="", fragment="").geturl() if "://" in spec else spec
    if isinstance(spec, dict):
        return repr({k: (sorted(v) if k in ("headers", "env") and isinstance(v, dict) else v)
                     for k, v in spec.items()})
    return type(spec).__name__


@dataclass(frozen=True)
class _Server:
    """An MCP server and the settings for reaching it — one value, checked
    here once, instead of five keywords threaded through every signature."""

    spec:         Any
    headers:      dict | None  = None
    env:          dict | None  = None
    cwd:          str | None   = None
    transport:    str | None   = None
    call_timeout: float | None = None

    def __post_init__(self) -> None:
        if self.spec is None or (isinstance(self.spec, str) and not self.spec.strip()):
            raise ValueError("MCP needs a server: a URL, a command such as "
                             "'python server.py', or a dict with 'url' or 'command'")
        if self.transport is not None and self.transport not in _TRANSPORTS:
            raise ValueError(f"MCP transport {self.transport!r} is not one of {list(_TRANSPORTS)}")
        if isinstance(self.spec, dict):
            if not (self.spec.get("url") or self.spec.get("command")):
                raise ValueError("an MCP server dict must have either 'url' or 'command'")
            # One source for each: given both as an argument and in the server
            # dict, it is an error rather than a merge where one side wins.
            for key in ("headers", "env", "cwd", "transport"):
                if getattr(self, key) and self.spec.get(key):
                    raise ValueError(f"MCP {key} given twice — as {key}= and in the server dict; "
                                     "give it in one place")

    def __repr__(self) -> str:
        return f"_Server({_redacted(self.spec)})"


def _build_client(server: _Server) -> Any:
    """A ``fastmcp.Client`` for *server*; see the module docstring for the forms."""
    Client = require("fastmcp", package="fastmcp", who="MCP", extra="mcp").Client
    from fastmcp.client.transports import SSETransport, StdioTransport, StreamableHttpTransport

    spec = server.spec
    if isinstance(spec, dict):
        headers   = server.headers or spec.get("headers") or None
        env       = server.env or spec.get("env") or None
        cwd       = server.cwd or spec.get("cwd")
        transport = server.transport or spec.get("transport")
        url       = spec.get("url")
        command   = [spec.get("command"), *(spec.get("args") or [])]
    elif isinstance(spec, str):
        headers, env, cwd, transport = server.headers, server.env, server.cwd, server.transport
        is_url  = spec.startswith(("http://", "https://"))
        url     = spec if is_url or transport in ("http", "streamable", "sse") else None
        command = shlex.split(spec)
    else:
        return Client(spec)                       # an in-process FastMCP server

    if url:
        kind = SSETransport if _is_sse(url, transport) else StreamableHttpTransport
        return Client(kind(url=url, headers=headers))
    # A script is run by its interpreter, not as a binary: `./server.py` has
    # no business needing an executable bit and a shebang.
    runner = {".py": sys.executable, ".js": "node"}.get(os.path.splitext(command[0])[1])
    if runner:
        command = [runner, *command]
    return Client(StdioTransport(command=command[0], args=command[1:], env=env, cwd=cwd))


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
    tool, and closes the connection on every ``run()`` call.  The usual
    way to get one is ``MCP(server).tool(name)``, which discovers the
    server's tools once.  Build an ``MCPTool`` directly when you already
    know the tool name and want to skip the discovery round-trip.

    Parameters
    ----------
    name : str
        Tool name as registered on the MCP server.
    description : str
        Human-readable description (shown to agents choosing tools).
    server : str | dict | object
        MCP server specification, in any form :class:`MCP` accepts
        (a path or command, a URL, or a dict).
    parameters : dict | None, optional
        JSON Schema describing the tool's input arguments.  Used by
        Chains and Agents to match accumulated variables to tool inputs.
        When omitted a permissive schema is used that accepts any keyword
        arguments.
    headers : dict | None, optional
        HTTP headers for HTTP / SSE transports.
    env : dict | None, optional
        Environment variables for a STDIO server, added to a minimal
        inherited set (``PATH``, ``HOME``, ``USER`` and the like) — not to
        the whole process environment, so a key in your shell reaches the
        server only when it is passed here.
    cwd : str | None, optional
        Working directory for STDIO subprocess servers.
    transport : str | None, optional
        Force transport: ``"http"``, ``"streamable"``, ``"sse"`` or
        ``"stdio"``. Auto-detected from *server* when omitted.
    risk : str, optional
        The risk class a permission policy reads. ``"privileged"`` when
        omitted: what another server's tool does is unknown here, and the
        unknown goes to a person, not through. Discovery sets it from the
        server's annotations — see :func:`risk_from_annotations`.

    Examples
    --------
    ::

        from yait_aichain.utilities.mcp import MCPTool

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
        from yait_aichain.chain import Chain
        chain = Chain(steps=[
            (tool, "files"),
        ])
        chain.run(variables={"path": "/home"})
    """

    def __init__(
        self,
        name:        str,
        description: str,
        server:      str | dict | Any,
        *,
        parameters:  dict | None = None,
        headers:     dict | None = None,
        env:         dict | None = None,
        cwd:         str | None  = None,
        transport:   str | None  = None,
        call_timeout: float | None = None,
        risk:        str           = "privileged",
    ) -> None:
        from ...tools._permissions import RISK_CLASSES
        if risk not in RISK_CLASSES:
            raise ValueError(f"MCPTool {name!r}: risk {risk!r} is not one of {sorted(RISK_CLASSES)}")
        self.risk        = risk
        self.name        = name
        self.description = description
        self._conn = server if isinstance(server, _Server) else _Server(
            server, headers=headers, env=env, cwd=cwd, transport=transport,
            call_timeout=call_timeout)

        # Exact schema when available (from discovery); otherwise
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
        input:   dict | str | None = None,
        options: dict | None        = None,
        **kwargs: Any,
    ) -> Any:
        """
        Call the MCP tool and return its result.

        Parameters
        ----------
        input : dict | str | None, optional
            Tool arguments as a dict.  A bare string is treated as a
            single ``"input"`` field, for Chain steps that pass a plain
            string.
        options : dict | None, optional
            None. A call's timeout is the tool's ``call_timeout`` setting;
            options given here are an error that says so.
        **kwargs
            Individual argument values, merged with *input*; an argument
            given in both is an error.  This is the form used by Chains — accumulated
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
        if clash := sorted(set(arguments) & set(kwargs)):
            raise ValueError(f"MCP tool {self.name!r}: {clash} given both in the input dict and as "
                             "keyword arguments — pass each argument once")
        arguments.update(kwargs)

        # A call's timeout is the tool's setting, not a per-call option.
        if options:
            raise ValueError(f"MCP tool {self.name!r}: "
                             + ("timeout is a setting — MCP(…, call_timeout=…)"
                                if "timeout" in options else
                                f"a call takes the server's arguments, not options {sorted(options)}"))
        return _run_sync(self._acall(arguments))

    def __call__(
        self,
        input:   dict | str | None = None,
        options: dict | None        = None,
        **kwargs: Any,
    ) -> ToolResult:
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
        try:
            output = self.run(input, options, **kwargs)
            return ToolResult(success=True, output=output)
        except Exception as exc:
            return ToolResult(success=False, output=None, error=str(exc))

    # ------------------------------------------------------------------
    # Async implementation
    # ------------------------------------------------------------------

    async def _acall(self, arguments: dict) -> Any:
        client = _build_client(self._conn)
        # The connection/handshake phase gets its own finite cap: the
        # per-call ``timeout`` option only covers ``call_tool``, so a STDIO
        # subprocess that spawns but never speaks the protocol would
        # otherwise hang here forever.
        try:
            await asyncio.wait_for(
                client.__aenter__(), DEFAULT_MCP_SETUP_TIMEOUT
            )
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"MCP server did not complete the connection handshake "
                f"within {DEFAULT_MCP_SETUP_TIMEOUT:.0f}s: {_redacted(self._conn.spec)}"
            ) from None
        try:
            call_kwargs: dict = {"raise_on_error": True}
            if self._conn.call_timeout is not None:
                call_kwargs["timeout"] = self._conn.call_timeout
            result = await client.call_tool(self.name, arguments, **call_kwargs)
        finally:
            await client.__aexit__(None, None, None)
        return _extract_result(result)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        srv = _redacted(self._conn.spec)
        return f"MCPTool(name={self.name!r}, server={srv[:47] + '...' if len(srv) > 50 else srv!r})"


# ---------------------------------------------------------------------------
# Discovery (internal; MCP calls it)
# ---------------------------------------------------------------------------

def discover(server: _Server, *, filter: list[str] | None = None,
             timeout: float | None = DEFAULT_MCP_SETUP_TIMEOUT) -> list[MCPTool]:
    """Ask *server* for its tools, one :class:`MCPTool` each, in its order.

    *filter* keeps the tools it names, and a name the server does not have is
    a ``KeyError`` listing the ones it has. *timeout* caps the whole discovery
    — connect, handshake, listing — and ``None`` waits indefinitely. The
    connection closes here; each tool opens its own when it runs.
    """
    return _run_sync(_discover(server, filter), timeout=timeout)


async def _discover(server: _Server, filter: list[str] | None) -> list[MCPTool]:
    async with _build_client(server) as client:
        listed = await client.list_tools()
    if filter is not None:
        missing = sorted(set(filter) - {t.name for t in listed})
        if missing:
            raise KeyError(f"filter= names {missing}, which the server does not have; "
                           f"it has {[t.name for t in listed]}")
    return [MCPTool(
        name        = t.name,
        description = getattr(t, "description", "") or "",
        server      = server,
        # `input_schema` since MCP SDK v2; `inputSchema` before it, and
        # deprecated now — read only when the new field is absent.
        parameters  = getattr(t, "input_schema", None) or getattr(t, "inputSchema", None),
        risk        = risk_from_annotations(getattr(t, "annotations", None)),
    ) for t in listed if filter is None or t.name in filter]


def _hint(annotations: Any, snake: str, camel: str) -> bool | None:
    """One hint, under its MCP SDK v2 name or the camelCase one before it."""
    value = getattr(annotations, snake, None)
    if value is None:
        value = getattr(annotations, camel, None)
    if value is None and isinstance(annotations, dict):
        value = annotations.get(camel, annotations.get(snake))
    return value if isinstance(value, bool) else None


def risk_from_annotations(annotations: Any) -> str:
    """The risk class an MCP tool's annotations claim.

    MCP lets a server describe each tool: ``readOnlyHint``,
    ``destructiveHint``, ``openWorldHint``. They are the server's word, which
    the specification itself calls untrusted — so only an explicit hint lowers
    the class, and a tool the server says nothing about is ``privileged``:
    a permission policy sends it to a person. ``MCP(server, risk={...})`` puts
    the caller's word over the server's.

    ============================================  ===============
    annotations                                    risk
    ============================================  ===============
    ``readOnlyHint: true``                         ``read``
    ``destructiveHint: true``                      ``destructive``
    ``destructiveHint: false, openWorldHint: false``  ``write``
    ``destructiveHint: false``, otherwise          ``external``
    none, or nothing that settles it               ``privileged``
    ============================================  ===============
    """
    if annotations is None:
        return "privileged"
    if _hint(annotations, "read_only_hint", "readOnlyHint") is True:
        return "read"
    destructive = _hint(annotations, "destructive_hint", "destructiveHint")
    if destructive is True:
        return "destructive"
    if destructive is False:
        open_world = _hint(annotations, "open_world_hint", "openWorldHint")
        return "write" if open_world is False else "external"
    return "privileged"
