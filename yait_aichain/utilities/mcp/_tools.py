"""
utilities.mcp._tools — MCPTool, MCPTools
=========================================

One ``MCPTool`` wraps one operation on an MCP (Model Context Protocol)
server behind the standard ``run(input, options=None)`` interface.
``MCPTools()`` is the internal discovery function :class:`~yait_aichain.utilities.MCP`
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
import threading
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any

from ...tools._base import Tool


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

    def run(self, coro: Any, timeout: "float | None" = None) -> Any:
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
    def get(cls) -> "_AsyncBridge":
        """Return (or create) the shared singleton bridge."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance


# Wall-clock cap for the connection/handshake phase (and tool discovery).
# Generous — covers slow STDIO subprocess startup — but finite, so a dead
# server fails with TimeoutError instead of hanging the pipeline forever.
DEFAULT_MCP_SETUP_TIMEOUT = 60.0


def _run_sync(coro: Any, timeout: "float | None" = None) -> Any:
    """Run *coro* synchronously from any context (sync or async)."""
    return _AsyncBridge.get().run(coro, timeout=timeout)


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
    Client = require("fastmcp", package="fastmcp", who="MCP", extra="mcp").Client
    from fastmcp.client.transports import SSETransport, StdioTransport, StreamableHttpTransport

    # ── Dict spec ─────────────────────────────────────────────────────────
    if isinstance(server, dict):
        # One source for each: given both as an argument and in the server
        # dict, it is an error rather than a merge where one side wins.
        for key, given in (("headers", headers), ("env", env), ("cwd", cwd), ("transport", transport)):
            if given and server.get(key):
                raise ValueError(f"MCP {key} given twice — as {key}= and in the server dict; "
                                 "give it in one place")
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
        Extra environment variables for STDIO subprocess servers.
    cwd : str | None, optional
        Working directory for STDIO subprocess servers.
    transport : str | None, optional
        Force transport: ``"http"``, ``"sse"``, or ``"stdio"``.
        Auto-detected from *server* when omitted.
    risk : str, optional
        The risk class a permission policy reads. ``"privileged"`` when
        omitted: what another server's tool does is unknown here, and the
        unknown goes to a person, not through. Discovery sets it from the
        server's annotations — see :func:`risk_from_annotations`.

    Examples
    --------
    ::

        from yait_aichain.tools import MCPTool

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
        server:      "str | dict | Any",
        *,
        parameters:  "dict | None" = None,
        headers:     "dict | None" = None,
        env:         "dict | None" = None,
        cwd:         "str | None"  = None,
        transport:   "str | None"  = None,
        call_timeout: "float | None" = None,
        risk:        str           = "privileged",
    ) -> None:
        from ...tools._permissions import RISK_CLASSES
        if risk not in RISK_CLASSES:
            raise ValueError(f"MCPTool {name!r}: risk {risk!r} is not one of {sorted(RISK_CLASSES)}")
        self.risk        = risk
        self.name        = name
        self._call_timeout = call_timeout
        self.description = description
        self._server     = server
        self._headers    = headers
        self._env        = env
        self._cwd        = cwd
        self._transport  = transport

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
        return _run_sync(self._acall(arguments, timeout=self._call_timeout))

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
        from ...tools._base import ToolResult  # local import avoids circular at module level

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
                f"within {DEFAULT_MCP_SETUP_TIMEOUT:.0f}s: {self._server!r}"
            ) from None
        try:
            call_kwargs: dict = {"raise_on_error": True}
            if timeout is not None:
                call_kwargs["timeout"] = timeout
            result = await client.call_tool(self.name, arguments, **call_kwargs)
        finally:
            await client.__aexit__(None, None, None)
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
# Discovery (internal; MCP calls it)
# ---------------------------------------------------------------------------

def MCPTools(
    server:    "str | dict | Any",
    *,
    filter:    "list[str] | None" = None,
    headers:   "dict | None"      = None,
    env:       "dict | None"      = None,
    cwd:       "str | None"       = None,
    transport: "str | None"       = None,
    timeout:   "float | None"     = DEFAULT_MCP_SETUP_TIMEOUT,
    call_timeout: "float | None"  = None,
) -> "list[MCPTool]":
    """
    Connect to an MCP server, discover all exposed tools, and return one
    :class:`MCPTool` per tool.

    Internal: this is the discovery function :class:`MCP` calls on first
    use of ``tools``. Callers use ``MCP(server)``, whose ``connect_timeout``
    becomes ``timeout`` here.

    The discovery connection is opened and closed within this call.  Each
    returned ``MCPTool`` opens its own connection independently when
    ``run()`` is called later.

    ``timeout`` caps the whole discovery (connect + handshake + list_tools);
    a hung server raises ``TimeoutError`` instead of blocking forever.
    Pass ``timeout=None`` to wait indefinitely.

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

        tools = MCPTools("https://api.example.com/mcp", filter=["search_web"])
        # what MCP("https://api.example.com/mcp", filter=["search_web"]).tools returns
    """

    return _run_sync(
        _discover_tools(
            server    = server,
            filter    = filter,
            headers   = headers,
            env       = env,
            cwd       = cwd,
            transport = transport,
            call_timeout = call_timeout,
        ),
        timeout = timeout,
    )


async def _discover_tools(
    server:    "str | dict | Any",
    *,
    filter:    "list[str] | None",
    headers:   "dict | None",
    env:       "dict | None",
    cwd:       "str | None",
    transport: "str | None",
    call_timeout: "float | None" = None,
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
            # `input_schema` since MCP SDK v2; `inputSchema` before it, and
            # deprecated now — read only when the new field is absent.
            parameters  = getattr(t, "input_schema", None) or getattr(t, "inputSchema", None),
            headers     = headers,
            env         = env,
            cwd         = cwd,
            transport   = transport,
            call_timeout = call_timeout,
            risk        = risk_from_annotations(getattr(t, "annotations", None)),
        ))
    return result


def _hint(annotations: Any, snake: str, camel: str) -> "bool | None":
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
