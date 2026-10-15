"""``MCPTool`` — one operation on an MCP server behind ``run(input, options)`` —
and ``discover()``, which ``MCP`` uses to list a server's tools. Calls are
synchronous over one shared background event loop; requires ``fastmcp``.
See docs/utilities/index.md.
"""

from __future__ import annotations

from ..._require import require

import asyncio
import os
import shlex
import sys
import threading
import concurrent.futures
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from ...tools._base import Tool


# ---------------------------------------------------------------------------
# Background event loop — shared singleton
# ---------------------------------------------------------------------------

class _AsyncBridge:
    """One daemon thread running an event loop that every MCP call blocks on:
    ``asyncio.run()`` fails inside a running loop (Jupyter, FastAPI), and a loop
    per call would rebuild STDIO connections each time."""

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

    def run(self, coro: Any, timeout: float | None = None, setting: str = "call_timeout") -> Any:
        """
        Submit *coro* to the background loop and block until it completes.

        ``timeout`` caps the total wall-clock wait.  Without it, a hung MCP
        server (e.g. a STDIO subprocess that spawned but never answers the
        protocol handshake) would block the calling thread forever — the
        daemon loop thread is not interruptible from the caller's side.
        ``setting`` is the name the timeout error tells the caller to raise.
        """
        if threading.current_thread() is self._thread:
            coro.close()
            raise RuntimeError("an MCP tool was called synchronously from the MCP loop's own "
                               "thread, which would wait on itself forever; await it instead")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        # wait(), not result(timeout): from 3.11 the futures TimeoutError is
        # the builtin one, so a TimeoutError the coroutine raised would be
        # taken for the wait's own.
        done, _ = concurrent.futures.wait([future], timeout)
        if not done:
            future.cancel()
            raise TimeoutError(
                f"MCP operation did not complete within {timeout:.0f}s "
                f"(server hung during connection or call). Pass a larger "
                f"{setting}=… if the operation is legitimately slow."
            )
        return future.result()                   # re-raises coroutine exceptions

    @classmethod
    def get(cls) -> _AsyncBridge:
        """Return (or create) the shared singleton bridge."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def _after_fork(cls) -> None:
        # A forked child inherits the singleton but not its thread: every
        # call would wait on a loop nobody runs. The child starts its own.
        cls._instance = None
        cls._lock = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_AsyncBridge._after_fork)


# Wall-clock cap for the connection/handshake phase (and tool discovery).
# Generous — covers slow STDIO subprocess startup — but finite, so a dead
# server fails with TimeoutError instead of hanging the pipeline forever.
DEFAULT_MCP_SETUP_TIMEOUT = 60.0

# Wall-clock cap for one call of a tool. Finite, so a server that accepts a
# call and never answers fails the call instead of holding the agent forever;
# ``call_timeout=None`` asks for that wait explicitly.
DEFAULT_MCP_CALL_TIMEOUT = 300.0


def _run_sync(coro: Any, timeout: float | None = None, setting: str = "call_timeout") -> Any:
    """Run *coro* synchronously from any context (sync or async)."""
    return _AsyncBridge.get().run(coro, timeout=timeout, setting=setting)


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
    call_timeout: float | None = DEFAULT_MCP_CALL_TIMEOUT

    def __post_init__(self) -> None:
        if self.spec is None or (isinstance(self.spec, str) and not self.spec.strip()):
            raise ValueError("MCP needs a server: a URL, a command such as "
                             "'python server.py', or a dict with 'url' or 'command'")
        if self.call_timeout is not None and not (
                isinstance(self.call_timeout, (int, float)) and not isinstance(self.call_timeout, bool)
                and self.call_timeout > 0):
            raise ValueError(f"MCP call_timeout is a number of seconds above 0, or None to wait "
                             f"indefinitely — not {self.call_timeout!r}")
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
    """A ``CallToolResult`` as a plain value: media as parts, else the server's
    ``structured_content`` (not ``.data``, a generated dataclass that reads as
    ``Root(code='LH1')``), else ``.data``, else the text; ``None`` when empty."""
    content = list(getattr(result, "content", None) or [])
    parts = [_part(item) for item in content]
    if any(isinstance(p, dict) for p in parts):
        return parts

    structured = getattr(result, "structured_content", None)
    if isinstance(structured, dict):
        meta = getattr(result, "meta", None)
        wrapped = isinstance(meta, dict) and (meta.get("fastmcp") or {}).get("wrap_result")
        return structured.get("result") if wrapped else structured

    data = getattr(result, "data", _MISSING)
    if data is not _MISSING and data is not None:
        import dataclasses
        if dataclasses.is_dataclass(data) and not isinstance(data, type):
            return dataclasses.asdict(data)
        return data

    if not parts:
        return None
    return parts[0] if len(parts) == 1 else parts


def _media(kind: str, mime: str, data: Any) -> dict:
    return {"type": kind, "source": {"kind": "base64", "mime": mime, "data": data}}


def _mime(item: Any, default: str) -> str:
    # `mime_type` since MCP SDK v2; `mimeType` before it.
    mime = getattr(item, "mime_type", None)
    if not isinstance(mime, str):
        mime = getattr(item, "mimeType", None)
    return mime if isinstance(mime, str) and mime else default


def _part(item: Any) -> Any:
    """One content block as a string of text or a media part."""
    itype = getattr(item, "type", None)
    if itype == "text":
        return item.text
    if itype in ("image", "audio"):
        return _media(itype, _mime(item, "image/png" if itype == "image" else "audio/wav"), item.data)
    if itype == "resource":
        res = item.resource
        text = getattr(res, "text", None)
        if isinstance(text, str):
            return text
        blob = getattr(res, "blob", None)
        if isinstance(blob, str):
            mime = _mime(res, "application/octet-stream")
            kind = mime.split("/", 1)[0]
            return _media(kind if kind in ("image", "audio", "video") else "document", mime, blob)
        return f"[resource {getattr(res, 'uri', '')}]"
    if itype == "resource_link":
        return f"[resource {getattr(item, 'uri', '')}]"
    return str(item)


_MISSING = object()  # sentinel for "attribute not present"


# ---------------------------------------------------------------------------
# MCPTool
# ---------------------------------------------------------------------------

class MCPTool(Tool):
    """
    A single tool from an MCP server, exposed as a standard aichain Tool.

    Every ``run()`` opens a connection, calls the tool and closes it. The
    usual way to get one is ``MCP(server).tool(name)``; build it directly to
    skip discovery when the name is known.

    Parameters
    ----------
    name : str
        Tool name as registered on the MCP server.
    description : str
        Human-readable description (shown to agents choosing tools).
    server : str | dict | object
        MCP server, in any form :class:`MCP` accepts (path, command, URL, dict).
    parameters : dict | None, optional
        JSON Schema of the arguments; omitted, any keyword is accepted.
    headers : dict | None, optional
        HTTP headers for HTTP / SSE transports.
    env : dict | None, optional
        Variables for a STDIO server, added to a minimal set (``PATH``,
        ``HOME``…) — not the whole process environment.
    cwd : str | None, optional
        Working directory for STDIO subprocess servers.
    transport : str | None, optional
        ``"http"``, ``"streamable"``, ``"sse"`` or ``"stdio"``; auto-detected.
    call_timeout : float | None, optional
        Seconds one call may take, 300 by default; ``None`` waits indefinitely.
    risk : str, optional
        The risk class a permission policy reads; ``"privileged"`` when
        omitted — see :func:`risk_from_annotations`.

    Examples
    --------
    ::

        tool = MCPTool("get_weather", "Get current weather for a city.",
                       server="https://weather.example.com/mcp")
        result = tool.run({"city": "Tokyo"})
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
        call_timeout: float | None = DEFAULT_MCP_CALL_TIMEOUT,
        risk:        str           = "privileged",
    ) -> None:
        from ...tools._permissions import RISK_CLASSES
        if risk not in RISK_CLASSES:
            raise ValueError(f"MCPTool {name!r}: risk {risk!r} is not one of {sorted(RISK_CLASSES)}")
        self.risk        = risk
        # The server's name for the call; `name` is what the model sees, which
        # a rename or a prefix changes — and a renamed tool sent its new name
        # to a server that had never heard it.
        self._remote     = name
        self.name        = _model_name(name)
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
        /,
        **kwargs: Any,
    ) -> Any:
        """Call the MCP tool and return its result.

        Parameters
        ----------
        input : dict | str | None, optional
            Tool arguments as a dict; a bare string is a single ``"input"`` field.
        options : dict | None, optional
            None; the timeout is the tool's ``call_timeout`` setting.
        **kwargs
            Arguments by name, merged with *input*. *input* and *options* are
            positional only, so ``tool.run(input={"a": 1})`` sends ``input``.

        Returns
        -------
        str | dict | list | Any
            Text, structured data, or ``None`` for an empty response.

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
        # The bridge's own cap sits above the call's and the handshake's, so
        # the server's timeout fires first and names itself; this one catches
        # a client that ignored both.
        limit = self._conn.call_timeout
        return _run_sync(self._acall(arguments),
                         timeout=None if limit is None else limit + DEFAULT_MCP_SETUP_TIMEOUT + 5)

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
            result = await client.call_tool(self._remote, arguments, **call_kwargs)
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

def _model_name(name: str) -> str:
    """A server's tool name as providers take one: letters, digits, ``_`` and
    ``-``, at most 64. ``files.read`` and ``fs/list`` were a 400 from OpenAI."""
    import re
    return re.sub(r"[^A-Za-z0-9_-]", "_", name)[:64] or "tool"


def discover(server: _Server, *, filter: list[str] | None = None,
             timeout: float | None = DEFAULT_MCP_SETUP_TIMEOUT) -> list[MCPTool]:
    """Ask *server* for its tools, one :class:`MCPTool` each, in its order.

    *filter* keeps the tools it names, and a name the server does not have is
    a ``KeyError`` listing the ones it has. *timeout* caps the whole discovery
    — connect, handshake, listing — and ``None`` waits indefinitely. The
    connection closes here; each tool opens its own when it runs.
    """
    return _run_sync(_discover(server, filter), timeout=timeout, setting="connect_timeout")


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
        # Bounded: a server's description goes into every request's tool
        # list, and an unbounded one is where a hostile server hides its
        # instructions.
        description = _bounded(getattr(t, "description", "") or ""),
        server      = server,
        # `input_schema` since MCP SDK v2; `inputSchema` before it, and
        # deprecated now — read only when the new field is absent.
        parameters  = getattr(t, "input_schema", None) or getattr(t, "inputSchema", None),
        risk        = risk_from_annotations(getattr(t, "annotations", None)),
    ) for t in listed if filter is None or t.name in filter]


#: The most of a server's tool description passed on to a model.
MAX_DESCRIPTION_CHARS = 2000


def _bounded(text: str) -> str:
    return text if len(text) <= MAX_DESCRIPTION_CHARS else text[:MAX_DESCRIPTION_CHARS] + "…"


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

    Only an explicit hint lowers the class; ``MCP(server, risk={...})`` puts
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
