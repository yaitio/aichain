"""``MCPTool`` — one operation on an MCP server behind ``run(input, options)`` —
and ``discover()``, which ``MCP`` uses to list a server's tools. Calls are
synchronous over one shared background event loop; requires ``fastmcp``.
See docs/utilities/index.md.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import dataclasses
import os
import re
import shlex
import sys
import threading
from typing import Any
from urllib.parse import urlsplit

from ..._require import require
from ...tools._base import Tool

# Reaching the server and listing its tools; finite, so a dead server fails.
DEFAULT_MCP_SETUP_TIMEOUT = 60.0
# One call of a tool; ``call_timeout=None`` asks for an unbounded wait.
DEFAULT_MCP_CALL_TIMEOUT = 300.0
#: The most of a server's tool description passed on to a model.
MAX_DESCRIPTION_CHARS = 2000
_TRANSPORTS = ("http", "streamable", "sse", "stdio")
_MISSING = object()


class _AsyncBridge:
    """One daemon thread running the loop every MCP call blocks on: ``asyncio.run()``
    fails inside a running loop (Jupyter, FastAPI)."""

    _instance: _AsyncBridge | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True,
                                        name="aichain-mcp-loop")
        self._thread.start()

    def run(self, coro: Any, timeout: float | None = None, setting: str = "call_timeout") -> Any:
        """Run *coro* on the loop and wait at most *timeout*; *setting* is the name
        the timeout error tells the caller to raise."""
        if threading.current_thread() is self._thread:
            coro.close()
            raise RuntimeError("an MCP tool was called synchronously from the MCP loop's own "
                               "thread, which would wait on itself forever; await it instead")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        # wait(), not result(timeout): from 3.11 a TimeoutError the coroutine
        # raised would be taken for the wait's own.
        if not concurrent.futures.wait([future], timeout)[0]:
            future.cancel()
            raise TimeoutError(f"MCP operation did not complete within {timeout:.0f}s (server hung "
                               f"during connection or call). Pass a larger {setting}=… if the "
                               "operation is legitimately slow.")
        return future.result()

    @classmethod
    def get(cls) -> _AsyncBridge:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def _after_fork(cls) -> None:
        # A forked child inherits the singleton but not its thread.
        cls._instance, cls._lock = None, threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_AsyncBridge._after_fork)


def _run_sync(coro: Any, timeout: float | None = None, setting: str = "call_timeout") -> Any:
    return _AsyncBridge.get().run(coro, timeout=timeout, setting=setting)


def _is_sse(url: str, transport: str | None) -> bool:
    if transport in ("sse", "http", "streamable"):
        return transport == "sse"
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


@dataclasses.dataclass(frozen=True)
class _Server:
    """An MCP server and how to reach it, checked once; a dict server's own
    headers, env, cwd and transport are folded in. Its repr shows no value of them."""

    spec:         Any
    headers:      dict | None  = None
    env:          dict | None  = None
    cwd:          str | None   = None
    transport:    str | None   = None
    call_timeout: float | None = DEFAULT_MCP_CALL_TIMEOUT

    def __post_init__(self) -> None:
        spec, limit = self.spec, self.call_timeout
        if spec is None or (isinstance(spec, str) and not spec.strip()):
            raise ValueError("MCP needs a server: a URL, a command such as "
                             "'python server.py', or a dict with 'url' or 'command'")
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, (int, float))
                                  or limit <= 0):
            raise ValueError(f"MCP call_timeout is a number of seconds above 0, or None to wait "
                             f"indefinitely — not {limit!r}")
        if self.transport is not None and self.transport not in _TRANSPORTS:
            raise ValueError(f"MCP transport {self.transport!r} is not one of {list(_TRANSPORTS)}")
        if not isinstance(spec, dict):
            return
        if not (spec.get("url") or spec.get("command")):
            raise ValueError("an MCP server dict must have either 'url' or 'command'")
        for key in ("headers", "env", "cwd", "transport"):
            # One source for each, rather than a merge where one side wins.
            if getattr(self, key) and spec.get(key):
                raise ValueError(f"MCP {key} given twice — as {key}= and in the server dict; "
                                 "give it in one place")
            object.__setattr__(self, key, getattr(self, key) or spec.get(key) or None)

    def __repr__(self) -> str:
        return f"_Server({_redacted(self.spec)})"


def _build_client(server: _Server) -> Any:
    """A ``fastmcp.Client`` for *server*: a URL over HTTP or SSE, a command over STDIO,
    anything else handed to the client as an in-process server."""
    Client = require("fastmcp", package="fastmcp", who="MCP", extra="mcp").Client
    from fastmcp.client.transports import SSETransport, StdioTransport, StreamableHttpTransport
    spec, transport = server.spec, server.transport
    if isinstance(spec, dict):
        url = spec.get("url")
    elif isinstance(spec, str):
        named = transport in _TRANSPORTS[:3]
        url = spec if named or spec.startswith(("http://", "https://")) else None
    else:
        return Client(spec)
    if url:
        kind = SSETransport if _is_sse(url, transport) else StreamableHttpTransport
        return Client(kind(url=url, headers=server.headers))
    command = (shlex.split(spec) if isinstance(spec, str)
               else [spec["command"], *(spec.get("args") or [])])
    # A script is run by its interpreter, not as a binary needing an executable bit.
    runner = {".py": sys.executable, ".js": "node"}.get(os.path.splitext(command[0])[1])
    command = [runner, *command] if runner else command
    return Client(StdioTransport(command=command[0], args=command[1:], env=server.env,
                                 cwd=server.cwd))


def _extract_result(result: Any) -> Any:
    """A ``CallToolResult`` as a plain value: media as parts, else ``structured_content``
    (``.data`` is a generated dataclass), else ``.data``, else the text, else ``None``."""
    parts = [_part(item) for item in getattr(result, "content", None) or ()]
    if any(isinstance(p, dict) for p in parts):
        return parts
    structured = getattr(result, "structured_content", None)
    if isinstance(structured, dict):
        meta = getattr(result, "meta", None)
        wrapped = isinstance(meta, dict) and (meta.get("fastmcp") or {}).get("wrap_result")
        return structured.get("result") if wrapped else structured
    data = getattr(result, "data", None)
    if data is not None:
        is_dataclass = dataclasses.is_dataclass(data) and not isinstance(data, type)
        return dataclasses.asdict(data) if is_dataclass else data
    return (parts[0] if len(parts) == 1 else parts) if parts else None


def _media(kind: str, mime: str, data: Any) -> dict:
    return {"type": kind, "source": {"kind": "base64", "mime": mime, "data": data}}


def _mime(item: Any, default: str) -> str:
    # `mime_type` since MCP SDK v2; `mimeType` before it.
    for field in ("mime_type", "mimeType"):
        if isinstance(mime := getattr(item, field, None), str):
            return mime or default
    return default


def _part(item: Any) -> Any:
    itype = getattr(item, "type", None)
    if itype == "text":
        return item.text
    if itype in ("image", "audio"):
        return _media(itype, _mime(item, "image/png" if itype == "image" else "audio/wav"), item.data)
    if itype == "resource":
        res = item.resource
        if isinstance(text := getattr(res, "text", None), str):
            return text
        if isinstance(blob := getattr(res, "blob", None), str):
            mime = _mime(res, "application/octet-stream")
            kind = mime.split("/", 1)[0]
            return _media(kind if kind in ("image", "audio", "video") else "document", mime, blob)
        return f"[resource {getattr(res, 'uri', '')}]"
    if itype == "resource_link":
        return f"[resource {getattr(item, 'uri', '')}]"
    return str(item)


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

    def __init__(self, name: str, description: str, server: str | dict | Any, *,
                 parameters: dict | None = None, headers: dict | None = None,
                 env: dict | None = None, cwd: str | None = None, transport: str | None = None,
                 call_timeout: float | None = DEFAULT_MCP_CALL_TIMEOUT,
                 risk: str = "privileged") -> None:
        from ...tools._permissions import RISK_CLASSES
        if risk not in RISK_CLASSES:
            raise ValueError(f"MCPTool {name!r}: risk {risk!r} is not one of {sorted(RISK_CLASSES)}")
        # The server's name for the call; `name` is what the model sees, which
        # a rename or a prefix changes.
        self.risk, self._remote, self.name = risk, name, _model_name(name)
        self.description = description
        self._conn = server if isinstance(server, _Server) else _Server(
            server, headers=headers, env=env, cwd=cwd, transport=transport, call_timeout=call_timeout)
        self.parameters = parameters or {"type": "object", "properties": {},
                                         "additionalProperties": True}

    def run(self, input: dict | str | None = None, options: dict | None = None, /,
            **kwargs: Any) -> Any:
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
        arguments = (dict(input) if isinstance(input, dict)
                     else {"input": input} if isinstance(input, str) and input else {})
        if clash := sorted(set(arguments) & set(kwargs)):
            raise ValueError(f"MCP tool {self.name!r}: {clash} given both in the input dict and as "
                             "keyword arguments — pass each argument once")
        if options:
            raise ValueError(f"MCP tool {self.name!r}: "
                             + ("timeout is a setting — MCP(…, call_timeout=…)"
                                if "timeout" in options else
                                f"a call takes the server's arguments, not options {sorted(options)}"))
        # The bridge's cap sits above the call's and the handshake's, so the
        # server's own timeout fires first and names itself.
        limit = self._conn.call_timeout
        return _run_sync(self._acall({**arguments, **kwargs}),
                         timeout=None if limit is None else limit + DEFAULT_MCP_SETUP_TIMEOUT + 5)

    async def _acall(self, arguments: dict) -> Any:
        client = _build_client(self._conn)
        # The handshake has its own cap: a STDIO server that spawns but never
        # speaks the protocol would otherwise hang here.
        try:
            await asyncio.wait_for(client.__aenter__(), DEFAULT_MCP_SETUP_TIMEOUT)
        except asyncio.TimeoutError:
            raise TimeoutError(f"MCP server did not complete the connection handshake within "
                               f"{DEFAULT_MCP_SETUP_TIMEOUT:.0f}s: {_redacted(self._conn.spec)}"
                               ) from None
        limit = self._conn.call_timeout
        try:
            result = await client.call_tool(self._remote, arguments, raise_on_error=True,
                                            **({} if limit is None else {"timeout": limit}))
        finally:
            await client.__aexit__(None, None, None)
        return _extract_result(result)

    def __repr__(self) -> str:
        srv = _redacted(self._conn.spec)
        return f"MCPTool(name={self.name!r}, server={srv[:47] + '...' if len(srv) > 50 else srv!r})"


def _model_name(name: str) -> str:
    """A server's tool name as providers take one: letters, digits, ``_`` and
    ``-``, at most 64. ``files.read`` and ``fs/list`` were a 400 from OpenAI."""
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
    if missing := sorted(set(filter or ()) - {t.name for t in listed}):
        raise KeyError(f"filter= names {missing}, which the server does not have; "
                       f"it has {[t.name for t in listed]}")
    # A description is bounded: it goes into every request, and an unbounded
    # one is where a hostile server hides its instructions. `input_schema`
    # since MCP SDK v2, `inputSchema` before it.
    return [MCPTool(t.name, _bounded(getattr(t, "description", "") or ""), server,
                    parameters=getattr(t, "input_schema", None) or getattr(t, "inputSchema", None),
                    risk=risk_from_annotations(getattr(t, "annotations", None)))
            for t in listed if filter is None or t.name in filter]


def _bounded(text: str) -> str:
    return text if len(text) <= MAX_DESCRIPTION_CHARS else text[:MAX_DESCRIPTION_CHARS] + "…"


def _hint(annotations: Any, snake: str, camel: str) -> bool | None:
    """One hint, under its MCP SDK v2 name or the camelCase one before it."""
    found = ((annotations.get(camel, annotations.get(snake)),) if isinstance(annotations, dict)
             else (getattr(annotations, snake, None), getattr(annotations, camel, None)))
    value = next((v for v in found if v is not None), None)
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
    if _hint(annotations, "read_only_hint", "readOnlyHint") is True:
        return "read"
    destructive = _hint(annotations, "destructive_hint", "destructiveHint")
    if destructive is None:
        return "privileged"
    if destructive:
        return "destructive"
    return "write" if _hint(annotations, "open_world_hint", "openWorldHint") is False else "external"
