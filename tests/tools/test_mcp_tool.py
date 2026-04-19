"""
tests.tools.test_mcp_tool
==========================

Tests for ``tools.mcp`` — MCPTool, MCPTools, and the supporting helpers
(_is_sse, _extract_result, _AsyncBridge).

Test organisation
-----------------
TestMCPToolConstruction   — MCPTool.__init__: attributes, schema defaults
TestMCPToolRepr           — MCPTool.__repr__: format and truncation
TestTransportHelpers      — _is_sse(): SSE detection heuristic
TestExtractResult         — _extract_result(): result normalisation
TestMCPToolRunMocked      — MCPTool.run() with a mock fastmcp Client
TestMCPToolsFactoryMocked — MCPTools() factory with a mock fastmcp Client
TestMCPToolFliLive        — Live integration tests against fli-mcp (STDIO)

Live tests require ``fli-mcp`` on PATH (``pip install flights``).  They are
skipped automatically when it is not found.
"""

from __future__ import annotations

import os
import sys
import shutil
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tools.mcp import (
    MCPTool,
    MCPTools,
    _is_sse,
    _extract_result,
    _AsyncBridge,
    _MISSING,
)
from tools._base import ToolResult


# ---------------------------------------------------------------------------
# Small helpers / fixtures
# ---------------------------------------------------------------------------

_FLI_MCP = shutil.which("fli-mcp")
_SKIP_LIVE = unittest.skipUnless(
    _FLI_MCP,
    "fli-mcp not on PATH — skipping live tests (pip install flights)",
)

# Dates for live flight searches — far enough in the future
_DEPART_DATE = "2026-07-01"
_END_DATE    = "2026-07-15"


def _make_text_content(text: str):
    """Return a mock MCP TextContent block."""
    m = MagicMock()
    m.type = "text"
    m.text = text
    return m


def _make_image_content(data: str = "abc123", mime: str = "image/png"):
    """Return a mock MCP ImageContent block."""
    m = MagicMock()
    m.type     = "image"
    m.data     = data
    m.mimeType = mime
    return m


def _make_call_result(data=_MISSING, content=None):
    """Return a mock CallToolResult."""
    r = MagicMock()
    if data is not _MISSING:
        r.data = data
    else:
        # Simulate an object that doesn't have .data
        del r.data
    r.content = content or []
    return r


def _mock_client(list_tools_return=None, call_tool_return=None):
    """
    Return a mock fastmcp.Client that acts as an async context manager.
    Calls to list_tools() and call_tool() are pre-configured.
    """
    mock = AsyncMock()
    mock.__aenter__ = AsyncMock(return_value=mock)
    mock.__aexit__  = AsyncMock(return_value=False)
    if list_tools_return is not None:
        mock.list_tools = AsyncMock(return_value=list_tools_return)
    if call_tool_return is not None:
        mock.call_tool = AsyncMock(return_value=call_tool_return)
    return mock


def _make_mcp_tool_info(name: str, description: str = "", schema: dict | None = None):
    """Return a mock MCP Tool info object (as returned by list_tools())."""
    t = MagicMock()
    t.name        = name
    t.description = description
    t.inputSchema = schema or {
        "type": "object",
        "properties": {"q": {"type": "string", "description": "query"}},
        "required": ["q"],
    }
    return t


# ---------------------------------------------------------------------------
# MCPTool construction
# ---------------------------------------------------------------------------

class TestMCPToolConstruction(unittest.TestCase):

    def _make(self, **kw) -> MCPTool:
        defaults = dict(
            name        = "search",
            description = "Search for something.",
            server      = "https://example.com/mcp",
        )
        defaults.update(kw)
        return MCPTool(**defaults)

    def test_name_stored(self):
        t = self._make(name="my_tool")
        self.assertEqual(t.name, "my_tool")

    def test_description_stored(self):
        t = self._make(description="Does X.")
        self.assertEqual(t.description, "Does X.")

    def test_server_stored(self):
        t = self._make(server="https://api.example.com/mcp")
        self.assertEqual(t._server, "https://api.example.com/mcp")

    def test_default_schema_is_permissive(self):
        t = self._make()
        self.assertTrue(t.parameters.get("additionalProperties"))
        self.assertEqual(t.parameters["type"], "object")

    def test_custom_schema_preserved(self):
        schema = {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        }
        t = self._make(parameters=schema)
        self.assertEqual(t.parameters, schema)

    def test_optional_attrs_default_to_none(self):
        t = self._make()
        self.assertIsNone(t._headers)
        self.assertIsNone(t._env)
        self.assertIsNone(t._cwd)
        self.assertIsNone(t._transport)

    def test_headers_stored(self):
        t = self._make(headers={"Authorization": "Bearer tok"})
        self.assertEqual(t._headers, {"Authorization": "Bearer tok"})

    def test_env_stored(self):
        t = self._make(env={"API_KEY": "secret"})
        self.assertEqual(t._env, {"API_KEY": "secret"})

    def test_dict_server_stored(self):
        spec = {"command": "python", "args": ["server.py"]}
        t = self._make(server=spec)
        self.assertEqual(t._server, spec)

    def test_is_tool_subclass(self):
        from tools._base import Tool
        self.assertIsInstance(self._make(), Tool)


# ---------------------------------------------------------------------------
# MCPTool.__repr__
# ---------------------------------------------------------------------------

class TestMCPToolRepr(unittest.TestCase):

    def test_repr_contains_name(self):
        t = MCPTool("weather", "Get weather.", "https://example.com/mcp")
        self.assertIn("weather", repr(t))

    def test_repr_contains_server(self):
        t = MCPTool("x", "y", "https://example.com/mcp")
        self.assertIn("example.com", repr(t))

    def test_repr_truncates_long_url(self):
        long_url = "https://" + "a" * 60 + ".com/mcp"
        t = MCPTool("x", "y", long_url)
        r = repr(t)
        self.assertIn("...", r)
        self.assertLess(len(r), 120)

    def test_repr_dict_server(self):
        t = MCPTool("x", "y", {"command": "python", "args": ["s.py"]})
        self.assertIn("MCPTool", repr(t))


# ---------------------------------------------------------------------------
# _is_sse transport helper
# ---------------------------------------------------------------------------

class TestTransportHelpers(unittest.TestCase):

    def test_url_ending_in_sse_is_sse(self):
        self.assertTrue(_is_sse("https://api.example.com/sse", None))

    def test_url_ending_in_sse_trailing_slash(self):
        self.assertTrue(_is_sse("https://api.example.com/sse/", None))

    def test_plain_mcp_url_is_not_sse(self):
        self.assertFalse(_is_sse("https://api.example.com/mcp", None))

    def test_explicit_sse_hint_overrides(self):
        self.assertTrue(_is_sse("https://api.example.com/mcp", "sse"))

    def test_explicit_http_hint_suppresses_sse_detection(self):
        self.assertFalse(_is_sse("https://api.example.com/sse", "http"))

    def test_explicit_http_hint_on_plain_url(self):
        self.assertFalse(_is_sse("https://api.example.com/mcp", "http"))

    def test_streamable_hint_treated_as_http(self):
        self.assertFalse(_is_sse("https://api.example.com/sse", "streamable"))


# ---------------------------------------------------------------------------
# _extract_result normalisation
# ---------------------------------------------------------------------------

class TestExtractResult(unittest.TestCase):

    def test_data_str_returned_directly(self):
        r = _make_call_result(data="hello")
        self.assertEqual(_extract_result(r), "hello")

    def test_data_dict_returned_directly(self):
        d = {"flights": 3, "cheapest": 99.0}
        r = _make_call_result(data=d)
        self.assertEqual(_extract_result(r), d)

    def test_data_list_returned_directly(self):
        lst = [1, 2, 3]
        r   = _make_call_result(data=lst)
        self.assertEqual(_extract_result(r), lst)

    def test_data_int_returned_directly(self):
        r = _make_call_result(data=42)
        self.assertEqual(_extract_result(r), 42)

    def test_data_none_falls_through_to_content(self):
        r = _make_call_result(data=None, content=[_make_text_content("fallback")])
        self.assertEqual(_extract_result(r), "fallback")

    def test_missing_data_attribute_falls_through(self):
        # Object with no .data attribute at all
        r = _make_call_result(content=[_make_text_content("from content")])
        self.assertEqual(_extract_result(r), "from content")

    def test_single_text_content_returns_plain_string(self):
        r = _make_call_result(content=[_make_text_content("Just text.")])
        result = _extract_result(r)
        self.assertIsInstance(result, str)
        self.assertEqual(result, "Just text.")

    def test_multiple_text_blocks_return_list(self):
        r = _make_call_result(content=[
            _make_text_content("Part 1"),
            _make_text_content("Part 2"),
        ])
        result = _extract_result(r)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["text"], "Part 1")
        self.assertEqual(result[1]["text"], "Part 2")

    def test_image_content_preserved(self):
        r = _make_call_result(content=[_make_image_content("b64data", "image/jpeg")])
        result = _extract_result(r)
        self.assertIsInstance(result, list)
        self.assertEqual(result[0]["type"], "image")
        self.assertEqual(result[0]["data"], "b64data")
        self.assertEqual(result[0]["mime_type"], "image/jpeg")

    def test_mixed_text_and_image_returns_list(self):
        r = _make_call_result(content=[
            _make_text_content("caption"),
            _make_image_content(),
        ])
        result = _extract_result(r)
        self.assertIsInstance(result, list)
        self.assertEqual(result[0]["type"], "text")
        self.assertEqual(result[1]["type"], "image")

    def test_empty_content_returns_none(self):
        r = _make_call_result(content=[])
        self.assertIsNone(_extract_result(r))

    def test_no_data_no_content_returns_none(self):
        r = _make_call_result()   # no data attr, empty content
        self.assertIsNone(_extract_result(r))


# ---------------------------------------------------------------------------
# MCPTool.run() — mocked fastmcp Client
# ---------------------------------------------------------------------------

class TestMCPToolRunMocked(unittest.TestCase):
    """MCPTool.run() with _build_client patched to avoid network calls."""

    def _tool(self, **kw) -> MCPTool:
        return MCPTool(
            name        = "echo",
            description = "Echo back.",
            server      = "https://example.com/mcp",
            parameters  = {
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
            **kw,
        )

    def _run_with_mock(self, tool, arguments, result_data):
        """Call tool.run() with a mocked client returning result_data."""
        call_result = _make_call_result(data=result_data)
        mock = _mock_client(call_tool_return=call_result)
        with patch("tools.mcp._build_client", return_value=mock):
            return tool.run(input=arguments)

    # ── basic invocation ────────────────────────────────────────────────────

    def test_run_returns_string_result(self):
        result = self._run_with_mock(self._tool(), {"message": "hi"}, "hi back")
        self.assertEqual(result, "hi back")

    def test_run_returns_dict_result(self):
        result = self._run_with_mock(self._tool(), {"message": "q"}, {"answer": 42})
        self.assertEqual(result, {"answer": 42})

    def test_run_with_no_args_passes_empty_dict(self):
        call_result = _make_call_result(data="ok")
        mock = _mock_client(call_tool_return=call_result)
        with patch("tools.mcp._build_client", return_value=mock):
            result = self._tool().run()
        self.assertEqual(result, "ok")

    # ── argument merging ────────────────────────────────────────────────────

    def test_kwargs_are_merged_into_arguments(self):
        call_result = _make_call_result(data="ok")
        mock = _mock_client(call_tool_return=call_result)
        captured = {}

        async def capture_call(name, arguments, **kw):
            captured["args"] = arguments
            return call_result

        mock.call_tool = capture_call
        with patch("tools.mcp._build_client", return_value=mock):
            self._tool().run(message="hello via kwarg")
        self.assertEqual(captured["args"], {"message": "hello via kwarg"})

    def test_kwargs_win_over_input_dict(self):
        call_result = _make_call_result(data="ok")
        mock = _mock_client(call_tool_return=call_result)
        captured = {}

        async def capture_call(name, arguments, **kw):
            captured["args"] = arguments
            return call_result

        mock.call_tool = capture_call
        with patch("tools.mcp._build_client", return_value=mock):
            self._tool().run(input={"message": "from dict"}, message="from kwarg")
        self.assertEqual(captured["args"]["message"], "from kwarg")

    def test_string_input_stored_as_input_field(self):
        call_result = _make_call_result(data="ok")
        mock = _mock_client(call_tool_return=call_result)
        captured = {}

        async def capture_call(name, arguments, **kw):
            captured["args"] = arguments
            return call_result

        mock.call_tool = capture_call
        with patch("tools.mcp._build_client", return_value=mock):
            self._tool().run(input="plain string")
        self.assertEqual(captured["args"], {"input": "plain string"})

    def test_none_input_sends_empty_dict(self):
        call_result = _make_call_result(data="ok")
        mock = _mock_client(call_tool_return=call_result)
        captured = {}

        async def capture_call(name, arguments, **kw):
            captured["args"] = arguments
            return call_result

        mock.call_tool = capture_call
        with patch("tools.mcp._build_client", return_value=mock):
            self._tool().run(input=None)
        self.assertEqual(captured["args"], {})

    # ── __call__ (safe wrapper) ─────────────────────────────────────────────

    def test_call_style_returns_tool_result(self):
        call_result = _make_call_result(data="value")
        mock = _mock_client(call_tool_return=call_result)
        with patch("tools.mcp._build_client", return_value=mock):
            result = self._tool()(message="hi")
        self.assertIsInstance(result, ToolResult)
        self.assertTrue(result)
        self.assertEqual(result.output, "value")

    def test_call_style_error_returns_failure(self):
        mock = _mock_client()
        mock.call_tool = AsyncMock(side_effect=RuntimeError("tool failed"))
        with patch("tools.mcp._build_client", return_value=mock):
            result = self._tool()(message="hi")
        self.assertIsInstance(result, ToolResult)
        self.assertFalse(result)
        self.assertIn("tool failed", result.error)

    def test_call_style_never_raises(self):
        mock = _mock_client()
        mock.call_tool = AsyncMock(side_effect=RuntimeError("boom"))
        with patch("tools.mcp._build_client", return_value=mock):
            try:
                self._tool()(message="hi")
            except Exception as exc:
                self.fail(f"__call__ raised unexpectedly: {exc}")

    # ── timeout forwarded ───────────────────────────────────────────────────

    def test_timeout_forwarded_to_call_tool(self):
        call_result = _make_call_result(data="ok")
        mock = _mock_client(call_tool_return=call_result)
        captured = {}

        async def capture_call(name, arguments, **kw):
            captured["kw"] = kw
            return call_result

        mock.call_tool = capture_call
        with patch("tools.mcp._build_client", return_value=mock):
            self._tool().run(options={"timeout": 30.0})
        self.assertEqual(captured["kw"].get("timeout"), 30.0)

    def test_no_timeout_not_sent(self):
        call_result = _make_call_result(data="ok")
        mock = _mock_client(call_tool_return=call_result)
        captured = {}

        async def capture_call(name, arguments, **kw):
            captured["kw"] = kw
            return call_result

        mock.call_tool = capture_call
        with patch("tools.mcp._build_client", return_value=mock):
            self._tool().run()
        self.assertNotIn("timeout", captured.get("kw", {}))

    # ── correct tool name passed ────────────────────────────────────────────

    def test_correct_tool_name_passed_to_call_tool(self):
        call_result = _make_call_result(data="ok")
        mock = _mock_client(call_tool_return=call_result)
        captured = {}

        async def capture_call(name, arguments, **kw):
            captured["name"] = name
            return call_result

        mock.call_tool = capture_call
        tool = MCPTool("my_specific_tool", "desc", "https://example.com/mcp")
        with patch("tools.mcp._build_client", return_value=mock):
            tool.run()
        self.assertEqual(captured["name"], "my_specific_tool")


# ---------------------------------------------------------------------------
# MCPTools() factory — mocked
# ---------------------------------------------------------------------------

class TestMCPToolsFactoryMocked(unittest.TestCase):
    """MCPTools() with _build_client patched."""

    def test_returns_list(self):
        mock = _mock_client(list_tools_return=[
            _make_mcp_tool_info("search", "Search."),
            _make_mcp_tool_info("summarise", "Summarise."),
        ])
        with patch("tools.mcp._build_client", return_value=mock):
            tools = MCPTools("https://example.com/mcp")
        self.assertIsInstance(tools, list)
        self.assertEqual(len(tools), 2)

    def test_each_item_is_mcp_tool(self):
        mock = _mock_client(list_tools_return=[_make_mcp_tool_info("t1", "T1")])
        with patch("tools.mcp._build_client", return_value=mock):
            tools = MCPTools("https://example.com/mcp")
        self.assertIsInstance(tools[0], MCPTool)

    def test_tool_name_set_from_server(self):
        mock = _mock_client(list_tools_return=[_make_mcp_tool_info("weather", "Get wx.")])
        with patch("tools.mcp._build_client", return_value=mock):
            tools = MCPTools("https://example.com/mcp")
        self.assertEqual(tools[0].name, "weather")

    def test_tool_description_set_from_server(self):
        mock = _mock_client(list_tools_return=[_make_mcp_tool_info("wx", "Get weather data.")])
        with patch("tools.mcp._build_client", return_value=mock):
            tools = MCPTools("https://example.com/mcp")
        self.assertEqual(tools[0].description, "Get weather data.")

    def test_tool_schema_set_from_server(self):
        schema = {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        }
        mock = _mock_client(list_tools_return=[_make_mcp_tool_info("wx", "", schema)])
        with patch("tools.mcp._build_client", return_value=mock):
            tools = MCPTools("https://example.com/mcp")
        self.assertEqual(tools[0].parameters, schema)

    def test_tool_inherits_server_spec(self):
        server = "https://example.com/mcp"
        mock   = _mock_client(list_tools_return=[_make_mcp_tool_info("t")])
        with patch("tools.mcp._build_client", return_value=mock):
            tools = MCPTools(server)
        self.assertEqual(tools[0]._server, server)

    def test_tool_inherits_headers(self):
        hdrs = {"Authorization": "Bearer sk-x"}
        mock = _mock_client(list_tools_return=[_make_mcp_tool_info("t")])
        with patch("tools.mcp._build_client", return_value=mock):
            tools = MCPTools("https://example.com/mcp", headers=hdrs)
        self.assertEqual(tools[0]._headers, hdrs)

    def test_empty_server_returns_empty_list(self):
        mock = _mock_client(list_tools_return=[])
        with patch("tools.mcp._build_client", return_value=mock):
            tools = MCPTools("https://example.com/mcp")
        self.assertEqual(tools, [])

    def test_filter_includes_only_matching_tools(self):
        mock = _mock_client(list_tools_return=[
            _make_mcp_tool_info("search"),
            _make_mcp_tool_info("summarise"),
            _make_mcp_tool_info("translate"),
        ])
        with patch("tools.mcp._build_client", return_value=mock):
            tools = MCPTools("https://example.com/mcp", filter=["search", "translate"])
        self.assertEqual(len(tools), 2)
        self.assertEqual({t.name for t in tools}, {"search", "translate"})

    def test_filter_empty_list_returns_nothing(self):
        mock = _mock_client(list_tools_return=[_make_mcp_tool_info("search")])
        with patch("tools.mcp._build_client", return_value=mock):
            tools = MCPTools("https://example.com/mcp", filter=[])
        self.assertEqual(tools, [])

    def test_filter_none_returns_all(self):
        mock = _mock_client(list_tools_return=[
            _make_mcp_tool_info("a"), _make_mcp_tool_info("b"),
        ])
        with patch("tools.mcp._build_client", return_value=mock):
            tools = MCPTools("https://example.com/mcp", filter=None)
        self.assertEqual(len(tools), 2)


# ---------------------------------------------------------------------------
# AsyncBridge
# ---------------------------------------------------------------------------

class TestAsyncBridge(unittest.TestCase):

    def test_singleton(self):
        b1 = _AsyncBridge.get()
        b2 = _AsyncBridge.get()
        self.assertIs(b1, b2)

    def test_runs_coroutine_and_returns_result(self):
        import asyncio

        async def add(a, b):
            return a + b

        result = _AsyncBridge.get().run(add(3, 4))
        self.assertEqual(result, 7)

    def test_propagates_exception(self):
        import asyncio

        async def boom():
            raise ValueError("from coroutine")

        with self.assertRaises(ValueError) as ctx:
            _AsyncBridge.get().run(boom())
        self.assertIn("from coroutine", str(ctx.exception))

    def test_thread_is_daemon(self):
        bridge = _AsyncBridge.get()
        self.assertTrue(bridge._thread.daemon)


# ---------------------------------------------------------------------------
# Live integration tests — fli-mcp STDIO server
# ---------------------------------------------------------------------------

@_SKIP_LIVE
class TestMCPToolFliLive(unittest.TestCase):
    """
    Integration tests that launch fli-mcp as a real STDIO subprocess.

    These tests require:
      • fli-mcp on PATH:  pip install flights
      • A live internet connection (fli scrapes Google Flights)

    All assertions are structural — they verify the tool returns a
    non-empty result and has the expected schema, without hard-coding
    flight prices or availability (which change constantly).
    """

    _SERVER = {"command": "fli-mcp", "args": []}

    # ── Discovery ──────────────────────────────────────────────────────────

    def test_discovers_two_tools(self):
        tools = MCPTools(self._SERVER)
        self.assertEqual(len(tools), 2)

    def test_tool_names_are_search_flights_and_search_dates(self):
        tools = MCPTools(self._SERVER)
        names = {t.name for t in tools}
        self.assertIn("search_flights", names)
        self.assertIn("search_dates", names)

    def test_all_tools_have_descriptions(self):
        for t in MCPTools(self._SERVER):
            self.assertIsInstance(t.description, str)
            self.assertGreater(len(t.description), 0, f"{t.name} has empty description")

    def test_search_flights_required_params(self):
        tools = {t.name: t for t in MCPTools(self._SERVER)}
        t     = tools["search_flights"]
        req   = t.parameters.get("required", [])
        self.assertIn("origin",         req)
        self.assertIn("destination",    req)
        self.assertIn("departure_date", req)

    def test_search_dates_required_params(self):
        tools = {t.name: t for t in MCPTools(self._SERVER)}
        t     = tools["search_dates"]
        req   = t.parameters.get("required", [])
        self.assertIn("origin",      req)
        self.assertIn("destination", req)
        self.assertIn("start_date",  req)
        self.assertIn("end_date",    req)

    def test_tools_are_mcp_tool_instances(self):
        for t in MCPTools(self._SERVER):
            self.assertIsInstance(t, MCPTool)

    def test_tools_inherit_server_spec(self):
        for t in MCPTools(self._SERVER):
            self.assertEqual(t._server, self._SERVER)

    # ── MCPTool repr ────────────────────────────────────────────────────────

    def test_repr_format(self):
        tools = MCPTools(self._SERVER)
        for t in tools:
            r = repr(t)
            self.assertIn("MCPTool", r)
            self.assertIn(t.name,    r)

    # ── search_flights live call ────────────────────────────────────────────

    def test_search_flights_returns_non_empty_result(self):
        tools = {t.name: t for t in MCPTools(self._SERVER)}
        t     = tools["search_flights"]

        result = t.run(input={
            "origin":         "LAX",
            "destination":    "JFK",
            "departure_date": _DEPART_DATE,
        })

        self.assertIsNotNone(result, "search_flights returned None")
        if isinstance(result, str):
            self.assertGreater(len(result.strip()), 0, "search_flights returned empty string")
        elif isinstance(result, (list, dict)):
            self.assertTrue(bool(result), "search_flights returned empty list/dict")

    def test_search_flights_one_stop_filter(self):
        tools = {t.name: t for t in MCPTools(self._SERVER)}
        t     = tools["search_flights"]

        result = t.run(input={
            "origin":         "SFO",
            "destination":    "ORD",
            "departure_date": _DEPART_DATE,
            "max_stops":      "1",
        })

        self.assertIsNotNone(result)

    # ── search_flights via call-style (__call__) ────────────────────────────

    def test_search_flights_call_style_returns_tool_result(self):
        tools = {t.name: t for t in MCPTools(self._SERVER)}
        t     = tools["search_flights"]

        result = t(
            origin         = "BOS",
            destination    = "MIA",
            departure_date = _DEPART_DATE,
        )

        self.assertIsInstance(result, ToolResult)
        self.assertTrue(result, f"search_flights call failed: {result.error}")
        self.assertIsNotNone(result.output)

    # ── search_dates live call ──────────────────────────────────────────────

    def test_search_dates_returns_non_empty_result(self):
        tools = {t.name: t for t in MCPTools(self._SERVER)}
        t     = tools["search_dates"]

        result = t.run(input={
            "origin":      "LAX",
            "destination": "JFK",
            "start_date":  _DEPART_DATE,
            "end_date":    _END_DATE,
        })

        self.assertIsNotNone(result, "search_dates returned None")
        if isinstance(result, str):
            self.assertGreater(len(result.strip()), 0, "search_dates returned empty string")
        elif isinstance(result, (list, dict)):
            self.assertTrue(bool(result), "search_dates returned empty list/dict")

    def test_search_dates_round_trip(self):
        tools = {t.name: t for t in MCPTools(self._SERVER)}
        t     = tools["search_dates"]

        result = t.run(input={
            "origin":       "NYC",
            "destination":  "LAX",
            "start_date":   _DEPART_DATE,
            "end_date":     _END_DATE,
            "is_round_trip": True,
            "trip_duration": 7,
        })

        self.assertIsNotNone(result)

    # ── single MCPTool direct (skip discovery) ──────────────────────────────

    def test_direct_mcp_tool_without_discovery(self):
        """Build MCPTool directly without MCPTools() — same call works."""
        tool = MCPTool(
            name        = "search_flights",
            description = "Search for flights.",
            server      = self._SERVER,
        )

        result = tool.run(input={
            "origin":         "SEA",
            "destination":    "DEN",
            "departure_date": _DEPART_DATE,
        })

        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
