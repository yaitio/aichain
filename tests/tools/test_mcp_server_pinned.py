"""
What `_Server` settles for an MCP server, pinned before it was reshaped: the
transport each form of server gets, every timeout reaching what it caps, and
no header or env value in a repr, an error or a saved chain. Offline: the
fastmcp client is replaced, nothing is reached and no process is started.
"""

import asyncio
import os
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("OPENAI_API_KEY", "k")

from yait_aichain.utilities import MCP
from yait_aichain.utilities.mcp import _tools
from yait_aichain.utilities.mcp._tools import MCPTool, _Server, _build_client

SECRET = "SECRET-VALUE-1234"


def _transport(spec, **settings):
    with patch("fastmcp.Client", side_effect=lambda t: t):
        return _build_client(_Server(spec, **settings))


class TestTheTransportOfEachForm(unittest.TestCase):

    def test_a_url(self):
        t = _transport("https://h.example/mcp", headers={"A": "1"})
        self.assertEqual((type(t).__name__, t.url, t.headers),
                         ("StreamableHttpTransport", "https://h.example/mcp", {"A": "1"}))
        self.assertEqual(type(_transport("http://h.example/sse/")).__name__, "SSETransport")

    def test_a_transport_named_wins_over_the_path(self):
        self.assertEqual(type(_transport("https://h.example/mcp", transport="sse")).__name__,
                         "SSETransport")
        for name in ("http", "streamable"):
            self.assertEqual(type(_transport("https://h.example/sse", transport=name)).__name__,
                             "StreamableHttpTransport")

    def test_a_url_in_a_dict_takes_the_dicts_headers(self):
        t = _transport({"url": "https://h.example/mcp", "headers": {"A": "1"}, "transport": "sse"})
        self.assertEqual((type(t).__name__, t.headers), ("SSETransport", {"A": "1"}))

    def test_a_command_line(self):
        t = _transport("fli-mcp --port 1", env={"K": "v"}, cwd="/srv")
        self.assertEqual((type(t).__name__, t.command, t.args, t.env, t.cwd),
                         ("StdioTransport", "fli-mcp", ["--port", "1"], {"K": "v"}, "/srv"))

    def test_a_command_in_a_dict(self):
        t = _transport({"command": "srv", "args": ["-v"], "env": {"K": "v"}, "cwd": "/srv"})
        self.assertEqual((t.command, t.args, t.env, t.cwd), ("srv", ["-v"], {"K": "v"}, "/srv"))
        t = _transport({"command": "srv"}, env={"K": "v"}, cwd="/x")
        self.assertEqual((t.command, t.args, t.env, t.cwd), ("srv", [], {"K": "v"}, "/x"))

    def test_a_script_path(self):
        t = _transport({"command": "tools/server.py", "args": ["a"]})
        self.assertEqual((t.command, t.args), (sys.executable, ["tools/server.py", "a"]))
        self.assertEqual((_transport("srv.js x").command, _transport("srv.js x").args),
                         ("node", ["srv.js", "x"]))

    def test_an_in_process_server_is_handed_to_the_client(self):
        server = object()
        with patch("fastmcp.Client", side_effect=lambda t: ("client", t)):
            self.assertEqual(_build_client(_Server(server)), ("client", server))


def _client(listed=(), result=None):
    c = AsyncMock()
    c.__aenter__ = AsyncMock(return_value=c)
    c.__aexit__ = AsyncMock(return_value=False)
    c.list_tools = AsyncMock(return_value=list(listed))
    c.call_tool = AsyncMock(return_value=result)
    return c


class TestEveryTimeoutReachesWhatItCaps(unittest.TestCase):

    def test_connect_timeout_caps_the_discovery(self):
        with patch.object(_tools, "_run_sync", side_effect=lambda coro, **kw: coro.close() or []) as run:
            self.assertEqual(MCP("https://h.example/mcp", connect_timeout=7).tools, [])
        self.assertEqual(run.call_args.kwargs, {"timeout": 7, "setting": "connect_timeout"})

    def test_call_timeout_caps_the_call_and_the_bridge_sits_above_it(self):
        seen = []
        real = _tools._run_sync

        def spy(coro, **kw):
            seen.append(kw)
            return real(coro, **kw)

        result = MagicMock(content=[], structured_content=None, data="ok")
        for limit, bridge in ((30.0, 30.0 + _tools.DEFAULT_MCP_SETUP_TIMEOUT + 5), (None, None)):
            client = _client(result=result)
            with patch.object(_tools, "_build_client", return_value=client), \
                 patch.object(_tools, "_run_sync", side_effect=spy):
                self.assertEqual(MCPTool("t", "", "https://h.example/mcp", call_timeout=limit).run(), "ok")
            self.assertEqual(seen.pop()["timeout"], bridge)
            self.assertEqual(client.call_tool.call_args.kwargs.get("timeout"), limit)

    def test_the_handshake_has_its_own_cap_and_names_no_secret(self):
        async def hang(*_):
            await asyncio.sleep(5)

        client = _client()
        client.__aenter__ = hang
        tool = MCPTool("t", "", f"https://h.example/mcp?key={SECRET}")
        with patch.object(_tools, "_build_client", return_value=client), \
             patch.object(_tools, "DEFAULT_MCP_SETUP_TIMEOUT", 0.05):
            with self.assertRaisesRegex(TimeoutError, "handshake") as ctx:
                tool.run()
        self.assertNotIn(SECRET, str(ctx.exception))
        client.call_tool.assert_not_called()


class TestNoSecretIsShownOrSaved(unittest.TestCase):

    FORMS = (("https://h.example/mcp", {"headers": {"Authorization": f"Bearer {SECRET}"}}),
             ("srv", {"env": {"API_KEY": SECRET}}),
             (f"https://h.example/mcp?api_key={SECRET}", {}),
             ({"url": "https://h.example/mcp", "headers": {"X-Key": SECRET}}, {}),
             ({"command": "srv", "env": {"API_KEY": SECRET}}, {}))

    def test_a_repr(self):
        for spec, settings in self.FORMS:
            with self.subTest(spec=spec, **settings):
                shown = (repr(MCP(spec, **settings)) + repr(MCPTool("t", "", spec, **settings))
                         + repr(_Server(spec, **settings)) + str(_Server(spec, **settings)))
                self.assertNotIn(SECRET, shown)

    def test_a_saved_chain(self):
        from yait_aichain import Chain
        for spec, settings in self.FORMS:
            with self.subTest(spec=spec, **settings):
                path = os.path.join(tempfile.mkdtemp(), "chain.yaml")
                with self.assertRaisesRegex(TypeError, "credentials"):
                    Chain(steps=[(MCPTool("t", "", spec, **settings), "x")]).save(path)
                self.assertFalse(os.path.exists(path) and SECRET in open(path).read())


if __name__ == "__main__":
    unittest.main()
