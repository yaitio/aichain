"""
Two things the model cannot tell apart are refused, and a rename renames.

A user tool named ``pool``, ``delegate`` or ``write_plan`` was answered by the
loop's own; workers sharing a name were one entry in the enum; two
``Vectors`` or two ``Shell`` could not be renamed, so the recipe in the error
did not work; a renamed MCP tool sent its new name to the server; names with
``.`` or ``/`` were a 400 at OpenAI; a listed worker sat at depth 0, so the
depth guard never fired; an ``auto`` worker got tools that park a run; and a
worker's error was lost.
"""

import os
import sys
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from yait_aichain.agent import Agent                                   # noqa: E402
from yait_aichain.models import Model                                  # noqa: E402
from yait_aichain.tools._base import Tool                              # noqa: E402
from yait_aichain.utilities import Shell                               # noqa: E402
from yait_aichain.utilities.mcp import MCPTool                         # noqa: E402


class Named(Tool):
    parameters = {"type": "object", "properties": {}}

    def __init__(self, name):
        self.name = name

    def run(self):
        return "x"


class TestRefused(unittest.TestCase):

    def test_a_reserved_name_is_refused(self):
        for name in ("pool", "delegate", "write_plan"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "agent's own"):
                Agent(Model("gpt-4o", api_key="k"), tools=[Named(name)])

    def test_two_workers_of_one_name_are_refused(self):
        a = Agent(Model("gpt-4o", api_key="k"), name="w")
        b = Agent(Model("gpt-4o", api_key="k"), name="w")
        with self.assertRaisesRegex(ValueError, "share a name"):
            Agent(Model("gpt-4o", api_key="k"), team=[a, b])


class TestRenamed(unittest.TestCase):

    def test_two_shells_with_prefixes_live_in_one_agent(self):
        tools = [*Shell(".", prefix="a_").tools, *Shell(".", prefix="b_").tools]
        Agent(Model("gpt-4o", api_key="k"), tools=tools)
        self.assertTrue(all(t.name.startswith(("a_", "b_")) for t in tools))

    def test_an_mcp_tool_calls_the_server_by_its_own_name(self):
        tool = MCPTool("files.read", "Read.", "https://example.com/mcp")
        self.assertEqual(tool.name, "files_read")
        tool.name = "renamed"
        client = AsyncMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        client.call_tool = AsyncMock(return_value=type("R", (), {
            "content": [], "structured_content": None, "data": "ok"})())
        with patch("yait_aichain.utilities.mcp._tools._build_client", return_value=client):
            tool.run()
        self.assertEqual(client.call_tool.call_args.args[0], "files.read")


class TestDelegation(unittest.TestCase):

    def test_a_listed_worker_runs_one_level_down(self):
        worker = Agent(Model("gpt-4o", api_key="k"), name="w")
        boss = Agent(Model("gpt-4o", api_key="k"), team=[worker])
        boss._depth = boss._max_depth
        with self.assertRaisesRegex(RuntimeError, "depth"):
            boss._do_delegate({"task": "t", "worker": "w"}, boss.new_state())

    def test_an_auto_worker_gets_no_tool_that_parks(self):
        from yait_aichain.tools import Wait
        seen = {}
        from yait_aichain.state import InMemoryStore
        boss = Agent(Model("gpt-4o", api_key="k"), tools=[Named("look"), Wait(reason="r")],
                     team="auto", store=InMemoryStore())

        def fake_run(self, task, variables=None):
            seen["tools"] = [t.name for t in self.tools]
            from yait_aichain.agent import AgentResult
            return AgentResult(output=None, success=False, error="boom", mode="agile",
                               steps_taken=0, tokens_used=0)
        with patch.object(Agent, "run", fake_run):
            out = boss._do_delegate({"task": "t"}, boss.new_state())
        self.assertEqual(seen["tools"], ["look"])
        self.assertEqual(out["error"], "boom")


if __name__ == "__main__":
    unittest.main()
