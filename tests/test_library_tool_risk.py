"""
The library's own tools say what they risk.

Found 2026-10-11: every Shell tool and every MCP tool inherited the default
risk class, ``write``, which the shipped policy allows. With
``permissions=PermissionPolicy()`` on, an agent could run any shell command,
or any tool another server declared, without being asked. The default is
``write`` because the library cannot tell what a caller's function does — but
it wrote ``Shell.run``, and it can read what an MCP server says about its own.

Pinned: what each Shell tool is and what the policy then decides; how an MCP
tool's annotations become a class, in either spelling the MCP SDK has used;
that a tool the server says nothing about goes to a person; and that the
caller's ``risk=`` outranks the server's word.
"""

import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from yait_aichain.tools._permissions import APPROVE, ALLOW, DENY, PermissionPolicy
from yait_aichain.utilities import MCP, Shell
from yait_aichain.utilities.mcp import MCPTool, risk_from_annotations

POLICY = PermissionPolicy()


class TestShell(unittest.TestCase):

    def test_each_tool_says_what_it_risks(self):
        sh = Shell(tempfile.mkdtemp())
        got = {t.name: (t.risk, POLICY.decide(t)) for t in sh.tools}
        self.assertEqual(got, {
            "local_browse": ("read", ALLOW),
            "local_read":   ("read", ALLOW),
            "local_write":  ("write", ALLOW),
            "local_run":    ("privileged", APPROVE),   # any command, the host's rights
        })


def _ann(**hints):
    return SimpleNamespace(**hints)


class TestMCPAnnotations(unittest.TestCase):

    def test_the_table(self):
        cases = [
            (None,                                                   "privileged"),
            (_ann(),                                                 "privileged"),
            (_ann(read_only_hint=True),                              "read"),
            (_ann(read_only_hint=True, destructive_hint=True),       "read"),
            (_ann(destructive_hint=True),                            "destructive"),
            (_ann(destructive_hint=False, open_world_hint=False),    "write"),
            (_ann(destructive_hint=False, open_world_hint=True),     "external"),
            (_ann(destructive_hint=False),                           "external"),
            (_ann(read_only_hint=False),                             "privileged"),
            (_ann(open_world_hint=False),                            "privileged"),
        ]
        for annotations, risk in cases:
            with self.subTest(annotations):
                self.assertEqual(risk_from_annotations(annotations), risk)

    def test_both_spellings_and_a_dict(self):
        self.assertEqual(risk_from_annotations(_ann(readOnlyHint=True)), "read")
        self.assertEqual(risk_from_annotations({"destructiveHint": True}), "destructive")
        self.assertEqual(risk_from_annotations({"read_only_hint": True}), "read")

    def test_a_hint_that_is_not_a_boolean_settles_nothing(self):
        self.assertEqual(risk_from_annotations(_ann(read_only_hint="yes")), "privileged")

    def test_the_policy_then(self):
        for annotations, decision in ((None, APPROVE), (_ann(read_only_hint=True), ALLOW),
                                      (_ann(destructive_hint=True), DENY),
                                      (_ann(destructive_hint=False), APPROVE)):
            with self.subTest(annotations):
                tool = MCPTool("t", "d", "https://example.com/mcp",
                               risk=risk_from_annotations(annotations))
                self.assertEqual(POLICY.decide(tool), decision)


def _discovered(*pairs):
    return [MCPTool(name, "d", "https://example.com/mcp", risk=risk) for name, risk in pairs]


class TestTheMCPUtility(unittest.TestCase):

    def test_a_tool_made_by_hand_is_privileged(self):
        self.assertEqual(MCPTool("t", "d", "https://example.com/mcp").risk, "privileged")
        with self.assertRaisesRegex(ValueError, "risk 'safe' is not one of"):
            MCPTool("t", "d", "https://example.com/mcp", risk="safe")

    def test_discovery_reads_the_annotations(self):
        listed = [SimpleNamespace(name="ask", description="", input_schema=None,
                                  annotations=_ann(read_only_hint=True)),
                  SimpleNamespace(name="drop", description="", input_schema=None,
                                  annotations=_ann(destructive_hint=True)),
                  SimpleNamespace(name="other", description="", input_schema=None, annotations=None)]

        class Client:
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False
            async def list_tools(self): return listed

        with patch("yait_aichain.utilities.mcp._tools._build_client", return_value=Client()):
            tools = MCP("https://example.com/mcp").tools
        self.assertEqual({t.name: t.risk for t in tools},
                         {"ask": "read", "drop": "destructive", "other": "privileged"})

    def test_the_callers_word_outranks_the_servers(self):
        found = _discovered(("ask", "privileged"), ("drop", "read"))
        with patch("yait_aichain.utilities.mcp.MCPTools", return_value=found):
            m = MCP("https://example.com/mcp", risk={"ask": "read", "drop": "destructive"})
            self.assertEqual({t.name: t.risk for t in m.tools}, {"ask": "read", "drop": "destructive"})

    def test_mistakes_in_risk_are_said(self):
        with self.assertRaisesRegex(ValueError, "each must be one of"):
            MCP("https://example.com/mcp", risk={"ask": "safe"})
        with patch("yait_aichain.utilities.mcp.MCPTools", return_value=_discovered(("ask", "read"))):
            with self.assertRaisesRegex(KeyError, r"risk= names \['asc'\], which the server does not have"):
                MCP("https://example.com/mcp", risk={"asc": "read"}).tools


if __name__ == "__main__":
    unittest.main()
