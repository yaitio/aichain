"""
Interceptors act on a call before and after it runs — and say so.

Phase 5 of PLAN-agent-chain.md. A hook watches; an interceptor acts: its
`before` may block a call or change its arguments, ahead of the policy, so
the policy decides on what will really run; its `after` may change the
result. Pinned: each of those, that every change is written down with what
it was and what it became, that one which raises blocks rather than being
skipped, that it may not swap the call for another, that they run in order,
and that a Chain and a Pool take them too.
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from yait_aichain import Block, Interceptor
from yait_aichain.agent import Agent, step_count
from yait_aichain.chain import Chain
from yait_aichain.models import Model
from yait_aichain.models._calls import ToolCall
from yait_aichain.pool import Pool
from yait_aichain.tools import PermissionPolicy
from yait_aichain.tools._base import Tool

USAGE = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}


class Shell(Tool):
    """Risk by argument: `ls` reads, anything else is privileged."""
    name = "shell"
    description = "Run a command."
    parameters = {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}
    risk = "privileged"

    def __init__(self):
        self.ran = []

    def risk_for(self, arguments):
        return "read" if arguments.get("cmd") == "ls" else "privileged"

    def run(self, cmd, options=None):
        self.ran.append(cmd)
        return f"ran {cmd}"


def _agent(cmd, tool, **kw):
    m = Model("gpt-4o", api_key="k")
    m.sent = []
    replies = [{"choices": [{"message": {"content": None, "tool_calls": [
                   {"id": "c1", "type": "function",
                    "function": {"name": "shell", "arguments": json.dumps({"cmd": cmd})}}]}}],
                "usage": USAGE},
               {"choices": [{"message": {"content": "done"}}], "usage": USAGE}]
    turn = [0]

    def side(path, body, headers, *a, **k):
        m.sent.append(body)
        reply = replies[min(turn[0], 1)]
        turn[0] += 1
        return json.dumps(reply).encode()

    m.client._post = MagicMock(side_effect=side)
    m.client._auth_headers = MagicMock(return_value={})
    return Agent(m, tools=[tool], stop_when=[step_count(4)], **kw)


class Rewrite(Interceptor):
    def __init__(self, to):
        self.to = to

    def before(self, call, run):
        return ToolCall(id=call.id, name=call.name, arguments={"cmd": self.to})


class TestBefore(unittest.TestCase):

    def test_a_block_stops_the_call_and_the_model_hears_why(self):
        shell = Shell()

        class No(Interceptor):
            def before(self, call, run):
                return Block("not on a Friday")

        agent = _agent("ls", shell, interceptors=[No()])
        agent.run("go")
        self.assertEqual(shell.ran, [])
        self.assertIn("not on a Friday", json.dumps(agent.model.sent[-1]))

    def test_changed_arguments_run_and_the_policy_decides_on_them(self):
        policy = PermissionPolicy({"privileged": "deny"})
        shell = Shell()
        _agent("rm -rf /", shell, permissions=policy, interceptors=[Rewrite("ls")]).run("go")
        self.assertEqual(shell.ran, ["ls"])                   # rewritten to a read, so allowed
        shell = Shell()
        _agent("ls", shell, permissions=policy, interceptors=[Rewrite("rm -rf /")]).run("go")
        self.assertEqual(shell.ran, [])                       # rewritten to privileged, so denied

    def test_a_change_is_written_down(self):
        seen = []
        result = _agent("pwd", Shell(), hooks=[seen.append],
                        interceptors=[Rewrite("ls")]).run("go")
        notes = [e for e in result.journal if e.get("intent") == "interceptor"]
        self.assertEqual(len(notes), 1)
        self.assertIn('"pwd"', notes[0]["reason"])
        self.assertIn('"ls"', notes[0]["reason"])
        self.assertEqual([e.type for e in seen].count("interceptor.changed"), 1)

    def test_one_that_raises_blocks(self):
        shell = Shell()

        class Broken(Interceptor):
            def before(self, call, run):
                raise RuntimeError("guard down")

        _agent("ls", shell, interceptors=[Broken()]).run("go")
        self.assertEqual(shell.ran, [])

    def test_it_may_not_swap_the_call_for_another(self):
        shell = Shell()

        class Swap(Interceptor):
            def before(self, call, run):
                return ToolCall(id=call.id, name="other", arguments={})

        _agent("ls", shell, interceptors=[Swap()]).run("go")
        self.assertEqual(shell.ran, [])

    def test_they_run_in_order(self):
        shell = Shell()
        _agent("pwd", shell, interceptors=[Rewrite("a"), Rewrite("b")]).run("go")
        self.assertEqual(shell.ran, ["b"])


class TestAfter(unittest.TestCase):

    def test_a_changed_result_is_what_the_model_reads(self):
        class Redact(Interceptor):
            def after(self, call, result, run):
                return "[redacted]"

        agent = _agent("ls", Shell(), interceptors=[Redact()])
        agent.run("go")
        sent = json.dumps(agent.model.sent[-1])
        self.assertIn("[redacted]", sent)
        self.assertNotIn("ran ls", sent)


class TestChainAndPool(unittest.TestCase):

    def test_a_chain_step(self):
        shell = Shell()
        Chain([(shell, "out")], interceptors=[Rewrite("ls")]).run(variables={"cmd": "pwd"})
        self.assertEqual(shell.ran, ["ls"])

    def test_a_pool_item(self):
        shell = Shell()

        class No(Interceptor):
            def before(self, call, run):
                return Block("no")

        Pool(shell, items=[{"cmd": "ls"}], interceptors=[No()], on_error="collect").run()
        self.assertEqual(shell.ran, [])


if __name__ == "__main__":
    unittest.main()
