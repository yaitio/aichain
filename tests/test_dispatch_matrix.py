"""
A tool the policy refuses runs on no path.

The invariant the dispatcher work (PLAN-agent-chain.md, phase 2) is built to
hold: one place calls `tool.run()`, and every rule — policy, approval, schema,
budget — is applied there. Before it, the rules live in five places, and a
path that skips them runs a refused tool with nothing said.

One row per way a tool can be reached. The rows that fail today are marked
`xfail(strict=True)` with the phase that closes them: when a phase lands and a
row starts passing, the strict marker turns it red until the marker is
removed — so the matrix cannot fall behind the code, and no row can be fixed
in silence. A new path of execution does not land without its row here.
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from yait_aichain.agent import Agent, step_count
from yait_aichain.chain import Chain
from yait_aichain.models import Model
from yait_aichain.pool import Pool
from yait_aichain.state import InMemoryStore, SuspendedResult
from yait_aichain.tools import PermissionPolicy, Wait, tool
from yait_aichain.tools._base import Tool

USAGE = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
DENY = PermissionPolicy({"destructive": "deny"})


class Nuke(Tool):
    """The refused tool. Counts how often it actually ran."""
    name = "nuke"
    description = "Delete everything."
    parameters = {"type": "object", "properties": {"target": {"type": "string"}}}
    risk = "destructive"

    def __init__(self):
        self.ran = 0

    def run(self, target="all", options=None):
        self.ran += 1
        return {"deleted": target}


def _text(content):
    return {"choices": [{"message": {"content": content}}], "usage": USAGE}


def _calls(*specs):
    return {"choices": [{"message": {"content": None, "tool_calls": [
        {"id": i, "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}
        for i, n, a in specs]}}], "usage": USAGE}


def _model(*replies):
    """A model replaying *replies* in order, buffered or streamed; the last repeats."""
    m = Model("gpt-4o", api_key="k")
    turn = [0]

    def next_reply():
        reply = replies[min(turn[0], len(replies) - 1)]
        turn[0] += 1
        return reply

    m.client._post = MagicMock(side_effect=lambda *a, **k: json.dumps(next_reply()).encode())

    def sse(*a, **k):
        message = next_reply()["choices"][0]["message"]
        if message.get("content"):
            yield {"choices": [{"delta": {"content": message["content"]}}]}
        for index, tc in enumerate(message.get("tool_calls") or []):
            yield {"choices": [{"delta": {"tool_calls": [
                {"index": index, "id": tc["id"], "function": tc["function"]}]}}]}
        yield {"choices": [], "usage": USAGE}

    m.client._post_sse = MagicMock(side_effect=sse)
    m.client._auth_headers = MagicMock(return_value={})
    return m


def _agent(*replies, tools=None, **kw):
    return Agent(_model(*replies), tools=tools, permissions=_agent.policy,
                 stop_when=[step_count(6)], **kw)


_agent.policy = DENY


class TestAgentPaths(unittest.TestCase):
    """The agent's own loop and its external seams — guarded today."""

    #: What each row expects of `nuke`. The control class below flips it, so
    #: a row that never reaches the tool cannot pass for one that guards it.
    expect_ran = 0

    def assertRan(self, nuke):
        if self.expect_ran:
            self.assertGreaterEqual(nuke.ran, 1, "the scenario never reached the tool")
        else:
            self.assertEqual(nuke.ran, 0)

    def test_run(self):
        nuke = Nuke()
        _agent(_calls(("c", "nuke", {})), _text("done"), tools=[nuke]).run("go")
        self.assertRan(nuke)

    def test_stream(self):
        nuke = Nuke()
        list(_agent(_calls(("c", "nuke", {})), _text("done"), tools=[nuke]).stream("go"))
        self.assertRan(nuke)

    def test_step_and_execute(self):
        nuke = Nuke()
        agent = _agent(_calls(("c", "nuke", {})), tools=[nuke])
        state = agent.new_state()
        reply = agent.step(agent.opening("go"), state)
        for call in reply.calls:
            agent.execute(call, state)
        self.assertRan(nuke)

    def test_resume(self):
        nuke = Nuke()
        agent = _agent(_calls(("w", "wait", {})), _calls(("c", "nuke", {})), _text("done"),
                       tools=[nuke, Wait(reason="ok?")], store=InMemoryStore())
        parked = agent.run("go")
        self.assertIsInstance(parked, SuspendedResult)
        agent.resume(parked.run_id, signal={"ok": True})
        self.assertRan(nuke)

    def test_pool_in_the_agent(self):
        nuke = Nuke()
        _agent(_calls(("p", "pool", {"tool": "nuke", "items": [{"target": "a"}, {"target": "b"}]})),
               _text("done"), tools=[nuke]).run("go")
        self.assertRan(nuke)

    def test_delegate_to_an_auto_worker(self):
        nuke = Nuke()
        # Lead and worker share the scripted model: the lead delegates, the
        # worker asks for nuke, then both answer.
        _agent(_calls(("d", "delegate", {"task": "clean up", "instructions": "you clean"})),
               _calls(("c", "nuke", {})), _text("done"),
               tools=[nuke], team="auto").run("go")
        self.assertRan(nuke)


class TestTheRowsReachTheTool(TestAgentPaths):
    """The same rows with no policy: `nuke` must run, or the row tests nothing."""

    expect_ran = 1

    def setUp(self):
        _agent.policy = None

    def tearDown(self):
        _agent.policy = DENY


class TestPathsOutsideTheLoop(unittest.TestCase):
    """Reached without the agent's gate. Each row names the phase that closes it."""

    def test_delegate_to_an_explicit_member(self):
        # Closed in phase 3, a phase early: the lead's rules govern whatever
        # its `delegate` call runs, and the worker runs inside it.
        nuke = Nuke()
        worker = Agent(_model(_calls(("c", "nuke", {})), _text("done")), tools=[nuke],
                       name="cleaner", description="Cleans up.")
        lead = Agent(_model(_calls(("d", "delegate", {"task": "clean up", "worker": "cleaner"})),
                            _text("done")),
                     team=[worker], permissions=DENY, stop_when=[step_count(4)])
        lead.run("go")
        self.assertEqual(nuke.ran, 0)

    def test_a_chain_step_on_its_own(self):
        nuke = Nuke()
        # A refused step fails the way any step fails — here, by raising.
        with self.assertRaises(PermissionError):
            Chain([(nuke, "out")], policy=DENY).run()
        self.assertEqual(nuke.ran, 0)

    def test_a_pool_item_on_its_own(self):
        nuke = Nuke()
        Pool(nuke, items=[{"target": "a"}], policy=DENY, on_error="collect").run()
        self.assertEqual(nuke.ran, 0)

    def test_a_chain_inside_the_agent(self):
        nuke = Nuke()

        @tool(risk="read")
        def tidy() -> str:
            """Tidy up."""
            Chain([(nuke, "out")]).run()
            return "tidied"

        _agent(_calls(("t", "tidy", {})), _text("done"), tools=[tidy]).run("go")
        self.assertEqual(nuke.ran, 0)

    def test_a_pool_inside_the_agent(self):
        nuke = Nuke()

        @tool(risk="read")
        def sweep() -> str:
            """Sweep."""
            Pool(nuke, items=[{"target": "a"}], on_error="collect").run()
            return "swept"

        _agent(_calls(("s", "sweep", {})), _text("done"), tools=[sweep]).run("go")
        self.assertEqual(nuke.ran, 0)



class TestOnePlaceRunsATool(unittest.TestCase):
    """The invariant itself, read off the source: `charged(...)` is how a tool
    is run, and it is called from `_dispatch.py` and from the paths not yet
    routed through it. A new caller fails this; so does a path that was
    routed and is still listed."""

    #: Where a tool runs outside the dispatcher, and the phase that removes it.
    NOT_YET = {
    }

    def test_only_the_dispatcher_runs_a_tool(self):
        import re
        from pathlib import Path
        package = Path(__file__).resolve().parents[1] / "yait_aichain"
        callers = {str(p.relative_to(package)) for p in package.rglob("*.py")
                   if "__pycache__" not in p.parts
                   and re.search(r"(?<!def )\bcharged\(", p.read_text())}
        self.assertEqual(callers - {"_dispatch.py"} - {"tools/_base.py"}, set(self.NOT_YET),
                         "a tool runs outside the dispatcher, or a routed path is still listed")


if __name__ == "__main__":
    unittest.main()
