"""
A run's data is the run's, not the agent's.

Phase 1 of PLAN-agent-chain.md: an agent is configuration, and what one
invocation accumulates — its id, a parked document, a pending approval, the
hooks it reports to — lives on an `AgentRun` instead of on the agent. Before,
`_fork()` copied the whole agent per run to keep concurrent runs apart.

Pinned here beside the stream and approval tests in
`test_runs_do_not_share_state.py`: runs fanned out by a Pool keep to
themselves, an agent comes back from a run as it went in, and what survives a
pause is decided in one place — which found that an unchecked change made
before a pause was forgotten by it.
"""

import json
import os
import re
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from yait_aichain.agent import Agent, step_count, verified
from yait_aichain.models import Model
from yait_aichain.pool import Pool
from yait_aichain.state import InMemoryStore, SuspendedResult
from yait_aichain.state._run import AgentRun
from yait_aichain.tools import Wait, tool

USAGE = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}


def _text(content):
    return {"choices": [{"message": {"content": content}}], "usage": USAGE}


def _calls(*specs):
    return {"choices": [{"message": {"content": None, "tool_calls": [
        {"id": i, "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}
        for i, n, a in specs]}}], "usage": USAGE}


def _by_task():
    """A model that refunds the amount named in the task, then answers."""
    m = Model("gpt-4o", api_key="k")

    def side(path, body, headers, *a, **k):
        messages = body["messages"]
        task = next(x["content"] for x in messages if x["role"] == "user")
        if not any(x["role"] == "tool" for x in messages):
            amount = int(re.search(r"refund (\d+)", json.dumps(task)).group(1))
            return json.dumps(_calls(("c", "refund", {"amount": amount}))).encode()
        return json.dumps(_text("done")).encode()

    m.client._post = MagicMock(side_effect=side)

    def sse(path, body, headers, *a, **k):
        message = json.loads(side(path, body, headers))["choices"][0]["message"]
        if message.get("content"):
            yield {"choices": [{"delta": {"content": message["content"]}}]}
        for index, tc in enumerate(message.get("tool_calls") or []):
            yield {"choices": [{"delta": {"tool_calls": [
                {"index": index, "id": tc["id"], "function": tc["function"]}]}}]}
        yield {"choices": [], "usage": USAGE}

    m.client._post_sse = MagicMock(side_effect=sse)
    m.client._auth_headers = MagicMock(return_value={})
    return m


def _script(*replies):
    m = Model("gpt-4o", api_key="k")
    m.sent = []
    turn = [0]

    def side(path, body, headers, *a, **k):
        m.sent.append(body)
        reply = replies[min(turn[0], len(replies) - 1)]
        turn[0] += 1
        return json.dumps(reply).encode()

    m.client._post = MagicMock(side_effect=side)
    m.client._auth_headers = MagicMock(return_value={})
    return m


@tool(risk="read")
def refund(amount: int) -> str:
    """Refund an amount."""
    return f"refunded {amount}"


@tool(risk="write")
def edit(text: str = "") -> str:
    """Change a file."""
    return "changed"


class TestEightRunsThroughAPool(unittest.TestCase):

    def test_each_keeps_its_own_id_and_its_own_call(self):
        seen: list = []
        agent = Agent(_by_task(), tools=[refund], hooks=[seen.append])
        Pool(agent, items=[{"task": f"refund {n}"} for n in range(1, 9)], max_flows=8).run()
        calls = {}
        for e in seen:
            if e.type == "tool_call.started":
                calls.setdefault(e.run_id, []).append(e.payload["arguments"]["amount"])
        self.assertEqual(len(calls), 8, "two runs shared an id, or one had none")
        self.assertEqual(sorted(a for amounts in calls.values() for a in amounts), list(range(1, 9)))
        self.assertTrue(all(len(amounts) == 1 for amounts in calls.values()))


class TestTheAgentComesBackAsItWent(unittest.TestCase):

    def test_no_run_data_is_left_on_the_agent(self):
        agent = Agent(_by_task(), tools=[refund])
        hooks_before = list(agent.hooks)
        agent.run("refund 3")
        list(agent.stream("refund 4"))
        self.assertEqual(agent.hooks, hooks_before)
        self.assertIsNone(agent._idle)              # the runs had runs of their own
        for gone in ("_run_id", "_doc_id", "_pending", "_streaming", "_failure"):
            self.assertNotIn(gone, vars(agent))


class TestWhatSurvivesAPause(unittest.TestCase):

    def test_portable_state_round_trips_as_json(self):
        state = {"steps": 3, "tokens": 9, "cost": 0.5, "plan": ["a"], "plan_calls": 1,
                 "unverified": "edit", "nudged": {"0:stalled"}, "adaptations": [{"x": 1}],
                 "truncated": True}
        kept = AgentRun.portable(state)
        self.assertEqual(json.loads(json.dumps(kept)), kept)
        self.assertNotIn("adaptations", kept)
        self.assertNotIn("truncated", kept)
        self.assertEqual(kept["unverified"], "edit")

    def test_an_unchecked_change_made_before_a_pause_is_checked_after_it(self):
        model = _script(_calls(("e", "edit", {"text": "x"})),
                        _calls(("w", "wait", {})),
                        _text("done"), _text("done, checked"))
        agent = Agent(model, tools=[edit, Wait(reason="ok?")], store=InMemoryStore(),
                      stop_when=[step_count(8), verified()])
        parked = agent.run("change it")
        self.assertIsInstance(parked, SuspendedResult)
        result = agent.resume(parked.run_id, signal={"ok": True})
        held = [b for b in model.sent if "have not looked at the result" in json.dumps(b)]
        self.assertEqual(len(held), 1, "the change made before the pause was forgotten by it")
        self.assertEqual(result.output, "done, checked")


if __name__ == "__main__":
    unittest.main()
