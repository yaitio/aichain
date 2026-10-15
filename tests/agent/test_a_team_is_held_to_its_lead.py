"""
A team member is held to its lead, and the lead sees what it did.

Phase 4 of PLAN-agent-chain.md. A worker given as a team member kept only its
own settings: its spend passed by the lead's `max_cost` (measured: a lead with
a $100 budget reported $3.75 and its budget recorded $2.50 — the worker's part
was never charged), and the lead's hooks saw nothing the worker did. The
lead's rules already govern it (phase 3, `test_dispatch_matrix.py`).

Also here: `delegate` and `write_plan` are asked about like any tool, as
`draft`, where they ran whatever the policy said; and a tool may ask for the
run it is called in with a keyword-only `run`, which no model ever fills.
"""

import json
import os
import sys
import unittest
import warnings
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from yait_aichain import Budget
from yait_aichain.agent import Agent, step_count
from yait_aichain.models import Model
from yait_aichain.state._run import Run
from yait_aichain.tools import PermissionPolicy, tool

PRICED = {"prompt_tokens": 100000, "completion_tokens": 100000, "total_tokens": 200000}


def _model(*replies):
    m = Model("gpt-4o", api_key="k")
    turn = [0]

    def side(*a, **k):
        reply = replies[min(turn[0], len(replies) - 1)]
        turn[0] += 1
        return json.dumps(reply).encode()

    m.client._post = MagicMock(side_effect=side)
    m.client._auth_headers = MagicMock(return_value={})
    return m


def _text(content):
    return {"choices": [{"message": {"content": content}}], "usage": PRICED}


def _call(name, arguments, call_id="d1"):
    return {"choices": [{"message": {"content": None, "tool_calls": [
        {"id": call_id, "type": "function",
         "function": {"name": name, "arguments": json.dumps(arguments)}}]}}], "usage": PRICED}


def _team(**lead_kw):
    worker = Agent(_model(_text("worker done")), name="w", description="Does the work.")
    lead = Agent(_model(_call("delegate", {"task": "t", "worker": "w"}), _text("done")),
                 team=[worker], stop_when=[step_count(4)], **lead_kw)
    return lead, worker


class TestTheLeadsBudgetCoversTheTeam(unittest.TestCase):

    def test_a_members_spend_is_charged_to_the_leads_budget(self):
        budget = Budget(100.0)
        lead, _ = _team(max_cost=budget)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = lead.run("go")
        self.assertGreater(result.cost, 0)
        self.assertAlmostEqual(budget.spent, result.cost)


class TestTheLeadSeesWhatTheTeamDid(unittest.TestCase):

    def test_the_leads_hooks_get_the_workers_events_named_by_the_call(self):
        seen = []
        lead, worker = _team(hooks=[seen.append])
        lead.run("go")
        inside = [e for e in seen if (e.payload or {}).get("parent_call_id") == "d1"]
        self.assertIn("run.started", [e.type for e in inside])
        self.assertIn("llm_call.ended", [e.type for e in inside])
        self.assertEqual(len({e.run_id for e in inside}), 1)
        self.assertNotIn(inside[0].run_id, {e.run_id for e in seen if e not in inside})
        self.assertEqual(worker.hooks, [])                       # nothing left on the worker


class TestBuiltinsAreAskedAbout(unittest.TestCase):

    def test_a_policy_can_refuse_delegate(self):
        lead, _ = _team(permissions=PermissionPolicy({"draft": "deny"}))
        seen = []
        lead.hooks.append(seen.append)
        lead.run("go")
        ended = next(e for e in seen if e.type == "tool_call.ended")
        self.assertIn("denied by policy", ended.error or "")
        self.assertNotIn("parent_call_id", json.dumps([e.payload for e in seen], default=str))

    def test_with_no_policy_they_run_as_before(self):
        seen = []
        lead, _ = _team(hooks=[seen.append])
        lead.run("go")
        ended = next(e for e in seen if e.type == "tool_call.ended" and e.name == "delegate")
        self.assertIsNone(ended.error)


class TestAToolMayAskForItsRun(unittest.TestCase):

    def test_a_keyword_only_run_is_handed_in_and_not_in_the_schema(self):
        got = {}

        @tool(risk="read")
        def where(note: str, *, run: Run) -> str:
            """Say where it ran."""
            got["run"] = run
            return note

        self.assertEqual(list(where.parameters["properties"]), ["note"])
        agent = Agent(_model(_call("where", {"note": "hi"}, "c1"), _text("done")), tools=[where],
                      stop_when=[step_count(4)])
        agent.run("go")
        self.assertIsInstance(got["run"], Run)
        self.assertIsNotNone(got["run"].id)

    def test_an_ordinary_argument_named_run_stays_the_models(self):
        @tool(risk="read")
        def launch(run: str) -> str:
            """Start a run by name."""
            return run

        self.assertEqual(list(launch.parameters["properties"]), ["run"])


if __name__ == "__main__":
    unittest.main()
