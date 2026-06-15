"""
tests.state.test_agent_suspend
==============================

Agent ↔ state: a Wait/Gate tool pauses the agent loop and parks a run document
(stage 2a — suspend). The orchestrator is stubbed (no network).
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from agent import Agent
from tools import Wait, Gate
from tools._base import Tool
from models import Model
from state import SuspendedResult


def _resp(obj):
    return json.dumps({"choices": [{"message": {"content": json.dumps(obj)}}]}).encode()


def _orchestrator(*responses):
    """A Model whose transport returns the given JSON responses in order."""
    m = Model("gpt-4o", api_key="k")
    enc = [_resp(r) for r in responses]
    call = [0]
    def side(*a, **k):
        i = min(call[0], len(enc) - 1); call[0] += 1; return enc[i]
    m.client._post = MagicMock(side_effect=side)
    m.client._auth_headers = MagicMock(return_value={})
    return m


class TestAgentSuspend(unittest.TestCase):

    def _run_until_wait(self, tool):
        plan   = {"steps": [{"id": 1, "type": "tool",
                             "tool_name": tool.name, "goal": "await"}]}
        action = {"type": "tool", "tool_name": tool.name, "kwargs": {}}
        agent  = Agent(orchestrator=_orchestrator(plan, action), tools=[tool])
        return agent, agent.run("do it")

    def test_wait_suspends_agent(self):
        agent, res = self._run_until_wait(
            Wait(reason="approval", resume_with={"approved": "bool"}))
        self.assertIsInstance(res, SuspendedResult)
        self.assertFalse(res)
        self.assertEqual(res.awaiting["reason"], "approval")

    def test_document_is_agent_kind_and_persisted(self):
        agent, res = self._run_until_wait(Wait(reason="r"))
        doc = agent._store.load(res.run_id)
        self.assertEqual(doc["kind"], "agent")
        self.assertEqual(doc["status"], "suspended")
        self.assertEqual(doc["steps"][0]["status"], "suspended")

    def test_resume_blob_captures_pending_action(self):
        agent, res = self._run_until_wait(Wait(reason="r"))
        defn = agent._store.load(res.run_id)["definition"]
        self.assertEqual(defn["pending_action"]["tool_name"], "wait")
        self.assertIn("plan", defn)
        self.assertEqual(defn["step_idx"], 0)
        self.assertEqual(defn["task"], "do it")


_REFLECT = {"decision": "continue", "store_as": "approval"}


class TestAgentResume(unittest.TestCase):

    def _suspend(self, tool, store=None):
        plan   = {"steps": [{"id": 1, "type": "tool",
                             "tool_name": tool.name, "goal": "g"}]}
        action = {"type": "tool", "tool_name": tool.name, "kwargs": {}}
        kw = {"store": store} if store is not None else {}
        agent  = Agent(orchestrator=_orchestrator(plan, action), tools=[tool], **kw)
        return agent, agent.run("do it")

    def test_wait_resume_completes(self):
        agent, res = self._suspend(Wait(reason="approval"))
        agent.orchestrator.client._post = MagicMock(side_effect=[_resp(_REFLECT)])
        out = agent.resume(res.run_id, signal={"approved": True})
        self.assertTrue(out.success)
        # the Wait output (the signal) is stored under the reflection's store_as
        self.assertEqual(out.memory.get("approval"), {"approved": True})

    def test_resume_cleans_store_and_is_idempotent(self):
        agent, res = self._suspend(Wait(reason="r"))
        agent.orchestrator.client._post = MagicMock(side_effect=[_resp(_REFLECT)])
        agent.resume(res.run_id, signal={"ok": True})
        self.assertIsNone(agent._store.load(res.run_id))
        with self.assertRaises(KeyError):
            agent.resume(res.run_id, signal={"ok": True})

    def test_gate_in_agent_runs_wrapped_tool_on_approval(self):
        ran = {"n": 0}
        class _Send(Tool):
            name = "send"
            parameters = {"type": "object", "properties": {}, "required": []}
            def run(self, **kw):
                ran["n"] += 1; return "sent"
        agent, res = self._suspend(Gate(_Send()))
        agent.orchestrator.client._post = MagicMock(side_effect=[_resp(_REFLECT)])
        out = agent.resume(res.run_id, signal={"approved": True})
        self.assertTrue(out.success)
        self.assertEqual(ran["n"], 1)

    def test_resume_cross_process_filestore(self):
        import tempfile
        from state import FileStore
        with tempfile.TemporaryDirectory() as d:
            _, res = self._suspend(Wait(reason="r"), store=FileStore(d))
            # a fresh Agent (other process) sharing only the store dir
            other = Agent(orchestrator=_orchestrator(_REFLECT),
                          tools=[Wait(reason="r")], store=FileStore(d))
            out = other.resume(res.run_id, signal={"ok": True})
            self.assertTrue(out.success)


if __name__ == "__main__":
    unittest.main()
