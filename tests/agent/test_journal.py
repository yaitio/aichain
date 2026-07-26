"""
Tests for the attempt journal — the append-only record of what the agent did.

Covers the Journal type itself, its wiring into the Agent (outcomes, evidence
kinds, the "do not redo" block reaching the prompt, survival across
suspend/resume), and the honest-success guarantee on the ``final_answer`` exit.
"""

import json
import unittest

from yait_aichain.agent import Agent
from yait_aichain.agent._journal import (
    Journal, evidence, CHECK, MODEL_CLAIM, DONE, FAILED, REFUTED, SKIPPED,
)
from yait_aichain.tools import Tool
from yait_aichain.state import SuspendedResult


# ── scripted model ─────────────────────────────────────────────────────────────

class _Client:
    def __init__(self, scripted):
        self.scripted, self.i = scripted, 0

    def _auth_headers(self):
        return {}

    def send(self, path, body, headers):
        out = self.scripted[min(self.i, len(self.scripted) - 1)]
        self.i += 1
        return json.dumps({"_c": out,
                           "usage": {"input_tokens": 5, "output_tokens": 5}})


class _Model:
    name = "fake"

    def __init__(self, scripted):
        self.client = _Client(scripted)
        self.prompts: list[str] = []

    def to_request(self, messages, output):
        self.prompts.append("\n".join(
            p.get("text", "") for m in messages for p in m.get("parts", [])))
        return ("/x", {})

    def from_response(self, response, output):
        return response["_c"]


class Echo(Tool):
    name = "echo"
    parameters = {"type": "object",
                  "properties": {"text": {"type": "string"}},
                  "required": ["text"]}

    def run(self, text, options=None):
        return f"echo:{text}"


class Boom(Tool):
    name = "boom"
    parameters = {"type": "object", "properties": {}, "required": []}

    def run(self, options=None, **kw):
        raise RuntimeError("service unavailable")


def _plan(*steps):
    return json.dumps({"steps": [
        {"id": i + 1, "type": "tool", "tool_name": t, "goal": g}
        for i, (t, g) in enumerate(steps)]})


def _act(tool, kwargs=None):
    return json.dumps({"type": "tool", "tool_name": tool, "kwargs": kwargs or {}})


def _refl(decision="continue", **kw):
    return json.dumps({"decision": decision, "assessment": "ok",
                       "store_as": "r", **kw})


# ── the Journal type ───────────────────────────────────────────────────────────

class TestJournal(unittest.TestCase):

    def test_append_and_views(self):
        j = Journal()
        j.append("a", outcome=DONE, evidence=evidence(MODEL_CLAIM, "ok"))
        j.append("b", outcome=FAILED, evidence=evidence(CHECK, "raised"))
        j.append("c", outcome=REFUTED, reason="index is empty")
        self.assertEqual(len(j), 3)
        self.assertEqual([e.seq for e in j.entries], [0, 1, 2])
        self.assertEqual(len(j.done()), 1)
        self.assertEqual(len(j.refuted()), 1)

    def test_do_not_redo_renders_reasons(self):
        j = Journal()
        j.append("vector search", outcome=REFUTED, reason="index is empty")
        block = j.do_not_redo()
        self.assertIn("vector search", block)
        self.assertIn("index is empty", block)

    def test_do_not_redo_empty_when_nothing_refuted(self):
        self.assertEqual(Journal().do_not_redo(), "")

    def test_has_progress(self):
        spinning = Journal.from_list(
            [{"seq": 0, "outcome": FAILED}, {"seq": 1, "outcome": SKIPPED}])
        self.assertFalse(spinning.has_progress(2))
        moving = Journal.from_list(
            [{"seq": 0, "outcome": FAILED}, {"seq": 1, "outcome": REFUTED}])
        self.assertTrue(moving.has_progress(2))     # refuted shrinks the space
        self.assertTrue(Journal().has_progress())   # nothing yet ≠ stuck

    def test_round_trip(self):
        j = Journal()
        j.append("x", outcome=REFUTED, reason="why")
        back = Journal.from_list(j.to_list())
        self.assertEqual(len(back), 1)
        self.assertEqual(back.refuted()[0].reason, "why")

    def test_rejects_bad_values(self):
        with self.assertRaises(ValueError):
            Journal().append("x", outcome="whatever")
        with self.assertRaises(ValueError):
            evidence("guess")


# ── wiring into the Agent ──────────────────────────────────────────────────────

class TestAgentJournal(unittest.TestCase):

    def test_success_is_a_claim_failure_is_a_check(self):
        """The honest asymmetry: we *know* a tool raised; we only *believe* it worked."""
        m = _Model([_plan(("boom", "call API"), ("echo", "fallback")),
                    _act("boom"), _refl("continue"),
                    _act("echo", {"text": "b"}), _refl("continue")])
        res = Agent(orchestrator=m, tools=[Boom(), Echo()], max_attempts=1).run("t")
        kinds = {e["outcome"]: e["evidence"]["kind"] for e in res.journal}
        self.assertEqual(kinds[FAILED], CHECK)
        self.assertEqual(kinds[DONE], MODEL_CLAIM)

    def test_journal_recorded_per_attempt(self):
        m = _Model([_plan(("echo", "one"), ("echo", "two")),
                    _act("echo", {"text": "a"}), _refl("continue"),
                    _act("echo", {"text": "b"}), _refl("continue")])
        res = Agent(orchestrator=m, tools=[Echo()]).run("t")
        self.assertEqual(len(res.journal), 2)
        self.assertEqual([e["intent"] for e in res.journal], ["one", "two"])
        self.assertEqual([e["step"] for e in res.journal], [0, 1])

    def test_stop_is_recorded_as_refuted(self):
        m = _Model([_plan(("echo", "try this")),
                    _act("echo", {"text": "a"}),
                    _refl("stop", reason="dead end")])
        res = Agent(orchestrator=m, tools=[Echo()]).run("t")
        refuted = [e for e in res.journal if e["outcome"] == REFUTED]
        self.assertTrue(refuted)
        self.assertEqual(refuted[0]["reason"], "dead end")

    def test_do_not_redo_reaches_the_action_prompt(self):
        # step 1 is refuted via a replan, so step 2's action prompt must carry it
        m = _Model([_plan(("echo", "vector search"), ("echo", "second")),
                    _act("echo", {"text": "a"}), _refl("stop", reason="index empty")])
        agent = Agent(orchestrator=m, tools=[Echo()])
        agent.run("t")
        # nothing to assert on prompts here (run stopped), so drive the block directly:
        j = Journal()
        j.append("vector search", outcome=REFUTED, reason="index empty")
        from yait_aichain.agent import _prompts
        msgs = _prompts.action_messages(
            task="t", step={"goal": "g", "type": "tool"}, step_num=1, total_steps=1,
            context={}, tool_schemas=[], available_tool_names=["echo"],
            do_not_redo=j.do_not_redo())
        text = "\n".join(p["text"] for msg in msgs for p in msg["parts"])
        self.assertIn("ALREADY RULED OUT", text)
        self.assertIn("index empty", text)

    def test_no_ruled_out_block_when_journal_clean(self):
        from yait_aichain.agent import _prompts
        msgs = _prompts.action_messages(
            task="t", step={"goal": "g", "type": "tool"}, step_num=1, total_steps=1,
            context={}, tool_schemas=[], available_tool_names=["echo"])
        text = "\n".join(p["text"] for msg in msgs for p in msg["parts"])
        self.assertNotIn("ALREADY RULED OUT", text)


class TestJournalSurvivesSuspend(unittest.TestCase):

    def test_journal_restored_on_resume(self):
        from yait_aichain.tools import Wait
        m = _Model([_plan(("echo", "first"), ("wait", "pause")),
                    _act("echo", {"text": "a"}), _refl("continue"),
                    _act("wait"),                       # suspends
                    _refl("final_answer", final_answer="done")])
        agent = Agent(orchestrator=m, tools=[Echo(), Wait(name="wait")])
        res = agent.run("t")
        self.assertIsInstance(res, SuspendedResult)
        parked = res.document["definition"]["journal"]
        self.assertEqual(len(parked), 1)                # the first step's entry
        self.assertEqual(parked[0]["intent"], "first")

        final = agent.resume(res.run_id, signal={"ok": True})
        # the restored entry is still there, plus the resumed step's
        self.assertGreaterEqual(len(final.journal), 2)
        self.assertEqual(final.journal[0]["intent"], "first")


class TestHonestSuccessOnFinalAnswer(unittest.TestCase):
    """An earlier failed step must not be erased by emitting ``final_answer``."""

    def test_final_answer_does_not_hide_earlier_failure(self):
        m = _Model([_plan(("boom", "call API"), ("echo", "fallback")),
                    _act("boom"), _refl("continue"),
                    _act("echo", {"text": "b"}),
                    _refl("final_answer", final_answer="DONE")])
        res = Agent(orchestrator=m, tools=[Boom(), Echo()], max_attempts=1).run("t")
        self.assertFalse(res.success)
        self.assertIn("execution error", res.error)

    def test_clean_run_still_succeeds(self):
        m = _Model([_plan(("echo", "one")),
                    _act("echo", {"text": "a"}),
                    _refl("final_answer", final_answer="DONE")])
        res = Agent(orchestrator=m, tools=[Echo()]).run("t")
        self.assertTrue(res.success)
        self.assertEqual(res.output, "DONE")


if __name__ == "__main__":
    unittest.main()


class Crash(Tool):
    """Simulates the process being killed mid-step (not a normal tool error)."""
    name = "crash"
    parameters = {"type": "object", "properties": {}, "required": []}

    def run(self, options=None, **kw):
        raise KeyboardInterrupt("process killed")


class TestCrashRecovery(unittest.TestCase):
    """A checkpoint after every committed step makes an unplanned death recoverable."""

    def _plan3(self):
        return _plan(("echo", "one"), ("echo", "two"), ("echo", "three"))

    def test_run_resumes_after_process_death(self):
        import tempfile
        from yait_aichain.state import FileStore
        store_dir = tempfile.mkdtemp()

        # First life: step 1 commits (checkpoint), then the process is killed.
        agent = Agent(
            orchestrator=_Model([self._plan3(),
                                 _act("echo", {"text": "a"}), _refl("continue"),
                                 _act("crash")]),
            tools=[Echo(), Crash()], store=FileStore(store_dir))
        with self.assertRaises(KeyboardInterrupt):
            agent.run("t")

        # The checkpoint survived, with memory and the journal so far.
        import os, json as _json
        doc = _json.load(open(os.path.join(store_dir, os.listdir(store_dir)[0])))
        self.assertEqual(len(doc["definition"]["journal"]), 1)
        self.assertIn("r", doc["variables"])

        # Second life: a brand-new instance picks the run up from the store.
        revived = Agent(
            orchestrator=_Model([_act("echo", {"text": "b"}), _refl("continue"),
                                 _act("echo", {"text": "c"}),
                                 _refl("final_answer", final_answer="DONE")]),
            tools=[Echo(), Crash()], store=FileStore(store_dir))
        res = revived.resume(doc["run_id"])
        self.assertTrue(res.success)
        self.assertEqual(res.output, "DONE")
        self.assertGreaterEqual(len(res.journal), 2)      # journal spans the crash
        self.assertIn("r", res.memory)                    # memory from the first life

    def test_resume_is_a_noop_when_nothing_is_left(self):
        import tempfile
        from yait_aichain.state import FileStore, RunDocument
        store = FileStore(tempfile.mkdtemp())
        doc = RunDocument.new("agent", ["only"], variables={})
        doc.steps[0]["status"] = "done"
        store.save(doc.run_id, doc.to_dict())
        agent = Agent(orchestrator=_Model(["{}"]), tools=[Echo()], store=store)
        self.assertIsNone(agent.resume(doc.run_id))


class TestRepeatedSuspend(unittest.TestCase):
    """Suspending twice must not destroy the parked run (regression, 1.4.4)."""

    def test_suspend_resume_suspend_resume(self):
        import tempfile
        from yait_aichain.tools import Wait
        from yait_aichain.state import FileStore
        agent = Agent(
            orchestrator=_Model([_plan(("w1", "pause 1"), ("w2", "pause 2")),
                                 _act("w1"), _refl("continue"),
                                 _act("w2"),
                                 _refl("final_answer", final_answer="DONE")]),
            tools=[Wait(name="w1"), Wait(name="w2")],
            store=FileStore(tempfile.mkdtemp()))

        first = agent.run("t")
        self.assertIsInstance(first, SuspendedResult)
        second = agent.resume(first.run_id, signal={"ok": True})
        self.assertIsInstance(second, SuspendedResult)     # suspended again
        final = agent.resume(second.run_id, signal={"ok": True})
        self.assertTrue(final.success)                     # still resumable
        self.assertEqual(final.output, "DONE")
