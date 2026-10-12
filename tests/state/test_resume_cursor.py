"""
Resume-cursor correctness (1.3.3): a resumed run must not re-run steps that
were already attempted (done / skipped / failed), and a terminal error during a
resumed run must clear the parked document so a duplicate trigger is a no-op.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from yait_aichain.chain import Chain
from yait_aichain.tools import Wait
from yait_aichain.tools._base import Tool
from yait_aichain.state import InMemoryStore, SuspendedResult
from yait_aichain.state._run_document import RunDocument, StepStatus


class Counter(Tool):
    parameters = {"type": "object", "properties": {}, "required": []}
    def __init__(self, name):
        self.name = name
        self.calls = 0
    def run(self, **kw):
        self.calls += 1
        return f"{self.name}:{self.calls}"


class Boom(Tool):
    name = "boom"
    parameters = {"type": "object", "properties": {}, "required": []}
    def run(self, **kw):
        raise RuntimeError("boom")


class TestFirstPendingSkipsTerminal(unittest.TestCase):

    def test_cursor_skips_done_skipped_failed(self):
        doc = RunDocument.new("chain", ["a", "b", "c", "d"])
        doc.steps[0]["status"] = StepStatus.DONE
        doc.steps[1]["status"] = StepStatus.SKIPPED
        doc.steps[2]["status"] = StepStatus.SUSPENDED
        # step 3 stays pending
        self.assertEqual(doc.first_pending(), 2)   # the suspended step, not the skipped one


class TestResumeDoesNotRerunSkipped(unittest.TestCase):

    def test_skipped_and_done_steps_not_rerun_on_resume(self):
        b = Counter("b")
        c = Counter("c")
        store = InMemoryStore()
        chain = Chain(
            steps=[(Boom(), "a"),
                   (b, "b"),
                   Wait(reason="hold"),
                   (c, "c")],
            store=store,
            on_step_error="skip",
        )
        res = chain.run(variables={})
        self.assertIsInstance(res, SuspendedResult)
        self.assertEqual(b.calls, 1)               # b ran once before the pause

        chain.resume(res.run_id, signal={"ok": True})
        self.assertEqual(b.calls, 1)               # b NOT re-run on resume (bug guard)
        self.assertEqual(c.calls, 1)               # c ran once after resume


class TestErrorDuringResumeKeepsTheRun(unittest.TestCase):
    """A step that fails after a resume leaves the run in the store with that
    step marked failed: the store gave it up to this resume, and dropping it
    lost a run a person had approved to a transient error. Resuming again
    retries the failed step only — the approved one is never re-run."""

    def test_a_failure_after_resume_is_kept_and_retried_from_the_failed_step(self):
        from yait_aichain.state import StepStatus
        store, waits = InMemoryStore(), []

        class Counted(Wait):
            def run(self, _signal=None, **kw):
                if _signal is not None:
                    waits.append(_signal)
                return super().run(_signal=_signal, **kw)

        chain = Chain(steps=[Counted(reason="hold"), (Boom(), "x")], store=store,
                      on_step_error="raise")
        res = chain.run(variables={})
        self.assertIsInstance(res, SuspendedResult)

        with self.assertRaises(RuntimeError):       # Boom fails after resume
            chain.resume(res.run_id, signal={"ok": True})
        kept = store.load(res.run_id)
        self.assertEqual([s["status"] for s in kept["steps"]], [StepStatus.DONE, StepStatus.FAILED])

        with self.assertRaises(RuntimeError):       # the retry runs Boom again, not the Wait
            chain.resume(res.run_id)
        self.assertEqual(waits, [{"ok": True}])

    def test_two_deliveries_of_one_signal_run_it_once(self):
        store, done = InMemoryStore(), []

        class Once(Wait):
            def run(self, _signal=None, **kw):
                if _signal is not None:
                    done.append(_signal)
                return super().run(_signal=_signal, **kw)

        chain = Chain(steps=[Once(reason="approve")], store=store)
        res = chain.run(variables={})
        chain.resume(res.run_id, signal={"ok": True})
        with self.assertRaises(KeyError):
            chain.resume(res.run_id, signal={"ok": True})
        self.assertEqual(len(done), 1)


if __name__ == "__main__":
    unittest.main()
