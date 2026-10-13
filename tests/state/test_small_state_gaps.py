"""
Small gaps in state, stores and the journal, closed.

InMemoryStore took what FileStore refused (a set, bytes); `resume(run_id)`
with no signal took the parked question and lost it; the approval request did
not show the call; the fingerprint compared a 240-character excerpt; a
Journal was not iterable; FileStore('~/runs') made a folder named "~"; and a
document that could not be written left a temporary file behind.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agent"))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from yait_aichain.agent import Agent, acceptance_from                    # noqa: E402
from yait_aichain.agent._journal import Journal                           # noqa: E402
from yait_aichain.state import FileStore, InMemoryStore                   # noqa: E402
from yait_aichain.tools import Wait                                       # noqa: E402
from test_loop import _calls, _model, _text                               # noqa: E402


class TestStores(unittest.TestCase):

    def test_both_stores_refuse_what_json_cannot_hold(self):
        for store in (InMemoryStore(), FileStore(tempfile.mkdtemp())):
            with self.subTest(type(store).__name__), self.assertRaises(ValueError):
                store.save("r", {"tags": {"a", "b"}})

    def test_a_refused_document_leaves_no_file(self):
        folder = tempfile.mkdtemp()
        with self.assertRaises(ValueError):
            FileStore(folder).save("r", {"raw": b"x"})
        self.assertEqual(os.listdir(folder), [])

    def test_a_home_path_is_the_home(self):
        home = tempfile.mkdtemp()
        os.environ["HOME"] = home
        self.assertEqual(FileStore("~/runs")._dir, os.path.join(home, "runs"))


class TestResume(unittest.TestCase):

    def test_a_waiting_run_needs_its_answer_and_keeps_it(self):
        agent = Agent(_model(_calls(("c1", "wait", {})), _text("done")),
                      tools=[Wait(reason="approve?")], store=InMemoryStore(), verbose=0)
        parked = agent.run("refund")
        self.assertEqual((parked.awaiting["tool"], parked.awaiting["arguments"]), ("wait", {}))
        with self.assertRaises(TypeError):
            agent.resume(parked.run_id)
        self.assertIsNotNone(agent.store.load(parked.run_id))      # still there
        self.assertTrue(agent.resume(parked.run_id, signal={"ok": True}).success)


class TestJournal(unittest.TestCase):

    def test_a_change_past_the_excerpt_breaks_the_fingerprint(self):
        from yait_aichain.agent._journal import CHECK, evidence
        def journal(tail):
            j = Journal()
            j.append(action={"name": "test"}, evidence=evidence(CHECK, "ran"),
                     observation="x" * 300 + tail)
            return j
        self.assertNotEqual(acceptance_from(journal("0 failed")).fingerprint,
                            acceptance_from(journal("7 failed")).fingerprint)

    def test_a_journal_is_iterable(self):
        j = Journal()
        j.append(action={"name": "a"})
        self.assertEqual(len(list(j)), 1)


if __name__ == "__main__":
    unittest.main()
