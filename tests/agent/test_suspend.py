"""
The model may park the run — when there is somewhere to keep it.

A `Wait` or a `Gate` in an agent's tool list lets the model decide that the
world has to answer before the run can go on: approve this refund, confirm the
payment, come back at nine. Until 2026-09-18 that decision was swallowed —
`_execute` caught `Suspend` with every other exception and handed the model
back the string "Suspend: ..." as a tool result, so the pause never happened
and nobody outside the loop learned that anything was owed.

What is pinned here:

  * a suspend-capable tool with no store is refused **at construction**, since
    a pause nobody can resume is otherwise found at the worst moment;
  * `run()` returns a falsy `SuspendedResult` carrying the run id and what is
    awaited, and the document lands in the store;
  * the other calls of the same turn are answered as not executed, and the
    parked call is left unanswered — that gap is what the signal fills;
  * `resume()` delivers the signal as that call's result and the run finishes;
  * a resumed run is gone from the store, so an at-least-once trigger that
    delivers twice gets `KeyError` rather than a second tail;
  * parking twice keeps one run id.
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from yait_aichain.agent import Agent, step_count
from yait_aichain.models import Model
from yait_aichain.state import InMemoryStore, SuspendedResult
from yait_aichain.tools import Gate, Wait
from yait_aichain.tools._base import Tool


def _text(content):
    return json.dumps({
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }).encode()


def _calls(*specs):
    return json.dumps({
        "choices": [{"message": {
            "content": None,
            "tool_calls": [
                {"id": i, "type": "function",
                 "function": {"name": n, "arguments": json.dumps(a)}}
                for i, n, a in specs
            ],
        }}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }).encode()


def _model(*responses):
    """A model whose transport replays a script; the last reply repeats."""
    m = Model("gpt-4o", api_key="k")
    m.sent = []
    call = [0]

    def side(path, body, headers, *a, **k):
        m.sent.append(body)
        i = min(call[0], len(responses) - 1)
        call[0] += 1
        return responses[i]

    m.client._post = MagicMock(side_effect=side)
    m.client._auth_headers = MagicMock(return_value={})
    return m


class Note(Tool):
    name = "note"
    description = "Write a note."
    parameters = {"type": "object", "properties": {"text": {"type": "string"}},
                  "required": ["text"]}
    risk = "read"

    def __init__(self):
        self.written = []

    def run(self, text, options=None):
        self.written.append(text)
        return {"written": text}


def _agent(*responses, tools=None, store=None):
    return Agent(
        model     = _model(*responses),
        tools     = tools if tools is not None else [Wait(reason="Approve?",
                                                          resume_with={"approved": "bool"})],
        store     = store if store is not None else InMemoryStore(),
        stop_when = [step_count(6)],
    )


class TestRefusedWithoutAStore(unittest.TestCase):

    def test_a_suspend_tool_without_a_store_is_refused_at_construction(self):
        with self.assertRaises(ValueError) as caught:
            Agent(model=_model(_text("hi")), tools=[Wait(reason="Approve?")])
        message = str(caught.exception)
        self.assertIn("'wait'", message)
        self.assertIn("store=", message)

    def test_a_gate_is_refused_the_same_way(self):
        with self.assertRaises(ValueError):
            Agent(model=_model(_text("hi")), tools=[Gate(Note())])

    def test_an_ordinary_tool_needs_no_store(self):
        agent = Agent(model=_model(_text("hi")), tools=[Note()])
        self.assertIsNone(agent.store)


class TestParking(unittest.TestCase):

    def test_the_run_parks_and_says_what_it_waits_for(self):
        agent  = _agent(_calls(("c1", "wait", {})), _text("done"))
        result = agent.run("send the refund")

        self.assertIsInstance(result, SuspendedResult)
        self.assertFalse(result)                       # falsy, like a Chain's
        self.assertEqual(result.awaiting["reason"], "Approve?")
        self.assertEqual(result.awaiting["resume_with"], {"approved": "bool"})
        self.assertIsNotNone(agent.store.load(result.run_id))

    def test_the_pause_is_announced(self):
        seen  = []
        agent = _agent(_calls(("c1", "wait", {})), _text("done"))
        agent.hooks.append(lambda e: seen.append(e))
        result = agent.run("send the refund")

        parked = [e for e in seen if e.type == "run.suspended"]
        self.assertEqual(len(parked), 1)
        self.assertEqual(parked[0].payload["run_id"], result.run_id)
        self.assertEqual(parked[0].payload["tool"], "wait")
        self.assertEqual([e.type for e in seen].count("run.finished"), 0)

    def test_other_calls_of_the_same_turn_are_answered(self):
        agent = _agent(_calls(("c1", "wait", {}), ("c2", "note", {"text": "x"})),
                       _text("done"),
                       tools=[Wait(reason="Approve?"), Note()])
        result = agent.run("do both")

        self.assertIsInstance(result, SuspendedResult)
        messages = result.document["variables"]["messages"]
        answered = json.dumps(messages)
        self.assertIn("not executed", answered)
        # The parked call is the one thing left owed.
        self.assertEqual(answered.count("c1"), 1)

    def test_nothing_after_the_pause_ran(self):
        note  = Note()
        agent = _agent(_calls(("c1", "wait", {}), ("c2", "note", {"text": "x"})),
                       _text("done"),
                       tools=[Wait(reason="Approve?"), note])
        agent.run("do both")
        self.assertEqual(note.written, [])


class TestResuming(unittest.TestCase):

    def test_the_signal_becomes_the_answer_and_the_run_finishes(self):
        agent  = _agent(_calls(("c1", "wait", {})), _text("refund sent"))
        parked = agent.run("send the refund")

        result = agent.run and agent.resume(parked.run_id, signal={"approved": True})
        self.assertTrue(result.success)
        self.assertEqual(result.output, "refund sent")
        self.assertIn("approved", json.dumps(agent.model.sent[-1]))

    def test_a_resumed_run_leaves_the_store(self):
        agent  = _agent(_calls(("c1", "wait", {})), _text("refund sent"))
        parked = agent.run("send the refund")
        agent.resume(parked.run_id, signal={"approved": True})

        self.assertIsNone(agent.store.load(parked.run_id))
        with self.assertRaises(KeyError):
            agent.resume(parked.run_id, signal={"approved": True})

    def test_an_unknown_run_id_is_a_key_error(self):
        agent = _agent(_text("hi"))
        with self.assertRaises(KeyError):
            agent.resume("run-nothing", signal={})

    def test_the_journal_and_the_counters_survive_the_pause(self):
        agent  = _agent(_calls(("c1", "note", {"text": "first"})),
                        _calls(("c2", "wait", {})),
                        _text("done"),
                        tools=[Wait(reason="Approve?"), Note()])
        parked = agent.run("note then ask")
        result = agent.resume(parked.run_id, signal={"approved": True})

        self.assertTrue(result.success)
        # Two acting turns — the note and the wait; the answering turn takes
        # no step. Without the counters travelling in the document this
        # would restart at zero and read as one.
        self.assertEqual(result.steps_taken, 2)
        actions = json.dumps(result.journal)
        self.assertIn("note", actions)
        self.assertIn("resumed with an external signal", actions)

    def test_parking_twice_keeps_one_run_id(self):
        agent  = _agent(_calls(("c1", "wait", {})),
                        _calls(("c2", "wait", {})),
                        _text("done"))
        first  = agent.run("ask twice")
        second = agent.resume(first.run_id, signal={"approved": True})

        self.assertIsInstance(second, SuspendedResult)
        self.assertEqual(second.run_id, first.run_id)
        self.assertTrue(agent.resume(second.run_id, signal={"approved": True}).success)


class TestGateInAnAgent(unittest.TestCase):

    def test_an_approved_gate_runs_the_wrapped_tool(self):
        note  = Note()
        agent = _agent(_calls(("c1", "gate_note", {"text": "send it"})),
                       _text("sent"),
                       tools=[Gate(note, reason="Send the note?")])
        parked = agent.run("send the note")
        self.assertEqual(note.written, [])

        result = agent.resume(parked.run_id, signal={"approved": True})
        self.assertEqual(note.written, ["send it"])
        self.assertTrue(result.success)

    def test_a_refused_gate_skips_it(self):
        note  = Note()
        agent = _agent(_calls(("c1", "gate_note", {"text": "send it"})),
                       _text("not sent"),
                       tools=[Gate(note, reason="Send the note?")])
        parked = agent.run("send the note")
        result = agent.resume(parked.run_id, signal={"approved": False})

        self.assertEqual(note.written, [])
        self.assertTrue(result.success)


if __name__ == "__main__":
    unittest.main()
