"""
A tool call with no result never reaches a provider.

`run()` cannot produce one — it appends a result turn for every call in a
reply, failed ones included. The externally driven seam could: `step()`
returns a decision without executing it and its docstring hands the obligation
to the caller — *"the caller appends the reply and whatever the world
answered, then calls again"* — with nothing verifying that they did. That seam
is what a benchmark harness and a serverless driver use, which is exactly
where a result crosses a process boundary and can be lost.

Why it matters more than it looks. On a dangling call the providers disagree
in the worst possible way: OpenAI chat completions, the Responses API and
Anthropic reject the request; Google keys results by name and is looser; a
self-hosted OpenAI-compatible server validates *nothing* and templates
whatever it was handed, so the model meets its own unanswered call and
improvises — repeats it, apologises, or invents the result, differently each
time. Provider interchangeability is this library's main promise and that is a
place it did not hold.

And the condition producing a dangling call is itself intermittent, so it
turned into unbounded behaviour on some trials and not others: run-to-run
variance manufactured by us. What the check converts is not a wrong answer
into a right one — it is quiet weirdness into the same named failure every
time. An instrument for reliability, not for accuracy.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "agent"))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from yait_aichain import Agent                                      # noqa: E402
from yait_aichain.models._calls import (dangling_calls,             # noqa: E402
                                        tool_result_turn)
from test_loop import Echo, _calls, _model, _text                   # noqa: E402

USER = {"role": "user", "parts": [{"type": "text", "text": "hi"}]}


def _asked(*ids):
    return {"role": "assistant",
            "tool_calls": [{"id": i, "name": "echo", "arguments": {}}
                           for i in ids]}


class TestWhatCountsAsDangling(unittest.TestCase):

    def test_an_unanswered_call(self):
        self.assertEqual(dangling_calls([USER, _asked("c1")]), ["c1"])

    def test_an_answered_one_is_not(self):
        self.assertEqual(
            dangling_calls([USER, _asked("c1"), tool_result_turn("c1", "ok")]),
            [])

    def test_several_calls_in_one_turn_are_tracked_apart(self):
        """A provider may ask for several and the agent honours all of them,
        so answering one is not answering the turn."""
        history = [USER, _asked("c1", "c2"), tool_result_turn("c1", "ok")]
        self.assertEqual(dangling_calls(history), ["c2"])

    def test_a_failed_call_counts_as_answered(self):
        """The error *is* the result. A driver that skips a failed call
        because it has nothing good to report creates the exact history this
        check exists to reject."""
        history = [USER, _asked("c1"),
                   tool_result_turn("c1", "PermissionError: denied")]
        self.assertEqual(dangling_calls(history), [])

    def test_a_conversation_with_no_tools_is_clean(self):
        self.assertEqual(dangling_calls([USER, {"role": "assistant",
                                                "parts": [{"type": "text",
                                                           "text": "hi"}]}]),
                         [])


class TestTheSeamRefusesToSendOne(unittest.TestCase):

    def _agent(self):
        return Agent(_model(_text("done")), tools=[Echo()], verbose=0,
                     name="driver")

    def test_step_raises_rather_than_asking_the_provider(self):
        with self.assertRaises(ValueError) as caught:
            self._agent().step([USER, _asked("c1")])
        self.assertIn("c1", str(caught.exception))

    def test_the_message_says_what_to_append(self):
        """It fails locally, before the network, and points at the caller —
        so it has to name the fix rather than the symptom."""
        with self.assertRaises(ValueError) as caught:
            self._agent().step([USER, _asked("c1")])
        text = str(caught.exception)
        self.assertIn("tool_result_turn", text)
        self.assertIn("failed ones included", text)

    def test_nothing_was_sent(self):
        """Local failure is a third of the point: no tokens, no latency."""
        agent = self._agent()
        with self.assertRaises(ValueError):
            agent.step([USER, _asked("c1")])
        self.assertEqual(agent.model.sent, [])

    def test_a_well_formed_history_goes_through(self):
        agent = self._agent()
        reply = agent.step([USER, _asked("c1"), tool_result_turn("c1", "ok")])
        self.assertEqual(reply, "done")

    def test_the_same_input_always_fails_the_same_way(self):
        """The property the plan asks for: an intermittent defect becomes a
        deterministic, named failure."""
        seen = set()
        for _ in range(3):
            try:
                self._agent().step([USER, _asked("c1", "c2")])
            except ValueError as exc:
                seen.add(str(exc))
        self.assertEqual(len(seen), 1)


class TestTheLoopWasAlreadyHoldingIt(unittest.TestCase):
    """`run()` appends a result for every call including failures, so the
    invariant is not a new obligation on it — which is why turning the check
    on cost the existing suite nothing."""

    def test_a_tool_that_raises_still_leaves_a_clean_history(self):
        class Broken(Echo):
            def run(self, value, options=None):
                raise RuntimeError("upstream down")

        agent = Agent(_model(_calls(("c1", "echo", {"value": "x"})),
                             _text("done")),
                      tools=[Broken()], verbose=0)
        result = agent.run("go")
        self.assertTrue(result.success)
        self.assertTrue(any("upstream down" in e.get("reason", "")
                            for e in result.journal))


if __name__ == "__main__":
    unittest.main()
