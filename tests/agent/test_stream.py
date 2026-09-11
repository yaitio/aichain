"""
`Agent.stream()` — the same loop, walked instead of exhausted.

The thing being defended here is not the streaming; it is that there is still
**one** loop. A second copy for the streaming case is the mistake 2.0 undid
when `Agent` stopped hand-rolling its own call path beside `Skill`: two paths
do not stay the same, and the divergence shows up first in whichever one has
fewer tests. So every assertion below has a twin — what `run()` produced must
be what `stream()` produced, for the same script.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from agent import Agent, step_count                                # noqa: E402
from test_loop import Echo, _calls, _model, _text                  # noqa: E402


SCRIPT = (_calls(("c1", "echo", {"value": "one"})),
          _text("done"))


def _agent(*responses, **kw):
    return Agent(model=_model(*responses), tools=[Echo()], verbose=0, **kw)


class TestTheStreamIsTheRunSeenFromInside(unittest.TestCase):

    def test_the_result_matches_what_run_would_have_returned(self):
        streamed = _agent(*SCRIPT)
        list(streamed.stream("do it"))
        ran = _agent(*SCRIPT).run("do it")

        self.assertEqual(streamed.last_result.success, ran.success)
        self.assertEqual(streamed.last_result.output, ran.output)
        self.assertEqual(streamed.last_result.stopped_by, ran.stopped_by)
        self.assertEqual(streamed.last_result.steps_taken, ran.steps_taken)

    def test_the_journal_is_attached_the_same_way(self):
        agent = _agent(*SCRIPT)
        list(agent.stream("do it"))
        entry = agent.last_result.journal[0]
        self.assertTrue(agent.last_result.journal)
        # `action` is stored as text — the journal is a written record, so
        # what goes in it is a rendering, not the live object.
        self.assertIn("echo", str(entry["action"]))

    def test_the_tools_actually_ran(self):
        sink = []
        agent = Agent(model=_model(*SCRIPT), tools=[Echo(sink)], verbose=0)
        list(agent.stream("do it"))
        self.assertEqual(sink, ["one"])


class TestWhatTheStreamCarries(unittest.TestCase):

    def _types(self, *responses, task="do it", **kw):
        agent = _agent(*responses, **kw)
        return [e.type for e in agent.stream(task)], agent

    def test_it_opens_and_closes_the_run(self):
        types, _ = self._types(*SCRIPT)
        self.assertEqual(types[0], "run.started")
        self.assertEqual(types[-1], "run.finished")

    def test_a_tool_call_is_visible_while_the_run_is_still_going(self):
        """The point of the feature: a caller sees the step, not a summary
        after the fact."""
        types, _ = self._types(*SCRIPT)
        self.assertIn("tool_call.started", types)
        self.assertIn("tool_call.ended", types)
        self.assertLess(types.index("tool_call.started"),
                        types.index("run.finished"))

    def test_the_model_calls_come_through_too(self):
        """These are emitted by `Skill`, not by the loop — they reach the
        stream only because it is a view of the hook channel rather than a
        vocabulary of its own."""
        types, _ = self._types(*SCRIPT)
        self.assertIn("llm_call.started", types)
        self.assertIn("llm_call.ended", types)

    def test_the_tool_name_rides_along(self):
        agent = _agent(*SCRIPT)
        steps = [e for e in agent.stream("do it")
                 if e.type == "tool_call.started"]
        self.assertEqual([e.name for e in steps], ["echo"])

    def test_a_stop_condition_still_stops_it(self):
        types, agent = self._types(
            _calls(("c1", "echo", {"value": "again"})),
            stop_when=[step_count(1)])
        self.assertEqual(types[-1], "run.finished")
        self.assertFalse(agent.last_result.success)
        self.assertEqual(agent.last_result.stopped_by, "step_count")


class TestTheHookChannelIsLeftAsItWasFound(unittest.TestCase):
    """A collector added for the duration and not removed is a leak that
    only appears under load — the list goes on filling and nobody drains it."""

    def test_a_caller_s_own_hooks_still_fire(self):
        seen = []
        agent = Agent(model=_model(*SCRIPT), tools=[Echo()], verbose=0,
                      hooks=[seen.append])
        list(agent.stream("do it"))
        self.assertIn("run.finished", [e.type for e in seen])

    def test_nothing_is_left_behind_afterwards(self):
        seen = []
        agent = Agent(model=_model(*SCRIPT), tools=[Echo()], verbose=0,
                      hooks=[seen.append])
        list(agent.stream("do it"))
        self.assertEqual(agent.hooks, [seen.append])

    def test_not_even_when_the_caller_walks_away_early(self):
        """Abandoning the generator abandons the run — but the hook list is
        still restored, because that part is a `finally`."""
        agent = _agent(*SCRIPT)
        stream = agent.stream("do it")
        next(stream)
        stream.close()
        self.assertEqual(agent.hooks, [])
        self.assertIsNone(agent.last_result)


if __name__ == "__main__":
    unittest.main()
