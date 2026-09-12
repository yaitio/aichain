"""
`nudge` — a stop condition that speaks instead of ending the run.

`stop_when` knew two kinds and both ended the run. A ceiling says "stop",
where a stalled run needs "change approach"; so a stuck agent either spun
until a ceiling caught it, or was cut short by a ceiling set low enough to
catch the spin — and low enough to cut off a run that was genuinely working.

The predicates for "stuck" were already written and **never called from
library code**: `Journal.has_progress` and `Journal.is_repeating`, documented
and available only to someone writing their own loop. This is the first stage
of *iterating toward a target*, and it is ordered first because it needs no
scorer — stall and repetition are read off the journal, not off the task, so
it applies to every task and has nothing to overfit to.
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from yait_aichain import Agent                                      # noqa: E402
from yait_aichain.agent import (nudge, repeating, stalled,          # noqa: E402
                                step_count)
from yait_aichain.tools._base import Tool                           # noqa: E402
from test_loop import Echo, _calls, _model, _text                   # noqa: E402


class Fussy(Tool):
    """Fails unless handed "ok" — so a script can choose, call by call,
    whether an attempt moved anything."""
    name = "fussy"
    parameters = {"type": "object",
                  "properties": {"value": {"type": "string"}},
                  "required": ["value"]}

    def run(self, value, options=None):
        if value != "ok":
            raise ValueError("not like that")
        return "accepted"


def _script(*values, answer="done"):
    return [_calls((f"c{i}", "fussy", {"value": v}))
            for i, v in enumerate(values)] + [_text(answer)]


def _run(responses, stop_when, tools=None):
    events = []
    agent = Agent(_model(*responses), tools=tools or [Fussy()],
                  stop_when=stop_when, verbose=0, hooks=[events.append])
    return agent, agent.run("go"), events


def _fired(events):
    return [e for e in events if e.type == "nudge.fired"]


class TestItSpeaksAndTheRunContinues(unittest.TestCase):

    def test_a_stall_produces_one_message_and_no_stop(self):
        agent, result, events = _run(_script("bad", "bad", "bad"),
                                     [stalled(3), step_count(20)])
        self.assertEqual(len(_fired(events)), 1)
        self.assertTrue(result.success)
        self.assertEqual(result.stopped_by, "answered")

    def test_the_model_is_actually_told(self):
        """An ordinary user turn in the conversation, not an invisible edit
        to the system prompt — so the transcript shows what it was told."""
        agent, _, _ = _run(_script("bad", "bad", "bad"),
                           [stalled(3), step_count(20)])
        later = json.dumps(agent.model.sent[-1])
        self.assertIn("did not move the task forward", later)

    def test_it_names_what_was_already_ruled_out_when_there_is_any(self):
        agent, _, events = _run(_script("bad", "bad", "bad"),
                                [stalled(3), step_count(20)])
        self.assertIn("did not move", _fired(events)[0].payload["message"])


class TestOncePerStreak(unittest.TestCase):
    """A reminder repeated every turn is how a reminder stops being read."""

    def test_it_stays_quiet_while_the_stall_continues(self):
        _, _, events = _run(_script("bad", "bad", "bad", "bad", "bad", "bad"),
                            [stalled(3), step_count(20)])
        self.assertEqual(len(_fired(events)), 1)

    def test_it_re_arms_once_something_moves(self):
        _, _, events = _run(
            _script("bad", "bad", "bad", "ok", "bad", "bad", "bad"),
            [stalled(3), step_count(20)])
        self.assertEqual(len(_fired(events)), 2)


class TestRepeating(unittest.TestCase):
    """The other half: a run succeeding pointlessly. Every result is a clean
    `done`, so nothing in the outcomes tells it from real work."""

    def test_identical_successful_actions_are_noticed(self):
        responses = [_calls((f"c{i}", "echo", {"value": "same"}))
                     for i in range(4)] + [_text("done")]
        _, result, events = _run(responses, [repeating(3), step_count(20)],
                                 tools=[Echo()])
        self.assertEqual([e.payload["name"] for e in _fired(events)],
                         ["repeating"])
        self.assertTrue(result.success)

    def test_varied_successful_actions_are_not(self):
        responses = [_calls((f"c{i}", "echo", {"value": f"v{i}"}))
                     for i in range(4)] + [_text("done")]
        _, _, events = _run(responses, [repeating(3), step_count(20)],
                            tools=[Echo()])
        self.assertEqual(_fired(events), [])


class TestItCannotTalkTheLoopPastItsBudget(unittest.TestCase):

    def test_a_ceiling_still_ends_the_run_even_listed_after_a_nudge(self):
        """Terminal conditions are evaluated first. A nudge listed first in
        `stop_when` must not be a way to run past a ceiling."""
        _, result, _ = _run(_script("bad", "bad", "bad", "bad"),
                            [stalled(2), step_count(2)])
        self.assertFalse(result.success)
        self.assertEqual(result.stopped_by, "step_count")


class TestTheGeneralForm(unittest.TestCase):

    def test_any_predicate_over_state(self):
        _, _, events = _run(
            _script("ok", "ok", "ok"),
            [nudge(lambda s: s["steps"] >= 2, "Halfway: summarise so far.",
                   name="halfway"),
             step_count(20)])
        fired = _fired(events)
        self.assertEqual([e.payload["name"] for e in fired], ["halfway"])
        self.assertEqual(fired[0].payload["message"],
                         "Halfway: summarise so far.")

    def test_it_is_not_mistaken_for_a_terminal_condition(self):
        """`_fired` returns the first terminal condition. A nudge whose
        predicate is true must not be returned from it and end the run."""
        _, result, _ = _run(_script("ok", "ok"),
                            [nudge(lambda s: True, "hello", name="always"),
                             step_count(20)])
        self.assertTrue(result.success)


if __name__ == "__main__":
    unittest.main()


class TestTheJournalCanBeReadBack(unittest.TestCase):
    """Two defects found while wiring `nudge` to the journal, both latent
    because nothing in the library called these views before."""

    def test_the_loop_records_no_intent(self):
        """It passed the tool's name, so every call to one tool shared an
        "intent" and `is_repeating` read ten different searches as one
        search repeated ten times."""
        _, result, _ = _run(_script("ok", "bad"), [step_count(20)])
        self.assertTrue(all(e["intent"] == "" for e in result.journal
                            if e["action"]))

    def test_progress_summary_reads_an_agent_journal(self):
        """The loop stores the action as text; the renderer called `.get`
        on it and raised."""
        from yait_aichain.agent import Journal
        _, result, _ = _run(_script("ok", "bad"), [step_count(20)])
        summary = Journal.from_list(result.journal).progress_summary()
        self.assertIn("fussy", summary)
        self.assertNotIn("(no intent)", summary)
