"""
The lead reads what each worker is for.

`Agent(description=...)` is documented as "what a lead agent reads when
choosing a worker", and llms.txt teaches it. It was stored and never read:
the `delegate` schema carried the names alone, so a lead picking between
`worker0` and `worker1` — or `researcher` and `analyst` — was guessing, and a
wrong guess costs a whole worker run. Found 2026-10-14.

Pinned on what the model actually receives: the `delegate` tool schema the
lead sends. And a worker the lead cannot tell apart is said at construction.
"""

import json
import os
import sys
import unittest
import warnings
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from yait_aichain.agent import Agent
from yait_aichain.models import Model


def _model():
    m = Model("gpt-4o", api_key="k")
    m.sent = []

    def side(path, body, headers, *a, **k):
        m.sent.append(body)
        return json.dumps({"choices": [{"message": {"content": "done"}}],
                           "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                                     "total_tokens": 2}}).encode()

    m.client._post = MagicMock(side_effect=side)
    m.client._auth_headers = MagicMock(return_value={})
    return m


def _delegate_sent(lead):
    lead.run("go")
    tools = lead.model.sent[-1]["tools"]
    return next(t for t in tools if t["function"]["name"] == "delegate")["function"]


def _quiet(make):
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        value = make()
    return value, [str(w.message) for w in seen if "no name or no description" in str(w.message)]


class TestTheDescriptionsReachTheModel(unittest.TestCase):

    def test_each_worker_is_described_beside_its_name(self):
        chemist = Agent(_model(), name="chemist", description="Answers chemistry questions.")
        scout = Agent(_model(), name="scout", description="Searches the web.")
        lead, _ = _quiet(lambda: Agent(_model(), team=[chemist, scout]))
        worker = _delegate_sent(lead)["parameters"]["properties"]["worker"]
        self.assertEqual(worker["enum"], ["chemist", "scout"])
        self.assertIn("chemist: Answers chemistry questions.", worker["description"])
        self.assertIn("scout: Searches the web.", worker["description"])

    def test_a_missing_description_is_said_in_the_schema_too(self):
        lead, _ = _quiet(lambda: Agent(_model(), team=[Agent(_model(), name="anon")]))
        worker = _delegate_sent(lead)["parameters"]["properties"]["worker"]
        self.assertIn("anon: no description given", worker["description"])


class TestABlindWorkerIsSaid(unittest.TestCase):

    def test_no_name_or_no_description_warns_once_at_construction(self):
        named = Agent(_model(), name="chemist", description="Answers chemistry questions.")
        nameless = Agent(_model(), description="Searches the web.")
        undescribed = Agent(_model(), name="analyst")
        _, said = _quiet(lambda: Agent(_model(), team=[named, nameless, undescribed]))
        self.assertEqual(len(said), 1)
        self.assertIn("worker1", said[0])
        self.assertIn("analyst", said[0])
        self.assertNotIn("chemist", said[0])

    def test_a_described_team_is_quiet(self):
        team = [Agent(_model(), name="chemist", description="Answers chemistry questions.")]
        self.assertEqual(_quiet(lambda: Agent(_model(), team=team))[1], [])

    def test_an_auto_team_has_nobody_to_describe(self):
        self.assertEqual(_quiet(lambda: Agent(_model(), team="auto"))[1], [])


if __name__ == "__main__":
    unittest.main()
