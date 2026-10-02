"""
decideTypeSafe — TypeSafe's Jev as a tool.

Jev writes no text: it answers typed questions about a state with
probabilities. Tried first as a `Model` on 2026-10-02 and moved here the same
day — as a model it bent ten places outside its own client and an `Agent`
could not use it at all; as a tool it touches nothing else and an agent calls
it by name.

What is pinned, offline:

  * the questions are fixed at construction and checked there, each malformed
    one named by its id — before any network;
  * the model is told what it will learn: the description is written from the
    questions, and the only argument is the text, ``state``;
  * the request is the documented one, and a structured state from code
    travels as structure;
  * answers come back untouched, beside the version that answered and the
    cost of input — output is free;
  * an API error is raised with its status by ``run()`` and returned, not
    raised, by a call;
  * an agent calls it and reads the probabilities.

The response fixture is a real one, captured from jev-1.13.0 on 2026-10-02.
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from yait_aichain.tools import decideTypeSafe

QUESTIONS = {
    "urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
    "team": {"type": "choice", "instructions": "Which team should handle it?",
             "criteria": {"billing": "Payments, invoicing, refunds",
                          "technical": "Bugs, outages, integrations",
                          "sales": "Pricing, upgrades"}},
    "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                    "criteria": ["Calm", "Frustrated", "Very angry"]},
}
TICKET = "Help! My payouts have been failing for 3 days."

#: Captured live from jev-1.13.0, 2026-10-02.
LIVE = {
    "model": "jev-1.13.0",
    "answers": {
        "urgent": {"type": "noul", "noul": 0.95},
        "team": {"type": "choice", "choice": "billing", "confidence": 0.83,
                 "probabilities": {"technical": 0.11, "billing": 0.89, "sales": 0.0}},
        "frustration": {"type": "score", "score": 1.04, "confidence": 0.93,
                        "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
                        "probabilities": {"0": 0.0, "1": 0.96, "2": 0.04}},
    },
    "usage": {"input_tokens": 399, "output_tokens": 73},
}


def _tool(response=LIVE, status=200, **kw):
    tool = decideTypeSafe(QUESTIONS, api_key="k", **kw)
    reply = MagicMock()
    reply.status = status
    reply.data = (json.dumps(response) if isinstance(response, dict) else response).encode()
    tool._http = MagicMock()
    tool._http.request.return_value = reply
    return tool


class TestConstruction(unittest.TestCase):

    def test_a_malformed_question_is_named_before_any_network(self):
        cases = {
            "no questions": ({}, "at least one question"),
            "unknown type": ({"q": {"type": "rank", "instructions": "?"}}, "'q'"),
            "no instructions": ({"q": {"type": "noul"}}, "'q'"),
            "choice without criteria": ({"team": {"type": "choice", "instructions": "?"}}, "'team'"),
            "score with one level": ({"s": {"type": "score", "instructions": "?",
                                            "criteria": ["only"]}}, "'s'"),
            "score with eleven levels": ({"s": {"type": "score", "instructions": "?",
                                                "criteria": [str(i) for i in range(11)]}}, "'s'"),
            "too many options": ({"c": {"type": "choice", "instructions": "?",
                                        "criteria": {str(i): None for i in range(256)}}}, "'c'"),
        }
        for label, (questions, named) in cases.items():
            with self.subTest(case=label):
                with self.assertRaises(ValueError) as ctx:
                    decideTypeSafe(questions, api_key="k")
                self.assertIn(named, str(ctx.exception))

    def test_no_key_is_refused_at_construction(self):
        saved = os.environ.pop("TYPESAFE_API_KEY", None)
        try:
            with self.assertRaises(ValueError) as ctx:
                decideTypeSafe(QUESTIONS)
            self.assertIn("TYPESAFE_API_KEY", str(ctx.exception))
        finally:
            if saved is not None:
                os.environ["TYPESAFE_API_KEY"] = saved

    def test_an_overloaded_answer_is_retried(self):
        # 529 is refused before inference, like the 429 and 503 already retried.
        tool = decideTypeSafe(QUESTIONS, api_key="k")
        self.assertIn(529, tool._http.connection_pool_kw["retries"].status_forcelist)


class TestWhatTheModelSees(unittest.TestCase):

    def test_the_only_argument_is_the_text(self):
        schema = decideTypeSafe(QUESTIONS, api_key="k").parameters
        self.assertEqual(list(schema["properties"]), ["state"])
        self.assertEqual(schema["required"], ["state"])

    def test_the_description_says_what_each_question_answers(self):
        d = decideTypeSafe(QUESTIONS, api_key="k").description
        self.assertIn("urgent: Does this convey urgency? → probability", d)
        self.assertIn("one of billing, technical, sales", d)
        self.assertIn("Calm < Frustrated < Very angry", d)

    def test_it_is_named_by_what_it_decides(self):
        tool = decideTypeSafe(QUESTIONS, name="triage", api_key="k")
        self.assertEqual(tool.name, "triage")
        self.assertEqual(tool.schema()["function"]["name"], "triage")

    def test_it_reads_and_changes_nothing(self):
        self.assertEqual(decideTypeSafe(QUESTIONS, api_key="k").risk, "read")


class TestTheRequest(unittest.TestCase):

    def test_the_documented_request_is_sent(self):
        tool = _tool(model="jev-1.13.0")
        tool.run(state=TICKET)
        method, url = tool._http.request.call_args.args
        kw = tool._http.request.call_args.kwargs
        self.assertEqual((method, url), ("POST", "https://api.typesafe.ai/v1/systemone"))
        self.assertEqual(kw["headers"]["Authorization"], "Bearer k")
        self.assertEqual(json.loads(kw["body"]),
                         {"state": TICKET, "model": "jev-1.13.0", "questions": QUESTIONS})

    def test_a_structured_state_travels_as_structure(self):
        tool = _tool()
        record = {"name": "John Smith", "last_employer": "Google"}
        tool.run(state=record)
        self.assertEqual(json.loads(tool._http.request.call_args.kwargs["body"])["state"], record)

    def test_an_empty_state_is_refused(self):
        with self.assertRaises(ValueError):
            _tool().run(state="   ")


class TestTheAnswer(unittest.TestCase):

    def test_answers_come_back_untouched(self):
        out = _tool().run(state=TICKET)
        self.assertEqual(out["answers"], LIVE["answers"])
        # Not thresholded into a bool: the probability is the product.
        self.assertEqual(out["answers"]["urgent"]["noul"], 0.95)

    def test_the_version_that_answered_is_reported(self):
        self.assertEqual(_tool().run(state=TICKET)["model"], "jev-1.13.0")

    def test_only_input_is_billed(self):
        out = _tool().run(state=TICKET)
        self.assertEqual(out["usage"], {"input_tokens": 399, "output_tokens": 73})
        self.assertAlmostEqual(out["cost"], 399 * 0.042 / 1_000_000)

    def test_an_api_error_is_raised_by_run_with_its_status(self):
        tool = _tool({"detail": {"error_type": "authentication_error",
                                 "message": "Must supply an API key!"}}, status=403)
        with self.assertRaises(RuntimeError) as ctx:
            tool.run(state=TICKET)
        self.assertIn("[403]", str(ctx.exception))
        self.assertIn("Must supply an API key", str(ctx.exception))

    def test_and_returned_not_raised_by_a_call(self):
        result = _tool({"detail": "x"}, status=422)(state=TICKET)
        self.assertFalse(result.success)
        self.assertIn("[422]", result.error)

    def test_a_reply_without_answers_is_an_error(self):
        with self.assertRaises(RuntimeError):
            _tool({"model": "jev-1.13.0"}).run(state=TICKET)


class TestInsideAnAgent(unittest.TestCase):
    """The reason it is a tool: an agent calls it and reads the numbers."""

    def test_the_agent_calls_it_and_gets_the_probabilities(self):
        from yait_aichain.agent import Agent, step_count
        from yait_aichain.models import Model

        replies = [
            {"choices": [{"message": {"content": None, "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "triage", "arguments": json.dumps({"state": TICKET})}}]}}],
             "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}},
            {"choices": [{"message": {"content": "billing, urgent"}}],
             "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}},
        ]
        model = Model("gpt-4o", api_key="k")
        sent = []

        def post(path, body, headers, *a, **k):
            sent.append(body)
            return json.dumps(replies[min(len(sent) - 1, 1)]).encode()

        model.client._post = MagicMock(side_effect=post)
        model.client._auth_headers = MagicMock(return_value={})

        triage = _tool(name="triage")
        result = Agent(model, tools=[triage], stop_when=[step_count(3)]).run("route this ticket")

        self.assertTrue(result.success)
        self.assertEqual(triage._http.request.call_count, 1)
        # What went back to the model as the tool result carries the numbers.
        self.assertIn("0.95", json.dumps(sent[-1]))
        self.assertIn("billing", json.dumps(sent[-1]))


class TestDecideTypeSafeLive(unittest.TestCase):
    """Real calls: three question types, and an agent that uses the tool."""

    def setUp(self):
        if os.environ.get("TYPESAFE_API_KEY", "") in ("", "test-key", "k"):
            self.skipTest("TYPESAFE_API_KEY not set to a real key")

    def test_three_typed_answers(self):
        out = decideTypeSafe(QUESTIONS).run(state=TICKET)
        a = out["answers"]
        self.assertEqual(set(a), set(QUESTIONS))
        self.assertGreater(a["urgent"]["noul"], 0.5)
        self.assertEqual(a["team"]["choice"], "billing")
        self.assertAlmostEqual(sum(a["team"]["probabilities"].values()), 1.0, places=2)
        self.assertTrue(0.0 <= a["frustration"]["score"] <= 2.0)
        self.assertTrue(out["model"].startswith("jev-"))
        self.assertGreater(out["cost"], 0)


if __name__ == "__main__":
    unittest.main()
