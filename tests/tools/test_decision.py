"""
Decision tools — one format of questions and answers, a provider per tool.

``decisionTypeSafe`` (TypeSafe's Jev) and ``decisionOpenAI`` (OpenAI's
Decisions API) answer the same three kinds of question with probabilities and
disagree on every wire detail: Jev takes a map of questions and calls a
predicate ``noul``; OpenAI takes a list carrying names and returns lists of
probabilities. Built 2026-10-08 the way the search tools are, replacing the
single-provider ``convertToDecision`` before it was released.

What is pinned, offline:

  * one question format, checked at construction for every provider, each
    malformed question named before any network;
  * each provider's documented request, built from that format;
  * one answer format, the same keys from both providers for the same
    question — the reason the base exists;
  * the model sees only ``state`` and a description written from the
    questions; risk is ``read``;
  * images go to the provider that takes them and are refused, by name, by
    the one that does not;
  * cost is input only, at each provider's own price;
  * an agent calls the tool and reads the numbers.

Both response fixtures are real: Jev's from jev-1.13.0 on 2026-10-02,
OpenAI's from gpt-6-luna on 2026-10-08.
"""

import base64
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from yait_aichain.tools import Decision, Record, decide, decisionOpenAI, decisionTypeSafe

QUESTIONS = {
    "urgent": {"type": "predicate", "instructions": "Does this convey urgency?"},
    "team": {"type": "choice", "instructions": "Which team should handle it?",
             "choices": {"billing": "Payments, invoicing, refunds",
                         "technical": "Bugs, outages, integrations",
                         "sales": None}},
    "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                    "levels": ["Calm", {"label": "Frustrated", "description": "Annoyed, still polite"},
                               "Very angry"]},
}
TICKET = "Help! My payouts have been failing for 3 days."

#: Captured live from jev-1.13.0, 2026-10-02.
JEV = {
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

#: Captured live from gpt-6-luna, 2026-10-08, renamed to these questions.
OPENAI = {
    "model": "gpt-6-luna",
    "answers": [
        {"type": "predicate", "name": "urgent", "probability": 0.79},
        {"type": "choice", "name": "team", "choice": "billing", "confidence": 1.0,
         "probabilities": [{"value": "billing", "probability": 1.0},
                           {"value": "technical", "probability": 0.0},
                           {"value": "sales", "probability": 0.0}]},
        {"type": "score", "name": "frustration", "score": 1.39, "confidence": 0.41,
         "probabilities": [{"value": 0, "label": "Calm", "probability": 0.0},
                           {"value": 1, "label": "Frustrated", "probability": 0.61},
                           {"value": 2, "label": "Very angry", "probability": 0.39}]},
    ],
    "usage": {"input_tokens": 395, "input_tokens_details": {"cached_tokens": 0},
              "output_tokens": 0, "total_tokens": 395},
}

PROVIDERS = {decisionTypeSafe: JEV, decisionOpenAI: OPENAI}


def _tool(cls, response=None, status=200, **kw):
    tool = cls(QUESTIONS, api_key="k", **kw)
    reply = MagicMock()
    reply.status = status
    response = PROVIDERS[cls] if response is None else response
    reply.data = (json.dumps(response) if isinstance(response, dict) else response).encode()
    tool._http = MagicMock()
    tool._http.request.return_value = reply
    return tool


def _sent(tool):
    return json.loads(tool._http.request.call_args.kwargs["body"])


class TestOneQuestionFormat(unittest.TestCase):

    def test_a_malformed_question_is_named_before_any_network(self):
        cases = {
            "no questions": ({}, "at least one question"),
            "unknown type": ({"q": {"type": "rank", "instructions": "?"}}, "'q'"),
            "the old word": ({"q": {"type": "noul", "instructions": "?"}}, "word for predicate"),
            "no instructions": ({"q": {"type": "predicate"}}, "'q'"),
            "one choice": ({"q": {"type": "choice", "instructions": "?", "choices": {"a": None}}}, "at least two"),
            "too many choices": ({"q": {"type": "choice", "instructions": "?",
                                        "choices": {str(i): None for i in range(256)}}}, "at most 255"),
            "one level": ({"q": {"type": "score", "instructions": "?", "levels": ["a"]}}, "2 to 10"),
            "eleven levels": ({"q": {"type": "score", "instructions": "?",
                                     "levels": [str(i) for i in range(11)]}}, "2 to 10"),
            "twin labels": ({"q": {"type": "score", "instructions": "?", "levels": ["a", "a"]}}, "must differ"),
            "jev's key": ({"q": {"type": "choice", "instructions": "?", "criteria": {"a": 1}}}, "unknown keys"),
            "levels on a choice": ({"q": {"type": "choice", "instructions": "?",
                                          "choices": {"a": None, "b": None}, "levels": ["x", "y"]}}, "takes no levels"),
        }
        for cls in PROVIDERS:
            for label, (questions, needle) in cases.items():
                with self.subTest(provider=cls.__name__, case=label):
                    with self.assertRaises(ValueError) as ctx:
                        cls(questions, api_key="k")
                    self.assertIn(needle, str(ctx.exception))

    def test_no_key_is_refused_at_construction_naming_the_variable(self):
        for cls, var in ((decisionTypeSafe, "TYPESAFE_API_KEY"), (decisionOpenAI, "OPENAI_API_KEY")):
            saved = os.environ.pop(var, None)
            try:
                with self.subTest(provider=cls.__name__), self.assertRaises(ValueError) as ctx:
                    cls(QUESTIONS)
                self.assertIn(var, str(ctx.exception))
            finally:
                if saved is not None:
                    os.environ[var] = saved


class TestEachProvidersRequest(unittest.TestCase):

    def test_typesafe_gets_its_map_its_words_and_its_criteria(self):
        tool = _tool(decisionTypeSafe, model="jev-1.13.0")
        tool.run(state=TICKET)
        method, url = tool._http.request.call_args.args
        self.assertEqual((method, url), ("POST", "https://api.typesafe.ai/v1/systemone"))
        self.assertEqual(tool._http.request.call_args.kwargs["headers"]["Authorization"], "Bearer k")
        self.assertEqual(_sent(tool), {"state": TICKET, "model": "jev-1.13.0", "questions": {
            "urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
            "team": {"type": "choice", "instructions": "Which team should handle it?",
                     "criteria": {"billing": "Payments, invoicing, refunds",
                                  "technical": "Bugs, outages, integrations", "sales": None}},
            "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                            "criteria": ["Calm", "Frustrated: Annoyed, still polite", "Very angry"]}}})

    def test_openai_gets_its_list_its_names_and_its_objects(self):
        tool = _tool(decisionOpenAI)
        tool.run(state=TICKET)
        method, url = tool._http.request.call_args.args
        self.assertEqual((method, url), ("POST", "https://api.openai.com/v1/decisions"))
        self.assertEqual(_sent(tool), {"model": "gpt-6-luna", "input": TICKET, "questions": [
            {"type": "predicate", "name": "urgent", "instructions": "Does this convey urgency?"},
            {"type": "choice", "name": "team", "instructions": "Which team should handle it?",
             "choices": [{"value": "billing", "description": "Payments, invoicing, refunds"},
                         {"value": "technical", "description": "Bugs, outages, integrations"},
                         {"value": "sales"}]},
            {"type": "score", "name": "frustration", "instructions": "How frustrated is the customer?",
             "levels": [{"label": "Calm"}, {"label": "Frustrated", "description": "Annoyed, still polite"},
                        {"label": "Very angry"}]}]})

    def test_a_structured_state_is_structure_to_jev_and_json_text_to_openai(self):
        record = {"name": "John Smith", "last_employer": "Google"}
        jev = _tool(decisionTypeSafe); jev.run(state=record)
        self.assertEqual(_sent(jev)["state"], record)
        oai = _tool(decisionOpenAI); oai.run(state=record)
        self.assertEqual(json.loads(_sent(oai)["input"]), record)

    def test_an_overloaded_typesafe_is_retried(self):
        # 529 is refused before inference, like the 429 and 503 already retried.
        tool = decisionTypeSafe(QUESTIONS, api_key="k")
        self.assertIn(529, tool._http.connection_pool_kw["retries"].status_forcelist)

    def test_an_empty_state_is_refused(self):
        for cls in PROVIDERS:
            with self.subTest(provider=cls.__name__), self.assertRaises(ValueError):
                _tool(cls).run(state="   ")


class TestOneAnswerFormat(unittest.TestCase):

    def test_both_providers_answer_in_the_same_shape(self):
        jev = _tool(decisionTypeSafe).run(state=TICKET)["answers"]
        oai = _tool(decisionOpenAI).run(state=TICKET)["answers"]
        for a in (jev, oai):
            self.assertEqual(set(a), set(QUESTIONS))
            self.assertEqual(set(a["urgent"]), {"type", "probability"})
            self.assertEqual(set(a["team"]), {"type", "choice", "confidence", "probabilities"})
            self.assertEqual(list(a["team"]["probabilities"]), ["billing", "technical", "sales"])
            self.assertEqual(set(a["frustration"]), {"type", "score", "confidence", "probabilities"})
            self.assertEqual(list(a["frustration"]["probabilities"]), ["Calm", "Frustrated", "Very angry"])

    def test_typesafe_values_survive_the_translation(self):
        a = _tool(decisionTypeSafe).run(state=TICKET)["answers"]
        self.assertEqual(a["urgent"], {"type": "predicate", "probability": 0.95})
        self.assertEqual(a["team"]["probabilities"], {"billing": 0.89, "technical": 0.11, "sales": 0.0})
        self.assertEqual(a["frustration"]["probabilities"], {"Calm": 0.0, "Frustrated": 0.96, "Very angry": 0.04})
        self.assertEqual(a["frustration"]["score"], 1.04)

    def test_openai_values_survive_the_translation(self):
        a = _tool(decisionOpenAI).run(state=TICKET)["answers"]
        self.assertEqual(a["urgent"], {"type": "predicate", "probability": 0.79})
        self.assertEqual(a["team"]["choice"], "billing")
        self.assertEqual(a["frustration"]["probabilities"], {"Calm": 0.0, "Frustrated": 0.61, "Very angry": 0.39})
        self.assertEqual(a["frustration"]["confidence"], 0.41)

    def test_the_version_that_answered_and_input_only_cost(self):
        jev = _tool(decisionTypeSafe).run(state=TICKET)
        self.assertEqual(jev["model"], "jev-1.13.0")
        self.assertEqual(jev["usage"], {"input_tokens": 399, "output_tokens": 73})
        self.assertAlmostEqual(jev["cost"], 399 * 0.042 / 1_000_000)
        oai = _tool(decisionOpenAI).run(state=TICKET)
        self.assertEqual(oai["model"], "gpt-6-luna")
        self.assertAlmostEqual(oai["cost"], 395 * 0.10 / 1_000_000)

    def test_an_api_error_is_raised_by_run_and_returned_by_a_call(self):
        for cls in PROVIDERS:
            with self.subTest(provider=cls.__name__):
                with self.assertRaises(RuntimeError) as ctx:
                    _tool(cls, {"error": "bad key"}, status=401).run(state=TICKET)
                self.assertIn("[401]", str(ctx.exception))
                result = _tool(cls, {"error": "x"}, status=422)(state=TICKET)
                self.assertFalse(result.success)
                self.assertIn("[422]", result.error)

    def test_a_reply_without_answers_is_an_error(self):
        for cls in PROVIDERS:
            with self.subTest(provider=cls.__name__), self.assertRaises(RuntimeError):
                _tool(cls, {"model": "m"}).run(state=TICKET)


class TestImages(unittest.TestCase):

    def test_openai_takes_them_as_data_urls_beside_the_text(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fh:
            fh.write(b"\x89PNG fake")
        try:
            tool = _tool(decisionOpenAI)
            tool.run(state="Inspect this photo.", images=[fh.name, b"raw", "data:image/jpeg;base64,AAAA"])
            msg = _sent(tool)["input"]
            self.assertEqual(msg[0]["role"], "user")
            parts = msg[0]["content"]
            self.assertEqual(parts[0], {"type": "input_text", "text": "Inspect this photo."})
            self.assertEqual(parts[1]["image_url"], "data:image/png;base64," + base64.b64encode(b"\x89PNG fake").decode())
            self.assertTrue(parts[2]["image_url"].startswith("data:image/png;base64,"))
            self.assertEqual(parts[3]["image_url"], "data:image/jpeg;base64,AAAA")
        finally:
            os.unlink(fh.name)

    def test_a_hosted_url_is_refused_before_the_network(self):
        tool = _tool(decisionOpenAI)
        with self.assertRaises(ValueError):
            tool.run(state="x", images=["https://example.com/a.png"])
        tool._http.request.assert_not_called()

    def test_typesafe_refuses_them_by_name(self):
        tool = _tool(decisionTypeSafe)
        with self.assertRaises(ValueError) as ctx:
            tool.run(state="x", images=[b"img"])
        self.assertIn("takes no images", str(ctx.exception))
        tool._http.request.assert_not_called()


class TestWhatTheModelSees(unittest.TestCase):

    def test_only_state_a_description_from_the_questions_and_read_risk(self):
        for cls in PROVIDERS:
            with self.subTest(provider=cls.__name__):
                tool = cls(QUESTIONS, name="triage", api_key="k")
                self.assertEqual(list(tool.parameters["properties"]), ["state"])
                self.assertEqual(tool.schema()["function"]["name"], "triage")
                self.assertIn("urgent: Does this convey urgency? → probability", tool.description)
                self.assertIn("one of billing, technical, sales", tool.description)
                self.assertIn("Calm < Frustrated < Very angry", tool.description)
                self.assertEqual(tool.risk, "read")
                self.assertIsInstance(tool, Decision)

    def test_the_default_name_is_the_providers(self):
        self.assertEqual(decisionTypeSafe(QUESTIONS, api_key="k").name, "decisionTypeSafe")
        self.assertEqual(decisionOpenAI(QUESTIONS, api_key="k").name, "decisionOpenAI")


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

        triage = _tool(decisionTypeSafe, name="triage")
        result = Agent(model, tools=[triage], stop_when=[step_count(3)]).run("route this ticket")

        self.assertTrue(result.success)
        self.assertEqual(triage._http.request.call_count, 1)
        self.assertIn("0.95", json.dumps(sent[-1]))
        self.assertIn("billing", json.dumps(sent[-1]))



class TestWhatACallReturns(unittest.TestCase):
    """3.3.0: a Record — the fields for code, one line per answer for a model."""

    def test_a_record_with_where_the_cost_came_from(self):
        for cls in PROVIDERS:
            with self.subTest(cls.__name__):
                out = _tool(cls).run(TICKET)
                self.assertIsInstance(out, Record)
                self.assertEqual(set(out), {"answers", "model", "usage", "cost", "cost_basis"})
                self.assertIn(f"input tokens reported by {cls._PROVIDER}", out["cost_basis"])
                self.assertIn("read 2026-", out["cost_basis"])

    def test_the_model_reads_one_line_per_answer(self):
        text = str(_tool(decisionTypeSafe).run(TICKET))
        lines = text.splitlines()
        self.assertEqual(len(lines), len(QUESTIONS))
        self.assertTrue(any(l.startswith("urgent: yes with probability 0.") for l in lines), text)

    def test_a_chain_keeps_it_under_its_key(self):
        # A plain dict is merged into a chain's variables; a record is one
        # result, so two decisions in one chain do not overwrite `answers`.
        from yait_aichain.chain import Chain
        a = _tool(decisionTypeSafe, name="first")
        b = _tool(decisionTypeSafe, name="second")
        res = Chain(steps=[(a, "one", {"state": "ticket"}), (b, "two", {"state": "ticket"})]).run(
            variables={"ticket": TICKET})
        self.assertIn("answers", res["one"])
        self.assertIn("answers", res["two"])


class TestFactory(unittest.TestCase):
    """decide(service, questions=...) — the one way the documentation shows."""

    def test_every_service_by_name(self):
        from yait_aichain.tools.decision import SERVICES
        self.assertEqual(SERVICES, {"openai": decisionOpenAI, "typesafe": decisionTypeSafe})
        for service, cls in SERVICES.items():
            with self.subTest(service):
                t = decide(service, questions=QUESTIONS, api_key="k", name="triage", model="m-1")
                self.assertIs(type(t), cls)
                self.assertEqual((t.name, t.model), ("triage", "m-1"))

    def test_questions_are_required_and_checked_here(self):
        with self.assertRaises(TypeError):
            decide("openai", api_key="k")
        with self.assertRaises(ValueError):
            decide("openai", api_key="k", questions={"x": {"type": "noul", "instructions": "?"}})

    def test_an_unknown_service_names_the_ones_that_exist(self):
        with self.assertRaisesRegex(ValueError, "one of: openai, typesafe"):
            decide("jev", questions=QUESTIONS, api_key="k")


def _real(var):
    return os.environ.get(var, "") not in ("", "test-key", "k")


class TestDecisionLive(unittest.TestCase):
    """Real calls: the same questions to both providers, the same shape back."""

    def _check(self, out):
        a = out["answers"]
        self.assertEqual(set(a), set(QUESTIONS))
        self.assertGreater(a["urgent"]["probability"], 0.5)
        self.assertEqual(a["team"]["choice"], "billing")
        self.assertAlmostEqual(sum(a["team"]["probabilities"].values()), 1.0, places=1)
        self.assertTrue(0.0 <= a["frustration"]["score"] <= 2.0)
        self.assertGreater(out["cost"], 0)

    def test_typesafe(self):
        if not _real("TYPESAFE_API_KEY"):
            self.skipTest("TYPESAFE_API_KEY not set to a real key")
        out = decisionTypeSafe(QUESTIONS).run(state=TICKET)
        self._check(out)
        self.assertTrue(out["model"].startswith("jev-"))

    def test_openai(self):
        if not _real("OPENAI_API_KEY"):
            self.skipTest("OPENAI_API_KEY not set to a real key")
        out = decisionOpenAI(QUESTIONS).run(state=TICKET)
        self._check(out)
        self.assertEqual(out["model"], "gpt-6-luna")


if __name__ == "__main__":
    unittest.main()
