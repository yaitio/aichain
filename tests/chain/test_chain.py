"""
tests.chain.test_chain
=======================

Unit tests for chain._chain.Chain.

All tests are pure — no network calls, no real API keys.  Each skill's
HTTP client is replaced with a mock that returns a pre-canned response.
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

_TEST_KEYS = {
    "OPENAI_API_KEY":    "test-openai-key",
    "ANTHROPIC_API_KEY": "test-anthropic-key",
    "GOOGLE_AI_API_KEY": "test-google-key",
}
for _k, _v in _TEST_KEYS.items():
    if not os.environ.get(_k):
        os.environ[_k] = _v

from models import OpenAIModel, AnthropicModel
from skills import Skill
from chain  import Chain


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_client(response_body: dict) -> MagicMock:
    client = MagicMock()
    client._post.return_value     = json.dumps(response_body).encode()
    client._auth_headers.return_value = {"Authorization": "Bearer test"}
    return client


def _openai_response(text: str) -> dict:
    return {"choices": [{"message": {"content": text}}]}


def _openai_json_response(data: dict) -> dict:
    return {"choices": [{"message": {"content": json.dumps(data)}}]}


def _make_skill(name: str, response_text: str, placeholder: str = "input") -> Skill:
    """Build a minimal Skill with a mocked client returning *response_text*."""
    model = OpenAIModel("gpt-4o")
    model.client = _mock_client(_openai_response(response_text))
    return Skill(
        model=model,
        input={"messages": [
            {"role": "user", "parts": [
                {"type": "text", "text": f"Process: {{{placeholder}}}"},
            ]},
        ]},
        output={"modalities": ["text"], "format": {"type": "text"}},
        name=name,
    )


def _make_json_skill(name: str, response_data: dict) -> Skill:
    """Build a Skill whose output is a JSON dict."""
    model = OpenAIModel("gpt-4o")
    model.client = _mock_client(_openai_json_response(response_data))
    return Skill(
        model=model,
        input={"messages": [
            {"role": "user", "parts": [
                {"type": "text", "text": "Return JSON for {topic}"},
            ]},
        ]},
        output={"modalities": ["text"], "format": {"type": "json"}},
        name=name,
    )


# ---------------------------------------------------------------------------
# Chain.__init__ — construction and validation
# ---------------------------------------------------------------------------

class TestChainInit(unittest.TestCase):

    def test_basic_construction(self):
        s = _make_skill("s1", "ok")
        c = Chain([s])
        self.assertEqual(len(c._steps), 1)

    def test_name_and_description_stored(self):
        s = _make_skill("s1", "ok")
        c = Chain([s], name="my_chain", description="does stuff")
        self.assertEqual(c.name,        "my_chain")
        self.assertEqual(c.description, "does stuff")

    def test_default_variables_stored(self):
        s = _make_skill("s1", "ok")
        c = Chain([s], variables={"lang": "French"})
        self.assertEqual(c.variables["lang"], "French")

    def test_empty_steps_raises(self):
        with self.assertRaises(ValueError):
            Chain([])

    def test_bare_skill_gets_default_output_key(self):
        s = _make_skill("s1", "ok")
        c = Chain([s])
        _, key, _imap, _kind, _opts = c._steps[0]
        self.assertEqual(key, "result")

    def test_tuple_step_uses_custom_output_key(self):
        s = _make_skill("s1", "ok")
        c = Chain([(s, "summary")])
        _, key, _imap, _kind, _opts = c._steps[0]
        self.assertEqual(key, "summary")

    def test_invalid_tuple_raises(self):
        s = _make_skill("s1", "ok")
        with self.assertRaises(ValueError):
            Chain([(s, 42)])  # key must be str

    def test_history_empty_before_run(self):
        s = _make_skill("s1", "ok")
        c = Chain([s])
        self.assertEqual(c.history, [])


# ---------------------------------------------------------------------------
# Chain.__repr__
# ---------------------------------------------------------------------------

class TestChainRepr(unittest.TestCase):

    def test_repr_contains_step_name(self):
        s = _make_skill("analyser", "ok")
        c = Chain([s], name="pipe")
        r = repr(c)
        self.assertIn("analyser", r)
        self.assertIn("pipe",     r)
        self.assertIn("result",   r)   # default output key

    def test_repr_shows_custom_output_key(self):
        s = _make_skill("analyser", "ok")
        c = Chain([(s, "analysis")])
        self.assertIn("analysis", repr(c))


# ---------------------------------------------------------------------------
# Chain.run() — single step
# ---------------------------------------------------------------------------

class TestChainRunSingleStep(unittest.TestCase):

    def test_returns_skill_output(self):
        s = _make_skill("s1", "hello world", placeholder="x")
        c = Chain([s])
        result = c.run(variables={"x": "test"})
        self.assertEqual(result, "hello world")

    def test_run_calls_skill_post_once(self):
        s = _make_skill("s1", "ok", placeholder="x")
        c = Chain([s])
        c.run(variables={"x": "v"})
        s.model.client._post.assert_called_once()

    def test_history_has_one_record(self):
        s = _make_skill("s1", "out", placeholder="x")
        c = Chain([s])
        c.run(variables={"x": "in"})
        self.assertEqual(len(c.history), 1)

    def test_history_record_fields(self):
        s = _make_skill("named_skill", "the output", placeholder="x")
        c = Chain([s])
        c.run(variables={"x": "the input"})
        rec = c.history[0]
        self.assertEqual(rec["step"],       0)
        self.assertEqual(rec["name"],       "named_skill")
        self.assertEqual(rec["output"],     "the output")
        self.assertEqual(rec["output_key"], "result")
        self.assertIn("x", rec["input"])

    def test_history_is_a_copy(self):
        s = _make_skill("s1", "ok", placeholder="x")
        c = Chain([s])
        c.run(variables={"x": "v"})
        h = c.history
        h.clear()
        self.assertEqual(len(c.history), 1)   # original unaffected


# ---------------------------------------------------------------------------
# Chain.run() — variable flow between steps
# ---------------------------------------------------------------------------

class TestChainVariableFlow(unittest.TestCase):

    def _two_step_chain(self, step1_output: str) -> tuple:
        """
        Returns (chain, step2_skill) where step1 always returns *step1_output*
        and step2's model client records what body it received.
        """
        # Step 1: summariser
        s1 = _make_skill("summariser", step1_output, placeholder="article")

        # Step 2: translator — needs {result} and {language}
        s2_model = OpenAIModel("gpt-4o")
        s2_model.client = _mock_client(_openai_response("translated text"))
        s2 = Skill(
            model=s2_model,
            input={"messages": [
                {"role": "user", "parts": [
                    {"type": "text",
                     "text": "Translate to {language}: {result}"},
                ]},
            ]},
            output={"modalities": ["text"], "format": {"type": "text"}},
            name="translator",
        )
        chain = Chain([s1, s2])
        return chain, s2

    def test_step1_output_injected_as_result(self):
        chain, s2 = self._two_step_chain("the summary")
        chain.run(variables={"article": "long text", "language": "French"})

        body = s2.model.client._post.call_args[0][1]
        user_content = next(
            m["content"] for m in body["messages"] if m["role"] == "user"
        )
        self.assertIn("the summary", user_content)
        self.assertIn("French",      user_content)

    def test_initial_vars_available_in_step2(self):
        chain, s2 = self._two_step_chain("summary text")
        chain.run(variables={"article": "text", "language": "Spanish"})

        body = s2.model.client._post.call_args[0][1]
        user_content = next(
            m["content"] for m in body["messages"] if m["role"] == "user"
        )
        self.assertIn("Spanish", user_content)

    def test_final_return_is_last_step_output(self):
        chain, _ = self._two_step_chain("the summary")
        result = chain.run(variables={"article": "text", "language": "DE"})
        self.assertEqual(result, "translated text")

    def test_custom_output_key_flows_to_next_step(self):
        # Step 1 uses output_key="analysis"
        s1 = _make_skill("analyst", "the analysis", placeholder="topic")

        s2_model = OpenAIModel("gpt-4o")
        s2_model.client = _mock_client(_openai_response("formatted"))
        s2 = Skill(
            model=s2_model,
            input={"messages": [
                {"role": "user", "parts": [
                    {"type": "text", "text": "Format this: {analysis}"},
                ]},
            ]},
            output={"modalities": ["text"], "format": {"type": "text"}},
            name="formatter",
        )
        chain = Chain([(s1, "analysis"), s2])
        chain.run(variables={"topic": "AI"})

        body = s2_model.client._post.call_args[0][1]
        user_content = next(
            m["content"] for m in body["messages"] if m["role"] == "user"
        )
        self.assertIn("the analysis", user_content)

    def test_dict_output_merged_into_accumulated(self):
        # Step 1 returns a dict; its keys become variables for step 2
        s1 = _make_json_skill("extractor", {"title": "AI News", "lang": "de"})

        s2_model = OpenAIModel("gpt-4o")
        s2_model.client = _mock_client(_openai_response("done"))
        s2 = Skill(
            model=s2_model,
            input={"messages": [
                {"role": "user", "parts": [
                    {"type": "text",
                     "text": "Translate {title} to {lang}"},
                ]},
            ]},
            output={"modalities": ["text"], "format": {"type": "text"}},
            name="translator",
        )
        chain = Chain([s1, s2])
        chain.run(variables={"topic": "something"})

        body = s2_model.client._post.call_args[0][1]
        user_content = next(
            m["content"] for m in body["messages"] if m["role"] == "user"
        )
        self.assertIn("AI News", user_content)
        self.assertIn("de",      user_content)


# ---------------------------------------------------------------------------
# Chain.run() — three-step pipeline
# ---------------------------------------------------------------------------

class TestChainThreeSteps(unittest.TestCase):

    def test_three_steps_all_called(self):
        s1 = _make_skill("s1", "out1", placeholder="x")
        s2 = _make_skill("s2", "out2", placeholder="result")
        s3 = _make_skill("s3", "out3", placeholder="result")
        chain = Chain([s1, s2, s3])
        result = chain.run(variables={"x": "start"})
        self.assertEqual(result, "out3")
        self.assertEqual(len(chain.history), 3)

    def test_history_order(self):
        s1 = _make_skill("first",  "a", placeholder="x")
        s2 = _make_skill("second", "b", placeholder="result")
        s3 = _make_skill("third",  "c", placeholder="result")
        chain = Chain([s1, s2, s3])
        chain.run(variables={"x": "go"})
        names = [r["name"] for r in chain.history]
        self.assertEqual(names, ["first", "second", "third"])

    def test_each_step_sees_all_prior_outputs(self):
        s1 = _make_skill("s1", "alpha", placeholder="start")

        s2_model = OpenAIModel("gpt-4o")
        s2_model.client = _mock_client(_openai_response("beta"))
        s2 = Skill(
            model=s2_model,
            input={"messages": [
                {"role": "user", "parts": [
                    {"type": "text", "text": "{result} + {start}"},
                ]},
            ]},
            output={"modalities": ["text"], "format": {"type": "text"}},
            name="s2",
        )
        chain = Chain([s1, s2])
        chain.run(variables={"start": "zero"})

        body = s2_model.client._post.call_args[0][1]
        user_content = next(
            m["content"] for m in body["messages"] if m["role"] == "user"
        )
        self.assertIn("alpha", user_content)   # from s1 output
        self.assertIn("zero",  user_content)   # from initial vars


# ---------------------------------------------------------------------------
# Chain.run() — variable precedence
# ---------------------------------------------------------------------------

class TestChainVariablePrecedence(unittest.TestCase):

    def test_call_time_vars_override_chain_defaults(self):
        s = _make_skill("s", "ok", placeholder="lang")
        c = Chain([s], variables={"lang": "French"})
        c.run(variables={"lang": "German"})
        body = s.model.client._post.call_args[0][1]
        user_content = next(
            m["content"] for m in body["messages"] if m["role"] == "user"
        )
        self.assertIn("German",    user_content)
        self.assertNotIn("French", user_content)

    def test_chain_defaults_used_when_no_call_time_vars(self):
        s = _make_skill("s", "ok", placeholder="lang")
        c = Chain([s], variables={"lang": "Italian"})
        c.run()
        body = s.model.client._post.call_args[0][1]
        user_content = next(
            m["content"] for m in body["messages"] if m["role"] == "user"
        )
        self.assertIn("Italian", user_content)


# ---------------------------------------------------------------------------
# Chain.run() — history reset between runs
# ---------------------------------------------------------------------------

class TestChainHistoryReset(unittest.TestCase):

    def test_history_reset_on_each_run(self):
        s = _make_skill("s", "out", placeholder="x")
        c = Chain([s])
        c.run(variables={"x": "first"})
        c.run(variables={"x": "second"})
        self.assertEqual(len(c.history), 1)
        self.assertEqual(c.history[0]["input"]["x"], "second")

    def test_input_snapshot_not_mutated_by_later_steps(self):
        s1 = _make_skill("s1", "step1_out", placeholder="x")
        s2 = _make_skill("s2", "step2_out", placeholder="result")
        chain = Chain([s1, s2])
        chain.run(variables={"x": "start"})
        # Step 1's snapshot should NOT contain "result" (it ran before step 1)
        self.assertNotIn("result", chain.history[0]["input"])
        # Step 2's snapshot SHOULD contain "result" (injected after step 1)
        self.assertIn("result", chain.history[1]["input"])


# ---------------------------------------------------------------------------
# Chain with unnamed skills
# ---------------------------------------------------------------------------

class TestChainUnnamedSkills(unittest.TestCase):

    def test_unnamed_skill_gets_step_label(self):
        model = OpenAIModel("gpt-4o")
        model.client = _mock_client(_openai_response("out"))
        s = Skill(
            model=model,
            input={"messages": [
                {"role": "user", "parts": [{"type": "text", "text": "hi {x}"}]},
            ]},
            output={"modalities": ["text"], "format": {"type": "text"}},
            # name intentionally omitted
        )
        chain = Chain([s])
        chain.run(variables={"x": "there"})
        self.assertEqual(chain.history[0]["name"], "step_0")


if __name__ == "__main__":
    unittest.main()
