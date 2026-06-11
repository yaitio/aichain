"""
tests.skills.test_skill
========================

Unit tests for skills._skill.Skill.

All tests are pure (no network calls, no env vars required).  The model's
HTTP client is replaced with a mock that records calls and returns
pre-canned responses.
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

_TEST_KEYS = {
    "OPENAI_API_KEY":     "test-openai-key",
    "ANTHROPIC_API_KEY":  "test-anthropic-key",
    "GOOGLE_AI_API_KEY":  "test-google-key",
    "XAI_API_KEY":        "test-xai-key",
    "PERPLEXITY_API_KEY": "test-perplexity-key",
}
for _k, _v in _TEST_KEYS.items():
    if not os.environ.get(_k):
        os.environ[_k] = _v

from models import (
    OpenAIModel,
    AnthropicModel,
    GoogleAIModel,
    XAIModel,
    PerplexityModel,
)
from skills import Skill


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_client(response_body: dict) -> MagicMock:
    """Return a mock client whose _post() returns *response_body* as bytes."""
    client = MagicMock()
    client._post.return_value    = json.dumps(response_body).encode()
    client._auth_headers.return_value = {"Authorization": "Bearer test"}
    return client


def _openai_response(text: str) -> dict:
    return {"choices": [{"message": {"content": text}}]}


def _anthropic_response(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}]}


def _google_response(text: str) -> dict:
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


def _simple_input(text: str = "Hello {name}") -> dict:
    return {
        "messages": [
            {"role": "user", "parts": [{"type": "text", "text": text}]},
        ]
    }


def _text_output() -> dict:
    return {"modalities": ["text"], "format": {"type": "text"}}


def _json_output() -> dict:
    return {"modalities": ["text"], "format": {"type": "json"}}


# ---------------------------------------------------------------------------
# Skill.__init__  — construction and validation
# ---------------------------------------------------------------------------

class TestSkillInit(unittest.TestCase):

    def _make_skill(self, **kw):
        model = OpenAIModel("gpt-4o")
        model.client = _mock_client(_openai_response("ok"))
        defaults = dict(
            model=model,
            input=_simple_input(),
            output=_text_output(),
        )
        defaults.update(kw)
        return Skill(**defaults)

    def test_basic_construction(self):
        skill = self._make_skill()
        self.assertEqual(skill.model.name, "gpt-4o")

    def test_name_and_description_stored(self):
        skill = self._make_skill(name="test_skill", description="A test")
        self.assertEqual(skill.name,        "test_skill")
        self.assertEqual(skill.description, "A test")

    def test_variables_stored(self):
        skill = self._make_skill(variables={"name": "Alice"})
        self.assertEqual(skill.variables["name"], "Alice")

    def test_empty_variables_default(self):
        skill = self._make_skill()
        self.assertEqual(skill.variables, {})

    def test_options_stored(self):
        skill = self._make_skill(options={"key": "val"})
        self.assertEqual(skill.options["key"], "val")

    def test_invalid_input_raises(self):
        model = OpenAIModel("gpt-4o")
        with self.assertRaises(ValueError):
            Skill(model=model, input={}, output=_text_output())

    def test_invalid_output_raises(self):
        model = OpenAIModel("gpt-4o")
        with self.assertRaises(ValueError):
            Skill(
                model=model,
                input=_simple_input(),
                output={"modalities": ["text"], "format": {"type": "xml"}},
            )


# ---------------------------------------------------------------------------
# Skill.__repr__
# ---------------------------------------------------------------------------

class TestSkillRepr(unittest.TestCase):

    def _make(self, **kw):
        model = OpenAIModel("gpt-4o")
        model.client = _mock_client(_openai_response("ok"))
        return Skill(model=model, input=_simple_input(), output=_text_output(), **kw)

    def test_repr_contains_model_name(self):
        self.assertIn("gpt-4o", repr(self._make()))

    def test_repr_contains_skill_name(self):
        s = self._make(name="my_skill")
        self.assertIn("my_skill", repr(s))

    def test_repr_contains_variables(self):
        s = self._make(variables={"x": "1"})
        self.assertIn("variables", repr(s))

    def test_repr_omits_empty_name(self):
        s = self._make()
        self.assertNotIn("name=", repr(s))


# ---------------------------------------------------------------------------
# Skill.run()  — OpenAI
# ---------------------------------------------------------------------------

class TestSkillRunOpenAI(unittest.TestCase):

    def _make_skill(self, input_=None, output=None, variables=None):
        model = OpenAIModel("gpt-4o")
        model.client = _mock_client(_openai_response("The answer is 42."))
        return Skill(
            model=model,
            input=input_    or _simple_input(),
            output=output   or _text_output(),
            variables=variables or {},
        )

    def test_run_returns_string(self):
        result = self._make_skill().run(variables={"name": "World"})
        self.assertIsInstance(result, str)
        self.assertEqual(result, "The answer is 42.")

    def test_run_calls_post(self):
        skill = self._make_skill()
        skill.run(variables={"name": "Test"})
        skill.model.client._post.assert_called_once()

    def test_run_path_is_v1_chat_completions(self):
        skill = self._make_skill()
        skill.run(variables={"name": "Test"})
        call_args = skill.model.client._post.call_args
        path = call_args[0][0]
        self.assertEqual(path, "/v1/chat/completions")

    def test_run_body_contains_model_name(self):
        skill = self._make_skill()
        skill.run(variables={"name": "Test"})
        body = skill.model.client._post.call_args[0][1]
        self.assertEqual(body["model"], "gpt-4o")

    def test_run_substitutes_variables(self):
        skill = self._make_skill()
        skill.run(variables={"name": "Alice"})
        body = skill.model.client._post.call_args[0][1]
        messages = body["messages"]
        # Find the user message content
        user_content = next(m["content"] for m in messages if m["role"] == "user")
        self.assertIn("Alice", user_content)

    def test_run_instance_variables_used_by_default(self):
        skill = self._make_skill(variables={"name": "Bob"})
        skill.run()
        body = skill.model.client._post.call_args[0][1]
        user_content = next(m["content"] for m in body["messages"] if m["role"] == "user")
        self.assertIn("Bob", user_content)

    def test_call_time_variables_override_instance(self):
        skill = self._make_skill(variables={"name": "Bob"})
        skill.run(variables={"name": "Carol"})
        body = skill.model.client._post.call_args[0][1]
        user_content = next(m["content"] for m in body["messages"] if m["role"] == "user")
        self.assertIn("Carol",     user_content)
        self.assertNotIn("Bob",    user_content)

    def test_run_json_output_returns_dict(self):
        model = OpenAIModel("gpt-4o")
        model.client = _mock_client(_openai_response('{"result": "ok"}'))
        skill = Skill(
            model=model,
            input=_simple_input("Respond with JSON."),
            output=_json_output(),
        )
        result = skill.run()
        self.assertIsInstance(result, dict)
        self.assertEqual(result["result"], "ok")

    def test_run_does_not_mutate_original_input(self):
        inp = _simple_input("Tell me about {topic}.")
        model = OpenAIModel("gpt-4o")
        model.client = _mock_client(_openai_response("ok"))
        skill = Skill(model=model, input=inp, output=_text_output())
        skill.run(variables={"topic": "gravity"})
        # Original template must be intact
        self.assertIn("{topic}", inp["messages"][0]["parts"][0]["text"])


# ---------------------------------------------------------------------------
# Skill.run()  — Anthropic
# ---------------------------------------------------------------------------

class TestSkillRunAnthropic(unittest.TestCase):

    def _make_skill(self, **kw):
        model = AnthropicModel("claude-sonnet-4-5", api_key="test-anthropic-key")
        model.client = _mock_client(_anthropic_response("Claude says hi."))
        return Skill(
            model=model,
            input=kw.get("input", _simple_input()),
            output=kw.get("output", _text_output()),
            variables=kw.get("variables", {}),
        )

    def test_run_returns_string(self):
        result = self._make_skill().run(variables={"name": "World"})
        self.assertEqual(result, "Claude says hi.")

    def test_path_is_v1_messages(self):
        skill = self._make_skill()
        skill.run(variables={"name": "T"})
        path = skill.model.client._post.call_args[0][0]
        self.assertEqual(path, "/v1/messages")

    def test_body_uses_max_tokens(self):
        skill = self._make_skill()
        skill.run(variables={"name": "T"})
        body = skill.model.client._post.call_args[0][1]
        self.assertIn("max_tokens", body)
        self.assertNotIn("max_completion_tokens", body)

    def test_system_message_extracted(self):
        inp = {
            "messages": [
                {"role": "system", "parts": [{"type": "text", "text": "Be brief."}]},
                {"role": "user",   "parts": [{"type": "text", "text": "hi {name}"}]},
            ]
        }
        skill = self._make_skill(input=inp)
        skill.run(variables={"name": "X"})
        body = skill.model.client._post.call_args[0][1]
        self.assertEqual(body["system"], "Be brief.")
        for m in body["messages"]:
            self.assertNotEqual(m["role"], "system")


# ---------------------------------------------------------------------------
# Skill.run()  — Google AI
# ---------------------------------------------------------------------------

class TestSkillRunGoogle(unittest.TestCase):

    def _make_skill(self, **kw):
        model = GoogleAIModel("gemini-2.0-flash", api_key="test-google-key")
        model.client = _mock_client(_google_response("Gemini says hello."))
        return Skill(
            model=model,
            input=kw.get("input", _simple_input()),
            output=kw.get("output", _text_output()),
            variables=kw.get("variables", {}),
        )

    def test_run_returns_string(self):
        result = self._make_skill().run(variables={"name": "World"})
        self.assertEqual(result, "Gemini says hello.")

    def test_path_contains_generate_content(self):
        skill = self._make_skill()
        skill.run(variables={"name": "T"})
        path = skill.model.client._post.call_args[0][0]
        self.assertIn("generateContent", path)

    def test_path_contains_api_key(self):
        skill = self._make_skill()
        skill.run(variables={"name": "T"})
        path = skill.model.client._post.call_args[0][0]
        self.assertIn("key=", path)

    def test_body_uses_generation_config(self):
        skill = self._make_skill()
        skill.run(variables={"name": "T"})
        body = skill.model.client._post.call_args[0][1]
        self.assertIn("generationConfig", body)
        self.assertIn("maxOutputTokens",  body["generationConfig"])


# ---------------------------------------------------------------------------
# Skill.run()  — xAI
# ---------------------------------------------------------------------------

class TestSkillRunXAI(unittest.TestCase):

    def test_run_returns_string(self):
        model = XAIModel("grok-3", api_key="test-xai-key")
        model.client = _mock_client(_openai_response("Grok here."))
        skill = Skill(
            model=model,
            input=_simple_input("Hi {name}"),
            output=_text_output(),
        )
        result = skill.run(variables={"name": "World"})
        self.assertEqual(result, "Grok here.")

    def test_path_is_v1_chat_completions(self):
        model = XAIModel("grok-3", api_key="test-xai-key")
        model.client = _mock_client(_openai_response("ok"))
        skill = Skill(model=model, input=_simple_input(), output=_text_output())
        skill.run(variables={"name": "T"})
        path = skill.model.client._post.call_args[0][0]
        self.assertEqual(path, "/v1/chat/completions")


# ---------------------------------------------------------------------------
# Skill.run()  — Perplexity
# ---------------------------------------------------------------------------

class TestSkillRunPerplexity(unittest.TestCase):

    def test_run_returns_string(self):
        model = PerplexityModel("sonar", api_key="test-perplexity-key")
        model.client = _mock_client(_openai_response("Sonar answer."))
        skill = Skill(
            model=model,
            input=_simple_input("What is {topic}?"),
            output=_text_output(),
        )
        result = skill.run(variables={"topic": "AI"})
        self.assertEqual(result, "Sonar answer.")

    def test_path_is_chat_completions(self):
        model = PerplexityModel("sonar-pro", api_key="test-perplexity-key")
        model.client = _mock_client(_openai_response("ok"))
        skill = Skill(model=model, input=_simple_input(), output=_text_output())
        skill.run(variables={"name": "T"})
        path = skill.model.client._post.call_args[0][0]
        self.assertEqual(path, "/chat/completions")


# ---------------------------------------------------------------------------
# Skill._build_request — unsupported model type raises TypeError
# ---------------------------------------------------------------------------

class TestSkillUnsupportedModel(unittest.TestCase):

    def test_unsupported_model_raises_not_implemented(self):
        from models._base import Model

        # Create a bare Model instance (no subclass) — to_request is abstract
        fake = object.__new__(Model)
        fake.name          = "unknown-model"
        fake.temperature   = 1.0
        fake.max_tokens    = 4096
        fake.top_p         = None
        fake.top_k         = None
        fake.cache_control = False
        fake.thinking      = None
        fake._api_key      = "key"
        fake.client        = _mock_client({})

        # Bypass __init__ validation by constructing Skill manually
        skill = object.__new__(Skill)
        skill.model       = fake
        skill._input      = _simple_input()
        skill._output     = _text_output()
        skill.variables   = {}
        skill.options     = {}
        skill.name        = None
        skill.description = None
        skill.max_retries = 0
        skill.retry_delay = 1.0

        with self.assertRaises(NotImplementedError):
            skill.run()


if __name__ == "__main__":
    unittest.main()
