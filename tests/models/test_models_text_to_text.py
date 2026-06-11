"""
tests.models.test_models_text_to_text
======================================

Tests for every provider's text-to-text path:
  1. Factory routing  — Model("name") → correct subclass
  2. to_request()     — correct endpoint path + essential body keys
  3. from_response()  — correct string extracted from raw JSON

All tests are pure (no network, no real API keys).

Live tests (gated on env vars) verify a real round-trip.
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Seed env vars before any model import.
# Use explicit assignment to override empty-string values that setdefault() would leave alone.
_TEST_KEYS = {
    "OPENAI_API_KEY":     "test-openai-key",
    "ANTHROPIC_API_KEY":  "test-anthropic-key",
    "GOOGLE_AI_API_KEY":  "test-google-key",
    "XAI_API_KEY":        "test-xai-key",
    "PERPLEXITY_API_KEY": "test-perplexity-key",
    "DEEPSEEK_API_KEY":   "test-deepseek-key",
    "MOONSHOT_API_KEY":   "test-kimi-key",
    "DASHSCOPE_API_KEY":  "test-qwen-key",
}
for _k, _v in _TEST_KEYS.items():
    if not os.environ.get(_k):
        os.environ[_k] = _v

from models import (
    Model,
    OpenAIModel,
    AnthropicModel,
    GoogleAIModel,
    XAIModel,
    PerplexityModel,
    DeepSeekModel,
    KimiModel,
    QwenModel,
)

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

_USER_MSG = [
    {"role": "user", "parts": [{"type": "text", "text": "Hello"}]},
]

_SYSTEM_USER_MSGS = [
    {"role": "system", "parts": [{"type": "text", "text": "Be helpful."}]},
    {"role": "user",   "parts": [{"type": "text", "text": "Hello"}]},
]

_TEXT_OUTPUT = {"modalities": ["text"], "format": {"type": "text"}}


# ---------------------------------------------------------------------------
# 1. Factory routing
# ---------------------------------------------------------------------------

class TestModelFactory(unittest.TestCase):
    """Model("name") must instantiate the exact provider subclass."""

    def test_gpt_routes_to_openai(self):
        self.assertIsInstance(Model("gpt-4o"), OpenAIModel)

    def test_o_series_routes_to_openai(self):
        self.assertIsInstance(Model("o3"), OpenAIModel)

    def test_claude_routes_to_anthropic(self):
        self.assertIsInstance(Model("claude-opus-4-6"), AnthropicModel)

    def test_gemini_routes_to_google(self):
        self.assertIsInstance(Model("gemini-2.5-flash"), GoogleAIModel)

    def test_grok_routes_to_xai(self):
        self.assertIsInstance(Model("grok-3"), XAIModel)

    def test_sonar_routes_to_perplexity(self):
        self.assertIsInstance(Model("sonar-pro"), PerplexityModel)

    def test_r1_routes_to_perplexity(self):
        self.assertIsInstance(Model("r1-1776"), PerplexityModel)

    def test_deepseek_routes_to_deepseek(self):
        self.assertIsInstance(Model("deepseek-chat"), DeepSeekModel)

    def test_kimi_routes_to_kimi(self):
        self.assertIsInstance(Model("kimi-k2-0905-preview"), KimiModel)

    def test_qwen_routes_to_qwen(self):
        self.assertIsInstance(Model("qwen-max"), QwenModel)

    def test_qwq_routes_to_qwen(self):
        self.assertIsInstance(Model("QwQ-32B"), QwenModel)

    def test_unknown_model_raises_value_error(self):
        with self.assertRaises(ValueError):
            Model("totally-unknown-model-xyz")

    def test_direct_subclass_construction_bypasses_factory(self):
        m = OpenAIModel("gpt-4o")
        self.assertIsInstance(m, OpenAIModel)


# ---------------------------------------------------------------------------
# 2a. to_request() — OpenAI (Chat Completions)
# ---------------------------------------------------------------------------

class TestOpenAIToRequest(unittest.TestCase):

    def setUp(self):
        self.model = OpenAIModel("gpt-4o")

    def test_path_is_chat_completions(self):
        path, _ = self.model.to_request(_USER_MSG, _TEXT_OUTPUT)
        self.assertEqual(path, "/v1/chat/completions")

    def test_body_has_model_name(self):
        _, body = self.model.to_request(_USER_MSG, _TEXT_OUTPUT)
        self.assertEqual(body["model"], "gpt-4o")

    def test_body_has_messages(self):
        _, body = self.model.to_request(_USER_MSG, _TEXT_OUTPUT)
        self.assertIn("messages", body)
        self.assertEqual(len(body["messages"]), 1)
        self.assertEqual(body["messages"][0]["role"], "user")

    def test_body_has_max_completion_tokens(self):
        _, body = self.model.to_request(_USER_MSG, _TEXT_OUTPUT)
        self.assertIn("max_completion_tokens", body)
        self.assertNotIn("max_tokens", body)

    def test_body_has_temperature(self):
        _, body = self.model.to_request(_USER_MSG, _TEXT_OUTPUT)
        self.assertIn("temperature", body)

    def test_system_message_stays_inside_messages(self):
        _, body = self.model.to_request(_SYSTEM_USER_MSGS, _TEXT_OUTPUT)
        roles = [m["role"] for m in body["messages"]]
        self.assertIn("system", roles)
        self.assertNotIn("system", body)  # NOT a top-level key

    def test_gpt5_routes_to_responses_api(self):
        model = OpenAIModel("gpt-5")
        path, body = model.to_request(_USER_MSG, _TEXT_OUTPUT)
        self.assertEqual(path, "/v1/responses")
        self.assertIn("input", body)
        self.assertNotIn("messages", body)

    def test_gpt5_uses_max_output_tokens_not_max_completion_tokens(self):
        model = OpenAIModel("gpt-5")
        _, body = model.to_request(_USER_MSG, _TEXT_OUTPUT)
        self.assertIn("max_output_tokens", body)
        self.assertNotIn("max_completion_tokens", body)


# ---------------------------------------------------------------------------
# 2b. to_request() — Anthropic
# ---------------------------------------------------------------------------

class TestAnthropicToRequest(unittest.TestCase):

    def setUp(self):
        self.model = AnthropicModel("claude-opus-4-6")

    def test_path_is_messages(self):
        path, _ = self.model.to_request(_USER_MSG, _TEXT_OUTPUT)
        self.assertEqual(path, "/v1/messages")

    def test_body_has_model_name(self):
        _, body = self.model.to_request(_USER_MSG, _TEXT_OUTPUT)
        self.assertEqual(body["model"], "claude-opus-4-6")

    def test_body_has_messages(self):
        _, body = self.model.to_request(_USER_MSG, _TEXT_OUTPUT)
        self.assertIn("messages", body)

    def test_body_uses_max_tokens_not_max_completion_tokens(self):
        _, body = self.model.to_request(_USER_MSG, _TEXT_OUTPUT)
        self.assertIn("max_tokens", body)
        self.assertNotIn("max_completion_tokens", body)

    def test_system_message_lifted_to_top_level(self):
        _, body = self.model.to_request(_SYSTEM_USER_MSGS, _TEXT_OUTPUT)
        self.assertIn("system", body)
        # System must NOT appear inside messages[]
        for msg in body["messages"]:
            self.assertNotEqual(msg["role"], "system")

    def test_system_message_content_is_extracted(self):
        _, body = self.model.to_request(_SYSTEM_USER_MSGS, _TEXT_OUTPUT)
        self.assertIn("Be helpful", body["system"])


# ---------------------------------------------------------------------------
# 2c. to_request() — Google AI
# ---------------------------------------------------------------------------

class TestGoogleToRequest(unittest.TestCase):

    def setUp(self):
        self.model = GoogleAIModel("gemini-2.5-flash")

    def test_path_contains_model_name(self):
        path, _ = self.model.to_request(_USER_MSG, _TEXT_OUTPUT)
        self.assertIn("gemini-2.5-flash", path)

    def test_path_contains_generate_content(self):
        path, _ = self.model.to_request(_USER_MSG, _TEXT_OUTPUT)
        self.assertIn("generateContent", path)

    def test_path_contains_api_key_param(self):
        path, _ = self.model.to_request(_USER_MSG, _TEXT_OUTPUT)
        self.assertIn("?key=", path)

    def test_body_has_contents_not_messages(self):
        _, body = self.model.to_request(_USER_MSG, _TEXT_OUTPUT)
        self.assertIn("contents", body)
        self.assertNotIn("messages", body)

    def test_body_has_generation_config(self):
        _, body = self.model.to_request(_USER_MSG, _TEXT_OUTPUT)
        self.assertIn("generationConfig", body)

    def test_generation_config_has_temperature_and_max_tokens(self):
        _, body = self.model.to_request(_USER_MSG, _TEXT_OUTPUT)
        gc = body["generationConfig"]
        self.assertIn("temperature", gc)
        self.assertIn("maxOutputTokens", gc)

    def test_system_message_in_system_instruction(self):
        _, body = self.model.to_request(_SYSTEM_USER_MSGS, _TEXT_OUTPUT)
        self.assertIn("system_instruction", body)
        # System content must NOT appear in contents[]
        for content in body["contents"]:
            self.assertNotEqual(content.get("role"), "system")

    def test_google_user_role_mapped_correctly(self):
        _, body = self.model.to_request(_USER_MSG, _TEXT_OUTPUT)
        self.assertEqual(body["contents"][0]["role"], "user")

    def test_google_assistant_role_mapped_to_model(self):
        msgs = [{"role": "assistant", "parts": [{"type": "text", "text": "Hi"}]}]
        _, body = self.model.to_request(msgs, _TEXT_OUTPUT)
        self.assertEqual(body["contents"][0]["role"], "model")


# ---------------------------------------------------------------------------
# 2d. to_request() — xAI
# ---------------------------------------------------------------------------

class TestXAIToRequest(unittest.TestCase):

    def setUp(self):
        self.model = XAIModel("grok-3")

    def test_path_is_chat_completions(self):
        path, _ = self.model.to_request(_USER_MSG, _TEXT_OUTPUT)
        self.assertEqual(path, "/v1/chat/completions")

    def test_body_has_model_name(self):
        _, body = self.model.to_request(_USER_MSG, _TEXT_OUTPUT)
        self.assertEqual(body["model"], "grok-3")

    def test_body_has_max_completion_tokens(self):
        _, body = self.model.to_request(_USER_MSG, _TEXT_OUTPUT)
        self.assertIn("max_completion_tokens", body)


# ---------------------------------------------------------------------------
# 2e. to_request() — Perplexity
# ---------------------------------------------------------------------------

class TestPerplexityToRequest(unittest.TestCase):

    def setUp(self):
        self.model = PerplexityModel("sonar")

    def test_path_is_chat_completions_without_v1_prefix(self):
        path, _ = self.model.to_request(_USER_MSG, _TEXT_OUTPUT)
        self.assertEqual(path, "/chat/completions")

    def test_body_has_model_name(self):
        _, body = self.model.to_request(_USER_MSG, _TEXT_OUTPUT)
        self.assertEqual(body["model"], "sonar")

    def test_body_has_messages(self):
        _, body = self.model.to_request(_USER_MSG, _TEXT_OUTPUT)
        self.assertIn("messages", body)


# ---------------------------------------------------------------------------
# 2f. to_request() — DeepSeek
# ---------------------------------------------------------------------------

class TestDeepSeekToRequest(unittest.TestCase):

    def setUp(self):
        self.model = DeepSeekModel("deepseek-chat")

    def test_path_is_chat_completions(self):
        path, _ = self.model.to_request(_USER_MSG, _TEXT_OUTPUT)
        self.assertEqual(path, "/v1/chat/completions")

    def test_body_model_is_deepseek_chat(self):
        _, body = self.model.to_request(_USER_MSG, _TEXT_OUTPUT)
        self.assertEqual(body["model"], "deepseek-chat")

    def test_chat_model_includes_temperature(self):
        _, body = self.model.to_request(_USER_MSG, _TEXT_OUTPUT)
        self.assertIn("temperature", body)

    def test_reasoner_omits_temperature(self):
        model = DeepSeekModel("deepseek-chat", options={"reasoning": "high"})
        _, body = model.to_request(_USER_MSG, _TEXT_OUTPUT)
        self.assertEqual(body["model"], "deepseek-reasoner")
        self.assertNotIn("temperature", body)


# ---------------------------------------------------------------------------
# 2g. to_request() — Kimi
# ---------------------------------------------------------------------------

class TestKimiToRequest(unittest.TestCase):

    def setUp(self):
        self.model = KimiModel("kimi-k2-0905-preview")

    def test_path_is_chat_completions(self):
        path, _ = self.model.to_request(_USER_MSG, _TEXT_OUTPUT)
        self.assertEqual(path, "/v1/chat/completions")

    def test_body_has_model_name(self):
        _, body = self.model.to_request(_USER_MSG, _TEXT_OUTPUT)
        self.assertEqual(body["model"], "kimi-k2-0905-preview")

    def test_body_has_messages(self):
        _, body = self.model.to_request(_USER_MSG, _TEXT_OUTPUT)
        self.assertIn("messages", body)


# ---------------------------------------------------------------------------
# 2h. to_request() — Qwen
# ---------------------------------------------------------------------------

class TestQwenToRequest(unittest.TestCase):

    def setUp(self):
        self.model = QwenModel("qwen-max")

    def test_path_uses_compatible_mode(self):
        path, _ = self.model.to_request(_USER_MSG, _TEXT_OUTPUT)
        self.assertEqual(path, "/compatible-mode/v1/chat/completions")

    def test_body_has_model_name(self):
        _, body = self.model.to_request(_USER_MSG, _TEXT_OUTPUT)
        self.assertEqual(body["model"], "qwen-max")

    def test_body_has_messages(self):
        _, body = self.model.to_request(_USER_MSG, _TEXT_OUTPUT)
        self.assertIn("messages", body)

    def test_qwq_always_gets_enable_thinking(self):
        model = QwenModel("QwQ-32B")
        _, body = model.to_request(_USER_MSG, _TEXT_OUTPUT)
        self.assertTrue(body.get("enable_thinking"))

    def test_qwen3_gets_enable_thinking_when_reasoning_set(self):
        model = QwenModel("qwen3-72b", options={"reasoning": "high"})
        _, body = model.to_request(_USER_MSG, _TEXT_OUTPUT)
        self.assertTrue(body.get("enable_thinking"))

    def test_qwen3_no_enable_thinking_when_reasoning_none(self):
        model = QwenModel("qwen3-72b")
        _, body = model.to_request(_USER_MSG, _TEXT_OUTPUT)
        self.assertNotIn("enable_thinking", body)


# ---------------------------------------------------------------------------
# 3a. from_response() — OpenAI / xAI / Perplexity / DeepSeek / Kimi / Qwen
#     All use choices[0].message.content
# ---------------------------------------------------------------------------

_OPENAI_COMPAT_RESPONSE = {
    "choices": [{"message": {"content": "Hello from the model"}}]
}


class TestOpenAICompatFromResponse(unittest.TestCase):
    """Tests providers that share the OpenAI-compatible response shape."""

    def _assert_text(self, model):
        result = model.from_response(_OPENAI_COMPAT_RESPONSE, _TEXT_OUTPUT)
        self.assertEqual(result, "Hello from the model")

    def test_openai(self):
        self._assert_text(OpenAIModel("gpt-4o"))

    def test_xai(self):
        self._assert_text(XAIModel("grok-3"))

    def test_perplexity(self):
        self._assert_text(PerplexityModel("sonar"))

    def test_deepseek(self):
        self._assert_text(DeepSeekModel("deepseek-chat"))

    def test_kimi(self):
        self._assert_text(KimiModel("kimi-k2-0905-preview"))

    def test_qwen(self):
        self._assert_text(QwenModel("qwen-max"))

    def test_returns_string_type(self):
        result = OpenAIModel("gpt-4o").from_response(_OPENAI_COMPAT_RESPONSE, _TEXT_OUTPUT)
        self.assertIsInstance(result, str)


# ---------------------------------------------------------------------------
# 3b. from_response() — Anthropic
# ---------------------------------------------------------------------------

class TestAnthropicFromResponse(unittest.TestCase):

    def setUp(self):
        self.model = AnthropicModel("claude-opus-4-6")

    def test_extracts_first_text_block(self):
        response = {"content": [{"type": "text", "text": "Hi there"}]}
        result = self.model.from_response(response, _TEXT_OUTPUT)
        self.assertEqual(result, "Hi there")

    def test_returns_empty_string_when_no_content(self):
        result = self.model.from_response({"content": []}, _TEXT_OUTPUT)
        self.assertEqual(result, "")

    def test_skips_non_text_blocks(self):
        response = {
            "content": [
                {"type": "tool_use", "name": "search", "input": {}},
                {"type": "text", "text": "Final answer"},
            ]
        }
        result = self.model.from_response(response, _TEXT_OUTPUT)
        self.assertEqual(result, "Final answer")


# ---------------------------------------------------------------------------
# 3c. from_response() — Google AI
# ---------------------------------------------------------------------------

class TestGoogleFromResponse(unittest.TestCase):

    def setUp(self):
        self.model = GoogleAIModel("gemini-2.5-flash")

    def test_extracts_text_from_candidates(self):
        response = {
            "candidates": [
                {"content": {"parts": [{"text": "Gemini reply"}]}}
            ]
        }
        result = self.model.from_response(response, _TEXT_OUTPUT)
        self.assertEqual(result, "Gemini reply")

    def test_returns_empty_string_when_no_candidates(self):
        result = self.model.from_response({"candidates": []}, _TEXT_OUTPUT)
        self.assertEqual(result, "")

    def test_returns_string_type(self):
        response = {
            "candidates": [{"content": {"parts": [{"text": "Hi"}]}}]
        }
        result = self.model.from_response(response, _TEXT_OUTPUT)
        self.assertIsInstance(result, str)


# ---------------------------------------------------------------------------
# 4. API key handling
# ---------------------------------------------------------------------------

class TestModelApiKey(unittest.TestCase):

    def test_missing_key_raises_value_error(self):
        """Unset env var with no api_key= → ValueError."""
        # Use a fresh key name that is definitely unset
        import importlib
        saved = os.environ.pop("OPENAI_API_KEY", None)
        try:
            with self.assertRaises(ValueError):
                OpenAIModel("gpt-4o")
        finally:
            if saved:
                os.environ["OPENAI_API_KEY"] = saved
            else:
                os.environ["OPENAI_API_KEY"] = "test-openai-key"

    def test_explicit_api_key_bypasses_env(self):
        m = OpenAIModel("gpt-4o", api_key="explicit-key")
        self.assertEqual(m._api_key, "explicit-key")

    def test_invalid_reasoning_level_raises(self):
        with self.assertRaises(ValueError):
            OpenAIModel("gpt-4o", options={"reasoning": "extreme"})


# ---------------------------------------------------------------------------
# 5. Model options stored correctly
# ---------------------------------------------------------------------------

class TestModelOptions(unittest.TestCase):

    def test_temperature_override_stored(self):
        m = OpenAIModel("gpt-4o", options={"temperature": 0.3})
        self.assertAlmostEqual(m.temperature, 0.3)

    def test_max_tokens_override_stored(self):
        m = OpenAIModel("gpt-4o", options={"max_tokens": 512})
        self.assertEqual(m.max_tokens, 512)

    def test_reasoning_stored(self):
        m = OpenAIModel("gpt-4o", options={"reasoning": "high"})
        self.assertEqual(m.reasoning, "high")

    def test_default_temperature_applied(self):
        m = OpenAIModel("gpt-4o")
        self.assertEqual(m.temperature, OpenAIModel._DEFAULT_TEMPERATURE)

    def test_default_max_tokens_applied(self):
        m = AnthropicModel("claude-opus-4-6")
        self.assertEqual(m.max_tokens, AnthropicModel._DEFAULT_MAX_TOKENS)


# ---------------------------------------------------------------------------
# Live integration tests — skip unless the key env var is set
# ---------------------------------------------------------------------------

_OPENAI_KEY      = os.getenv("OPENAI_API_KEY")     if os.getenv("OPENAI_API_KEY",     "").startswith("sk-") else None
_ANTHROPIC_KEY   = os.getenv("ANTHROPIC_API_KEY")  if os.getenv("ANTHROPIC_API_KEY",  "").startswith("sk-ant") else None
_GOOGLE_KEY      = os.getenv("GOOGLE_AI_API_KEY")  if os.getenv("GOOGLE_AI_API_KEY",  "").startswith("AIza") else None
_XAI_KEY         = os.getenv("XAI_API_KEY")        if os.getenv("XAI_API_KEY",        "").startswith("xai-") else None
_PERPLEXITY_KEY  = os.getenv("PERPLEXITY_API_KEY") if os.getenv("PERPLEXITY_API_KEY", "").startswith("pplx-") else None
_DEEPSEEK_KEY    = os.getenv("DEEPSEEK_API_KEY")   if os.getenv("DEEPSEEK_API_KEY",   "").startswith("sk-") else None
_KIMI_KEY        = os.getenv("MOONSHOT_API_KEY")   if os.getenv("MOONSHOT_API_KEY",   "").startswith("sk-") else None
_QWEN_KEY        = os.getenv("DASHSCOPE_API_KEY")  if os.getenv("DASHSCOPE_API_KEY",  "").startswith("sk-") else None


def _live_output():
    return {"modalities": ["text"], "format": {"type": "text"}}


def _live_messages(prompt="Say 'ok' and nothing else."):
    return [{"role": "user", "parts": [{"type": "text", "text": prompt}]}]


def _make_skill_and_run(model_name, api_key_env):
    """Instantiate a Model and run a minimal Skill call end-to-end."""
    from skills import Skill
    skill = Skill(
        model         = Model(model_name),
        input         = {"messages": [{"role": "user", "parts": [{"type": "text", "text": "Say 'ok'"}]}]},
        output_format = "text",
    )
    return skill.run()


@unittest.skipUnless(_OPENAI_KEY, "Set a real OPENAI_API_KEY to run live tests")
class TestOpenAILive(unittest.TestCase):
    def test_text_round_trip(self):
        from skills import Skill
        skill = Skill(
            model         = Model("gpt-4o-mini"),
            input         = {"messages": [{"role": "user", "parts": [{"type": "text", "text": "Say 'ok'"}]}]},
            output_format = "text",
        )
        result = skill.run()
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


@unittest.skipUnless(_ANTHROPIC_KEY, "Set a real ANTHROPIC_API_KEY to run live tests")
class TestAnthropicLive(unittest.TestCase):
    def test_text_round_trip(self):
        from skills import Skill
        skill = Skill(
            model         = Model("claude-haiku-4-5-20251001"),
            input         = {"messages": [{"role": "user", "parts": [{"type": "text", "text": "Say 'ok'"}]}]},
            output_format = "text",
        )
        result = skill.run()
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


@unittest.skipUnless(_GOOGLE_KEY, "Set a real GOOGLE_AI_API_KEY to run live tests")
class TestGoogleLive(unittest.TestCase):
    def test_text_round_trip(self):
        from skills import Skill
        skill = Skill(
            model         = Model("gemini-2.5-flash"),
            input         = {"messages": [{"role": "user", "parts": [{"type": "text", "text": "Say 'ok'"}]}]},
            output_format = "text",
        )
        result = skill.run()
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


@unittest.skipUnless(_XAI_KEY, "Set a real XAI_API_KEY to run live tests")
class TestXAILive(unittest.TestCase):
    def test_text_round_trip(self):
        from skills import Skill
        skill = Skill(
            model         = Model("grok-3-fast"),
            input         = {"messages": [{"role": "user", "parts": [{"type": "text", "text": "Say 'ok'"}]}]},
            output_format = "text",
        )
        result = skill.run()
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


@unittest.skipUnless(_DEEPSEEK_KEY, "Set a real DEEPSEEK_API_KEY to run live tests")
class TestDeepSeekLive(unittest.TestCase):
    def test_text_round_trip(self):
        from skills import Skill
        skill = Skill(
            model         = Model("deepseek-chat"),
            input         = {"messages": [{"role": "user", "parts": [{"type": "text", "text": "Say 'ok'"}]}]},
            output_format = "text",
        )
        result = skill.run()
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


@unittest.skipUnless(_KIMI_KEY, "Set a real MOONSHOT_API_KEY to run live tests")
class TestKimiLive(unittest.TestCase):
    def test_text_round_trip(self):
        from skills import Skill
        skill = Skill(
            model         = Model("kimi-k2-0905-preview"),
            input         = {"messages": [{"role": "user", "parts": [{"type": "text", "text": "Say 'ok'"}]}]},
            output_format = "text",
        )
        result = skill.run()
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


@unittest.skipUnless(_QWEN_KEY, "Set a real DASHSCOPE_API_KEY to run live tests")
class TestQwenLive(unittest.TestCase):
    def test_text_round_trip(self):
        from skills import Skill
        skill = Skill(
            model         = Model("qwen-turbo"),
            input         = {"messages": [{"role": "user", "parts": [{"type": "text", "text": "Say 'ok'"}]}]},
            output_format = "text",
        )
        result = skill.run()
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


if __name__ == "__main__":
    unittest.main()
