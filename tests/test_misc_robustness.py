"""
Small accounting/parsing robustness fixes (1.3.4 #50):
  - Agent._extract_tokens tolerates usage: null / non-numeric.
  - DeepSeek reasoner gate matches the model name exactly, not as a substring.
  - Google embeddings accept GOOGLE_API_KEY as well as GOOGLE_AI_API_KEY.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from agent import Agent
from models import Model
from clients._families.openai import _is_deepseek_reasoner


class TestExtractTokensRobust(unittest.TestCase):

    def _agent(self):
        return Agent(orchestrator=Model("gpt-4o", api_key="k"))

    def test_usage_null(self):
        self.assertEqual(self._agent()._extract_tokens({"usage": None}), 0)

    def test_usage_not_a_dict(self):
        self.assertEqual(self._agent()._extract_tokens({"usage": []}), 0)

    def test_usage_non_numeric(self):
        self.assertEqual(
            self._agent()._extract_tokens({"usage": {"prompt_tokens": "x"}}), 0)

    def test_usage_normal(self):
        self.assertEqual(
            self._agent()._extract_tokens(
                {"usage": {"input_tokens": 10, "output_tokens": 5}}), 15)


class TestDeepSeekReasonerGate(unittest.TestCase):

    def test_exact_and_prefix_match_only(self):
        self.assertTrue(_is_deepseek_reasoner("deepseek-reasoner"))
        self.assertFalse(_is_deepseek_reasoner("my-custom-reasoner"))   # not a substring hit
        self.assertFalse(_is_deepseek_reasoner("deepseek-chat"))


class TestGoogleEmbeddingEnv(unittest.TestCase):

    def test_accepts_google_api_key(self):
        from tools.embedding._google import EmbeddingGoogle
        os.environ.pop("GOOGLE_AI_API_KEY", None)
        os.environ["GOOGLE_API_KEY"] = "g-key"
        try:
            emb = EmbeddingGoogle("gemini-embedding-001")   # must not raise
            self.assertEqual(emb._api_key, "g-key")
        finally:
            os.environ.pop("GOOGLE_API_KEY", None)


if __name__ == "__main__":
    unittest.main()
