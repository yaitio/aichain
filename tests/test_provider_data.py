"""
tests.test_provider_data
========================

Consistency tests for the provider data files: the values in
models/providers/*.toml are now the single source of truth for provider
settings (defaults, env key, reasoning map, client family, pricing, caps).
These tests pin them so the data can't silently drift.

Pure: no network.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

for _k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_AI_API_KEY",
           "XAI_API_KEY", "PERPLEXITY_API_KEY", "DEEPSEEK_API_KEY",
           "MOONSHOT_API_KEY", "DASHSCOPE_API_KEY"):
    os.environ.setdefault(_k, "test-key")

from yait_aichain.models._data import PROVIDERS, provider_of


_PROVIDER_KEYS = {"openai", "anthropic", "google", "xai",
                  "perplexity", "kimi", "deepseek", "qwen",
                  "recraft", "bfl"}


class TestProviderDataConsistency(unittest.TestCase):

    def test_all_providers_present(self):
        self.assertEqual(set(PROVIDERS), _PROVIDER_KEYS)

    def test_defaults_present_and_typed(self):
        for k in PROVIDERS:
            d = PROVIDERS[k]["provider"]["defaults"]
            self.assertIsInstance(d["temperature"], (int, float), k)
            self.assertIsInstance(d["max_tokens"], int, k)

    def test_defaults_pinned_values(self):
        # Pin a representative spread so the data can't silently drift.
        self.assertEqual(PROVIDERS["openai"]["provider"]["defaults"]["max_tokens"], 16384)
        self.assertEqual(PROVIDERS["anthropic"]["provider"]["defaults"]["max_tokens"], 8192)
        self.assertEqual(PROVIDERS["deepseek"]["provider"]["defaults"]["temperature"], 0.0)
        self.assertEqual(PROVIDERS["qwen"]["provider"]["defaults"]["top_p"], 0.8)

    def test_env_key_pinned(self):
        expected = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
                    "google": "GOOGLE_AI_API_KEY", "xai": "XAI_API_KEY",
                    "perplexity": "PERPLEXITY_API_KEY", "deepseek": "DEEPSEEK_API_KEY",
                    "kimi": "MOONSHOT_API_KEY", "qwen": "DASHSCOPE_API_KEY"}
        for k, env in expected.items():
            self.assertEqual(PROVIDERS[k]["provider"]["env_key"], env, k)

    def test_client_family_assignment(self):
        # openai-compatible providers route to the 'openai' client family
        for k in ("openai", "xai", "kimi", "deepseek"):
            self.assertEqual(PROVIDERS[k]["provider"]["client"], "openai", k)
        self.assertEqual(PROVIDERS["perplexity"]["provider"]["client"], "perplexity")
        self.assertEqual(PROVIDERS["qwen"]["provider"]["client"], "qwen")
        self.assertEqual(PROVIDERS["anthropic"]["provider"]["client"], "anthropic")
        self.assertEqual(PROVIDERS["google"]["provider"]["client"], "google")

    def test_max_tokens_field(self):
        self.assertEqual(PROVIDERS["openai"]["provider"]["max_tokens_field"],
                         "max_completion_tokens")
        self.assertEqual(PROVIDERS["kimi"]["provider"]["max_tokens_field"],
                         "max_tokens")
        self.assertEqual(PROVIDERS["deepseek"]["provider"]["max_tokens_field"],
                         "max_tokens")

    def test_prices_well_formed(self):
        # Every priced model carries numeric input/output USD-per-1M rates.
        priced = 0
        for prov, data in PROVIDERS.items():
            for name, mdl in data.get("models", {}).items():
                if "price" in mdl:
                    priced += 1
                    self.assertIsInstance(mdl["price"]["input"], (int, float), name)
                    self.assertIsInstance(mdl["price"]["output"], (int, float), name)
        self.assertGreater(priced, 0)

    def test_caps_within_vocabulary(self):
        from yait_aichain.models._base import TASKS
        for prov, data in PROVIDERS.items():
            for name, mdl in data.get("models", {}).items():
                caps = mdl.get("caps", [])
                self.assertTrue(caps, f"{prov}/{name} has no caps")
                for c in caps:
                    self.assertIn(c, TASKS, f"{prov}/{name}: {c}")

    def test_provider_of_resolves(self):
        self.assertEqual(provider_of("gpt-4o"), "openai")
        self.assertEqual(provider_of("claude-sonnet-4-6"), "anthropic")
        self.assertEqual(provider_of("deepseek-chat"), "deepseek")
        self.assertIsNone(provider_of("nonexistent-model-xyz"))


if __name__ == "__main__":
    unittest.main()
