"""
tests.clients.test_local_provider
=================================

The local provider: self-hosted OpenAI-compatible servers, on the wire.

Everything here failed or misled before the provider existed. An HF-style id
raised "Cannot detect provider"; a missing key raised at construction even
though the server runs open; and the base URL every local server's docs quote
— the one ending in /v1 — produced /v1/v1/chat/completions and an unhelpful
404. Each of those is pinned as behaviour now, not as a bug report.

Pure: no network, no server. What a live server would add is only whether it
serves the named model — which is its answer to give, not ours.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from yait_aichain.models import Model
from yait_aichain.models._usage import Usage, estimate_cost

_OUT = {"modalities": ["text"], "format": {"type": "text"}}
_MSG = [{"role": "user", "parts": [{"type": "text", "text": "hi"}]}]


def request_of(model):
    path, body = model.to_request(_MSG, _OUT)
    return model.client._base_url + path, body


class TestRouting(unittest.TestCase):

    def test_hf_id_keeps_its_slashes(self):
        """
        Only the first slash routes; the rest belong to the model. Hugging
        Face ids are org/name, so eating the second slash would corrupt
        every real id there is.
        """
        m = Model("local/meta-llama/Llama-3.3-70B-Instruct")
        self.assertEqual(m._provider, "local")
        self.assertEqual(m.name, "meta-llama/Llama-3.3-70B-Instruct")
        _, body = request_of(m)
        self.assertEqual(body["model"], "meta-llama/Llama-3.3-70B-Instruct")

    def test_no_registry_gate(self):
        """The server decides what it serves; any name constructs."""
        Model("local/anything-at-all")

    def test_cost_is_none_not_a_guess(self):
        """
        Local models have no price per token — their economics are GPU-hours
        over throughput, which depends on the card. None is the honest
        answer; a number would be invented.
        """
        self.assertIsNone(estimate_cost(Usage(1000, 1000), "meta-llama/Llama-3.3-70B-Instruct"))


class TestKeyless(unittest.TestCase):

    def test_constructs_without_any_key(self):
        m = Model("local/qwen3-30b")
        self.assertNotIn("Authorization", m.client._auth_headers())

    def test_no_key_means_no_header_at_all(self):
        """
        "Bearer " with nothing after it is not neutral: some servers 401 on
        a malformed header where they would accept no header.
        """
        headers = Model("local/qwen3-30b").client._auth_headers()
        for v in headers.values():
            self.assertNotIn("Bearer", v)

    def test_a_provided_key_is_still_sent(self):
        """Started with --api-key, the server expects the token."""
        m = Model("local/qwen3-30b", api_key="secret")
        self.assertEqual(m.client._auth_headers()["Authorization"], "Bearer secret")

    def test_env_key_is_honoured(self):
        os.environ["LOCAL_API_KEY"] = "from-env"
        try:
            m = Model("local/qwen3-30b")
            self.assertEqual(m.client._auth_headers()["Authorization"], "Bearer from-env")
        finally:
            del os.environ["LOCAL_API_KEY"]

    def test_other_providers_still_require_a_key(self):
        """The gate is lifted for auth="none" only, not weakened globally."""
        env = os.environ.pop("OPENAI_API_KEY", None)
        try:
            with self.assertRaises(ValueError):
                Model("gpt-4o")
        finally:
            if env is not None:
                os.environ["OPENAI_API_KEY"] = env


class TestBaseURL(unittest.TestCase):
    """
    Every local server's docs quote the base URL with /v1 — Ollama's
    :11434/v1, LM Studio's :1234/v1 — while our paths already start with
    /v1. Both spellings must mean the same server.
    """

    def test_documented_v1_urls_do_not_double(self):
        for url in ("http://localhost:11434/v1",
                    "http://localhost:1234/v1",
                    "http://localhost:1234/v1/"):
            with self.subTest(url=url):
                m = Model("local/qwen3-30b", client_options={"url": url})
                full, _ = request_of(m)
                self.assertNotIn("/v1/v1/", full)
                self.assertTrue(full.endswith("/v1/chat/completions"))

    def test_bare_host_gets_the_path(self):
        m = Model("local/qwen3-30b", client_options={"url": "http://gpu-box:8000"})
        full, _ = request_of(m)
        self.assertEqual(full, "http://gpu-box:8000/v1/chat/completions")

    def test_env_var_points_at_a_remote_box(self):
        os.environ["LOCAL_BASE_URL"] = "http://gpu:9000"
        try:
            m = Model("local/qwen3-30b")
            self.assertTrue(request_of(m)[0].startswith("http://gpu:9000/"))
        finally:
            del os.environ["LOCAL_BASE_URL"]

    def test_explicit_url_beats_the_env_var(self):
        os.environ["LOCAL_BASE_URL"] = "http://wrong:9000"
        try:
            m = Model("local/qwen3-30b", client_options={"url": "http://right:8000"})
            self.assertTrue(request_of(m)[0].startswith("http://right:8000/"))
        finally:
            del os.environ["LOCAL_BASE_URL"]

    def test_qwen_compatible_mode_path_is_untouched(self):
        """
        The /v1 rule must not fire where the path does not start with /v1 —
        qwen's compatible-mode path is the counterexample in the tree.
        """
        os.environ.setdefault("DASHSCOPE_API_KEY", "test-key")
        m = Model("qwen-max", api_key="k")
        full, _ = request_of(m)
        self.assertIn("/compatible-mode/v1/chat/completions", full)


if __name__ == "__main__":
    unittest.main()
