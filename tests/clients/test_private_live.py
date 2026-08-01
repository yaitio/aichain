"""
Live tests for the `private` provider — against a real OpenAI-compatible
server (vLLM, vllm-metal, Ollama, LM Studio, llama.cpp).

Skipped unless a server answers. Nothing here needs an API key, a cloud
account or a cent: the whole point of this provider is that the test costs
nothing but the electricity of the machine it runs on.

What the offline suite already pins — routing, auth headers, URL forms,
request body — is not repeated. This file asserts only the things a mock
cannot: that a real server accepts what we build, that a keyless request is
actually accepted rather than merely well-formed, and that the counters we
read back are populated the way the cost accounting assumes.

    # start any of these first
    vllm serve Qwen/Qwen3-0.6B                     # :8000  (vLLM / vllm-metal)
    ollama run llama3.3                            # :11434
    # then
    python3 -m unittest tests.clients.test_private_live -v

Override the endpoint and model when they differ from the defaults:

    PRIVATE_BASE_URL=http://localhost:11434 \
    PRIVATE_TEST_MODEL=llama3.3 \
    python3 -m unittest tests.clients.test_private_live -v
"""

import json
import os
import sys
import unittest
import urllib.error
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from yait_aichain.models import Model
from yait_aichain.skills import Skill

BASE_URL = os.getenv("PRIVATE_BASE_URL", "http://localhost:8000").rstrip("/")
_OUT = {"modalities": ["text"], "format": {"type": "text"}}


def _served_models() -> "list[str]":
    """Model ids the server reports, or [] if nothing is listening."""
    url = BASE_URL if BASE_URL.endswith("/v1") else BASE_URL + "/v1"
    try:
        with urllib.request.urlopen(url + "/models", timeout=3) as r:
            return [m["id"] for m in json.loads(r.read()).get("data", [])]
    except (urllib.error.URLError, OSError, ValueError, KeyError):
        return []


_SERVED = _served_models()
# Prefer an explicitly named model, else whatever the server is serving. A
# server with exactly one model loaded is the common case, and asking it what
# that is beats hard-coding a name this suite cannot know.
_MODEL = os.getenv("PRIVATE_TEST_MODEL") or (_SERVED[0] if _SERVED else None)

_skip = unittest.skipIf(
    not _SERVED,
    f"no OpenAI-compatible server answering at {BASE_URL} "
    f"(start one, or set PRIVATE_BASE_URL)",
)


@_skip
class TestPrivateLive(unittest.TestCase):

    def setUp(self):
        self.model = Model(f"private/{_MODEL}",
                           options={"max_tokens": 64, "temperature": 0.0},
                           client_options={"url": BASE_URL})

    # ── the call itself ──────────────────────────────────────────────

    def test_a_keyless_call_is_accepted(self):
        """
        Offline we assert that no Authorization header is *built*. Only a real
        server can tell us it is *accepted* — that omitting the header is what
        an open server wants, rather than something it rejects.
        """
        self.assertNotIn("Authorization", self.model.client._auth_headers())
        skill = Skill(model=self.model, input={"messages": [
            {"role": "user", "parts": ["Reply with the single word: ok"]}]})
        self.assertTrue(str(skill.run()).strip())

    def test_the_answer_is_text_the_caller_can_use(self):
        skill = Skill(model=self.model, input={"messages": [
            {"role": "user", "parts": ["Name the capital of France. One word."]}]})
        answer = str(skill.run())
        self.assertIsInstance(answer, str)
        self.assertTrue(answer.strip(), "empty answer from the server")

    # ── the numbers we read back ─────────────────────────────────────

    def test_token_counts_arrive(self):
        """
        The docs promise counts still flow so a caller can compute their own
        rate from hardware cost. That promise is only worth as much as the
        server's usage block, which no fixture can vouch for.
        """
        skill = Skill(model=self.model, input={"messages": [
            {"role": "user", "parts": ["Count to three."]}]})
        skill.run()
        usage = skill.last_usage
        self.assertGreater(usage.input_tokens, 0)
        self.assertGreater(usage.output_tokens, 0)

    def test_cost_is_none_against_a_real_response(self):
        """
        `None` is the documented answer, not a gap — a hosted model has no
        price per token. Pinned here because a response carrying a usage
        block is exactly where an invented number would sneak in.
        """
        skill = Skill(model=self.model, input={"messages": [
            {"role": "user", "parts": ["Say hi."]}]})
        skill.run()
        self.assertIsNone(skill.last_usage.cost)

    # ── the URL trap, end to end ─────────────────────────────────────

    def test_both_url_spellings_reach_the_same_server(self):
        """
        The /v1-doubling bug: every server's docs quote the base URL with
        /v1, our paths already start with /v1. Offline this is a string
        assertion; here both spellings have to actually answer.
        """
        for url in (BASE_URL.removesuffix("/v1"), BASE_URL.removesuffix("/v1") + "/v1"):
            with self.subTest(url=url):
                m = Model(f"private/{_MODEL}", options={"max_tokens": 16},
                          client_options={"url": url})
                skill = Skill(model=m, input={"messages": [
                    {"role": "user", "parts": ["Say ok."]}]})
                self.assertTrue(str(skill.run()).strip())

    # ── discovery ────────────────────────────────────────────────────

    def test_the_client_can_list_what_the_server_serves(self):
        """`list_models()` is how a caller finds the name to pass."""
        self.assertIn(_MODEL, self.model.client.list_models())


if __name__ == "__main__":
    unittest.main()
