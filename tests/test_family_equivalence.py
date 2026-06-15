"""
tests.test_family_equivalence
=============================

R3 proof: the new clients/_families/* clients produce byte-identical output to
the old per-provider Model classes — build_request == to_request and
parse_response == from_response — across text, json and all modalities, for
every provider. This is the contract that lets R4 switch Model over to the
clients and R5 delete the old classes safely.

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

from yait_aichain.models import Model
from yait_aichain.models._data import PROVIDERS
from yait_aichain.clients._families.openai import OpenAIClient
from yait_aichain.clients._families.anthropic import AnthropicClient
from yait_aichain.clients._families.google import GoogleClient
from yait_aichain.clients._families.perplexity import PerplexityClient
from yait_aichain.clients._families.qwen import QwenClient


def _make_client(prov):
    data = PROVIDERS[prov]
    ct = data["provider"]["client"]
    return {
        "openai": OpenAIClient, "anthropic": AnthropicClient,
        "google": GoogleClient, "perplexity": PerplexityClient,
        "qwen": QwenClient,
    }[ct]("k", data=data)


def _params(m):
    return {"name": m.name, "temperature": m.temperature,
            "max_tokens": m.max_tokens, "top_p": m.top_p,
            "top_k": m.top_k, "reasoning": m.reasoning}


_MSG = [{"role": "system", "parts": [{"type": "text", "text": "Be helpful."}]},
        {"role": "user",   "parts": [{"type": "text", "text": "Hello"}]}]
_OUT_T = {"modalities": ["text"], "format": {"type": "text"}}
_OUT_J = {"modalities": ["text"], "format": {"type": "json"}}

_SAMPLE = {"openai": "gpt-4o", "anthropic": "claude-sonnet-4-6",
           "google": "gemini-2.5-flash", "xai": "grok-3",
           "perplexity": "sonar-pro", "kimi": "kimi-k2-0905-preview",
           "deepseek": "deepseek-chat", "qwen": "qwen-max"}


class TestFamilyEquivalence(unittest.TestCase):

    def _equiv_build(self, name, prov, msg, out, options=None):
        m = Model(name, api_key="k", options=options)
        client = _make_client(prov)
        self.assertEqual(
            client.build_request(msg, out, _params(m)),
            m.to_request(msg, out),
            f"build mismatch {prov}/{name} {out['format']['type']}")

    def test_build_text_and_json(self):
        for prov, name in _SAMPLE.items():
            self._equiv_build(name, prov, _MSG, _OUT_T)
            self._equiv_build(name, prov, _MSG, _OUT_J)

    def test_build_reasoning(self):
        for prov, name in _SAMPLE.items():
            self._equiv_build(name, prov, _MSG, _OUT_T, options={"reasoning": "high"})
            self._equiv_build(name, prov, _MSG, _OUT_T, options={"reasoning": "low"})

    def test_build_vision(self):
        msg = [{"role": "user", "parts": [
            {"type": "text", "text": "describe"},
            {"type": "image",
             "source": {"kind": "base64", "data": "AAAA", "mime": "image/png"}}]}]
        for prov, name in _SAMPLE.items():
            self._equiv_build(name, prov, msg, _OUT_T)

    def test_build_image_generation(self):
        prompt = [{"role": "user", "parts": [{"type": "text", "text": "a cat"}]}]
        img = {"modalities": ["image"], "format": {"type": "image"}}
        for name, prov in [("gpt-image-1", "openai"),
                           ("grok-imagine-image-pro", "xai"),
                           ("wan2.2-t2i-flash", "qwen"),
                           ("gemini-3.1-flash-image", "google")]:
            self._equiv_build(name, prov, prompt, img)

    def test_parse_text_and_json(self):
        responses = {
            "openai":     {"choices": [{"message": {"content": '{"a":1}'}}]},
            "anthropic":  {"content": [{"type": "text", "text": '{"a":1}'}]},
            "google":     {"candidates": [{"content": {"parts": [{"text": '{"a":1}'}]}}]},
        }
        for prov, resp in responses.items():
            m = Model(_SAMPLE[prov], api_key="k")
            client = _make_client(prov)
            for out in (_OUT_T, _OUT_J):
                self.assertEqual(
                    client.parse_response(resp, out),
                    m.from_response(resp, out),
                    f"parse mismatch {prov} {out['format']['type']}")


if __name__ == "__main__":
    unittest.main()
