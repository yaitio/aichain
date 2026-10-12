"""
A reply cut off at the token ceiling is not executed, and not a success.

The stop reason was read only when parsing JSON. A tool call cut at
`max_tokens` ran with `{}` or half its arguments — the tool worked on its
defaults and wrote nothing — and an empty answer, common from reasoning
models that spent the budget thinking, ended a run with `success=True`.
"""

import unittest
import warnings

from yait_aichain import Model, TruncatedResponseError

OUT = {"modalities": ["text"], "format": {"type": "text"}}
CALL = {"id": "c1", "type": "function", "function": {"name": "save", "arguments": '{"path": "a'}}


def _parse(name, response):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Model(name, api_key="k").from_response(response, OUT)


class TestEveryFamily(unittest.TestCase):

    def test_chat_completions(self):
        for message in ({"content": None, "tool_calls": [CALL]}, {"content": ""}):
            with self.subTest(message=message):
                with self.assertRaisesRegex(TruncatedResponseError, "cut off at 512"):
                    _parse("deepseek-chat", {"choices": [{"message": message, "finish_reason": "length"}],
                                             "usage": {"completion_tokens": 512}})

    def test_the_responses_api(self):
        with self.assertRaises(TruncatedResponseError):
            _parse("gpt-5.4-mini", {"status": "incomplete",
                                    "incomplete_details": {"reason": "max_output_tokens"},
                                    "output": [{"type": "reasoning"}]})
        with self.assertRaisesRegex(ValueError, "failed"):
            _parse("gpt-5.4-mini", {"status": "failed", "error": {"message": "server"}, "output": []})

    def test_anthropic(self):
        with self.assertRaises(TruncatedResponseError):
            _parse("claude-sonnet-4-5", {"stop_reason": "max_tokens", "content": [
                {"type": "tool_use", "id": "t", "name": "save", "input": {}}]})

    def test_gemini(self):
        with self.assertRaises(TruncatedResponseError):
            _parse("gemini-2.5-flash", {"candidates": [{"finishReason": "MAX_TOKENS",
                                                        "content": {"parts": []}}]})
        with self.assertRaisesRegex(ValueError, "SAFETY"):
            _parse("gemini-2.5-flash", {"candidates": [{"finishReason": "SAFETY", "content": {"parts": []}}]})

    def test_prose_cut_short_is_still_returned(self):
        self.assertEqual(_parse("deepseek-chat", {"choices": [{"message": {"content": "Partial"},
                                                               "finish_reason": "length"}]}), "Partial")


class TestGeminiLeavesRoomToAnswer(unittest.TestCase):

    def test_a_thinking_budget_above_the_limit_raises_the_limit(self):
        m = Model("gemini-2.5-flash", api_key="k", options={"reasoning": "high", "max_tokens": 1000})
        _, body = m.to_request([{"role": "user", "parts": [{"type": "text", "text": "hi"}]}], OUT)
        cfg = body["generationConfig"]
        self.assertGreater(cfg["maxOutputTokens"], cfg["thinkingConfig"]["thinkingBudget"])


class TestAnAgentSaysItWasCutOff(unittest.TestCase):

    def test_stopped_by_truncated(self):
        import json
        from unittest.mock import MagicMock
        from yait_aichain import Agent
        m = Model("deepseek-chat", api_key="k")
        m.client._post = MagicMock(return_value=json.dumps(
            {"choices": [{"message": {"content": None, "tool_calls": [CALL]}, "finish_reason": "length"}],
             "usage": {"prompt_tokens": 1, "completion_tokens": 512, "total_tokens": 513}}).encode())
        m.client._auth_headers = MagicMock(return_value={})
        result = Agent(m).run("go")
        self.assertEqual((result.success, result.stopped_by), (False, "truncated"))


if __name__ == "__main__":
    unittest.main()
