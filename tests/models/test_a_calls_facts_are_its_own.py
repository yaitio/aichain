"""
What a call learned travels with the call, not on the shared Model.

`last_stream_usage`, `last_stream_result` and `last_adaptations` were written
to the Model and read back after the network round: with one Model shared by
a Pool's threads or by tenants, stream A read stream B's usage (1,000 tokens
became 8, and `max_cost` undercounted), and adaptations were credited to
another request. A stream returns its outcome now, a build returns its
adaptations, and the attributes are mirrors the library does not read.
"""

import json
import unittest
from unittest.mock import MagicMock

from yait_aichain import Model, Skill
from yait_aichain.models._base import StreamOutcome


def _model():
    m = Model("gpt-4o", api_key="k")
    m.client._auth_headers = MagicMock(return_value={})
    return m


class TestAStreamReadsItsOwnUsage(unittest.TestCase):

    def test_another_stream_writing_the_model_mid_flight_changes_nothing(self):
        m = _model()

        def sse(path, body, headers, *a, **k):
            yield {"choices": [{"delta": {"content": "hi"}}]}
            m.last_stream_usage = {"usage": {"prompt_tokens": 1, "completion_tokens": 1}}   # someone else
            m.last_adaptations = ["not mine"]
            yield {"choices": [], "usage": {"prompt_tokens": 900, "completion_tokens": 100}}
        m.client._post_sse = MagicMock(side_effect=sse)
        skill = Skill(m, prompt="hi")
        self.assertEqual("".join(skill.stream()), "hi")
        self.assertEqual(skill.last_usage.total_tokens, 1000)
        self.assertNotIn("not mine", skill.last_adaptations)

    def test_the_outcome_is_returned(self):
        m = _model()
        m.client._post_sse = MagicMock(return_value=iter([
            {"choices": [{"delta": {"content": "ok"}}]},
            {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2}}]))
        stream = m.stream([{"role": "user", "parts": [{"type": "text", "text": "hi"}]}],
                          {"modalities": ["text"], "format": {"type": "text"}})
        pieces = []
        while True:
            try:
                pieces.append(next(stream))
            except StopIteration as done:
                outcome = done.value
                break
        self.assertIsInstance(outcome, StreamOutcome)
        self.assertEqual((pieces, outcome.usage["usage"]["completion_tokens"]), (["ok"], 2))


class TestACallReadsItsOwnAdaptations(unittest.TestCase):

    def test_adaptations_come_back_with_the_build(self):
        m = Model("gpt-5.4-mini", api_key="k", options={"temperature": 0.2})
        m.client._auth_headers = MagicMock(return_value={})

        def send(path, body, headers):
            m.last_adaptations = ["not mine"]                            # someone else's build
            return json.dumps({"output": [{"type": "message", "content": [
                {"type": "output_text", "text": "ok"}]}], "usage": {}}).encode()
        m.client.send = MagicMock(side_effect=send)
        skill = Skill(m, prompt="hi")
        skill.run()
        self.assertNotIn("not mine", skill.last_adaptations)
        self.assertIn("temperature", {a.option for a in skill.last_adaptations})


if __name__ == "__main__":
    unittest.main()
