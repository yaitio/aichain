"""
Multi-turn directed reasoning in a single Skill (1.5.0).

An ``assistant`` turn with no ``parts`` is a "generate here" marker: the model
produces that turn, the reply is appended to the running context, and later
turns see it. The last generation uses the skill's output format; intermediate
ones are plain text. Self-contained — a scripted FakeModel, no network.
"""

import json
import unittest

from yait_aichain.skills import Skill


class _FakeClient:
    def __init__(self, replies):
        self.replies, self.i, self.seen = replies, 0, []

    def _auth_headers(self):
        return {}

    def send(self, path, body, headers):
        out = self.replies[min(self.i, len(self.replies) - 1)]
        self.i += 1
        return json.dumps({"_c": out, "usage": {"input_tokens": 10, "output_tokens": 5}})


class _FakeModel:
    name = "fake"

    def __init__(self, replies):
        self.client = _FakeClient(replies)

    def to_request(self, messages, output):
        # record the per-call context (roles) so tests can assert it accumulates
        self.client.seen.append([m["role"] for m in messages])
        self._last_output = output
        return ("/x", {})

    def from_response(self, response, output):
        c = response["_c"]
        return c if output["format"]["type"] == "text" else {"value": c}


def _skill(model, msgs, **kw):
    return Skill(model=model, input={"messages": msgs}, **kw)


class TestMultiTurn(unittest.TestCase):

    def test_directed_sequence_accumulates_context(self):
        m = _FakeModel(["10 quotes", "cleaned", "translated"])
        sk = _skill(m, [
            {"role": "system",    "parts": ["Be concise."]},
            {"role": "user",      "parts": ["10 quotes about {topic}"]},
            {"role": "assistant"},
            {"role": "user",      "parts": ["drop 5 banal, add 5 new"]},
            {"role": "assistant"},
            {"role": "user",      "parts": ["translate to {language}"]},
        ])
        out = sk.run(variables={"topic": "grit", "language": "French"})
        self.assertEqual(out, "translated")
        self.assertEqual(m.client.i, 3)                 # 3 model calls
        self.assertEqual(sk.history, ["10 quotes", "cleaned", "translated"])
        # context grows each turn: 2 → 4 → 6 messages
        self.assertEqual([len(c) for c in m.client.seen], [2, 4, 6])
        # later calls actually see the spliced assistant replies
        self.assertEqual(m.client.seen[2],
                         ["system", "user", "assistant", "user", "assistant", "user"])

    def test_cumulative_usage(self):
        m = _FakeModel(["a", "b"])
        sk = _skill(m, [
            {"role": "user", "parts": ["x"]},
            {"role": "assistant"},
            {"role": "user", "parts": ["y"]},
        ])
        sk.run()
        self.assertEqual(sk.last_usage.total_tokens, 30)   # 2 calls * (10+5)

    def test_canned_assistant_is_not_a_call(self):
        m = _FakeModel(["refined", "translated"])
        sk = _skill(m, [
            {"role": "user",      "parts": ["10 quotes"]},
            {"role": "assistant", "parts": ["1..2..10"]},  # canned → NO model call
            {"role": "user",      "parts": ["drop banal"]},
            {"role": "assistant"},                          # call 1
            {"role": "user",      "parts": ["translate"]},  # final → call 2
        ])
        self.assertEqual(sk.run(), "translated")
        self.assertEqual(m.client.i, 2)

    def test_final_turn_uses_output_format(self):
        m = _FakeModel(["intermediate", "final"])
        sk = Skill(model=m,
                   input={"messages": [
                       {"role": "user", "parts": ["a"]},
                       {"role": "assistant"},
                       {"role": "user", "parts": ["b"]}]},
                   output={"format": {"type": "json"}})
        out = sk.run()
        self.assertEqual(out, {"value": "final"})          # final used json output


class TestBackwardCompatible(unittest.TestCase):

    def test_single_shot_unchanged(self):
        m = _FakeModel(["answer"])
        sk = _skill(m, [{"role": "user", "parts": ["{q}"]}])
        self.assertEqual(sk.run(variables={"q": "hi"}), "answer")
        self.assertEqual(m.client.i, 1)
        self.assertEqual(sk.history, ["answer"])

    def test_consecutive_users_one_call(self):
        m = _FakeModel(["answer"])
        sk = _skill(m, [{"role": "user", "parts": ["ctx"]},
                        {"role": "user", "parts": ["question"]}])
        sk.run()
        self.assertEqual(m.client.i, 1)                    # both users → one call


class TestStructureValidation(unittest.TestCase):

    def _bad(self, msgs, frag):
        with self.assertRaises(ValueError) as cm:
            _skill(_FakeModel(["x"]), msgs)
        self.assertIn(frag, str(cm.exception))

    def test_two_assistants_in_a_row(self):
        self._bad([{"role": "user", "parts": ["a"]},
                   {"role": "assistant"},
                   {"role": "assistant", "parts": ["b"]}], "two 'assistant'")

    def test_two_assistants_canned_then_marker(self):
        self._bad([{"role": "user", "parts": ["a"]},
                   {"role": "assistant", "parts": ["b"]},
                   {"role": "assistant"}], "two 'assistant'")

    def test_assistant_first(self):
        self._bad([{"role": "assistant"},
                   {"role": "user", "parts": ["a"]}], "cannot come first")

    def test_no_user(self):
        self._bad([{"role": "system", "parts": ["s"]}], "at least one 'user'")


if __name__ == "__main__":
    unittest.main()
