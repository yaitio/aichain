"""
Streaming, and the two ways it pretends to work.

Both are what this suite is for. A reader watching text appear cannot tell a
real stream from a buffered one, and a caller who asked to stream and got the
whole answer in one piece at the end has something that *looks* like it
worked — the same class of silent loss the option layer spent 2.3.0 removing.
So: the transport is tested against a wire that arrives in fragments, and the
provider that cannot stream is tested for saying so.
"""

import json
import unittest
from unittest import mock

from yait_aichain import Model, Skill, UnsupportedOption


def _sse(*events, chunks=None):
    """A fake urllib3 response whose body arrives in pieces.

    The fragmentation is the point: a single-chunk fake passes even when the
    reader has no buffering at all, and a real provider splits an event
    across TCP reads regularly.
    """
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
    raw = body.encode()
    pieces = chunks if chunks is not None else [raw[i:i + 7]
                                                for i in range(0, len(raw), 7)]

    class _Resp:
        status = 200
        headers = {}

        def stream(self, amt=None, decode_content=True):
            yield from pieces

        def read(self):
            return raw

        def release_conn(self):
            pass

    return _Resp()


def _delta(text=None, usage=None):
    event = {"choices": [{"delta": {"content": text} if text else {}}]}
    if usage:
        event["usage"] = usage
    return event


def _skill(model, output=None):
    return Skill(model=model,
                 input={"messages": [{"role": "user", "parts": [
                     {"type": "text", "text": "hi"}]}]},
                 output=output or {"format": {"type": "text"}})


class TestTheTransportReassembles(unittest.TestCase):

    def _stream(self, resp, name="gpt-4o", output=None):
        m = Skill(model=Model(name, api_key="k"),
                  input={"messages": [{"role": "user", "parts": [
                      {"type": "text", "text": "hi"}]}]},
                  output=output or {"format": {"type": "text"}})
        with mock.patch("urllib3.PoolManager.request", return_value=resp):
            return list(m.stream()), m

    def test_an_event_split_across_reads_arrives_whole(self):
        """Seven bytes at a time: every event boundary lands mid-JSON."""
        pieces, _ = self._stream(_sse(_delta("Hel"), _delta("lo"), _delta("!")))
        self.assertEqual("".join(pieces), "Hello!")

    def test_the_done_sentinel_is_not_decoded_as_json(self):
        """`[DONE]` through a JSON decoder is the commonest way an SSE reader
        ends with an exception instead of a return."""
        pieces, _ = self._stream(_sse(_delta("x")))
        self.assertEqual(pieces, ["x"])

    def test_an_event_carrying_no_text_yields_nothing(self):
        pieces, _ = self._stream(_sse(_delta(), _delta("only this"), _delta()))
        self.assertEqual(pieces, ["only this"])

    def test_usage_from_the_final_event_reaches_the_skill(self):
        usage = {"prompt_tokens": 11, "completion_tokens": 3,
                 "total_tokens": 14}
        _, skill = self._stream(_sse(_delta("hi"), _delta(usage=usage)))
        self.assertEqual(skill.last_usage.total_tokens, 14)

    def test_no_usage_reported_means_no_usage_invented(self):
        _, skill = self._stream(_sse(_delta("hi")))
        self.assertIsNone(skill.last_usage)

    def test_the_whole_answer_is_assembled_as_well_as_yielded(self):
        _, skill = self._stream(_sse(_delta("a"), _delta("b")))
        self.assertEqual(skill.last_result, "ab")

    def test_json_output_is_parsed_at_the_end(self):
        events = _sse(_delta('{"n":'), _delta(" 1}"))
        _, skill = self._stream(events, output={"format": {"type": "json"}})
        self.assertEqual(skill.last_result, {"n": 1})

    def test_unparseable_json_keeps_the_text_rather_than_none(self):
        events = _sse(_delta("not json"))
        _, skill = self._stream(events, output={"format": {"type": "json"}})
        self.assertEqual(skill.last_result, "not json")


class TestTheRequestSaysStream(unittest.TestCase):

    def test_the_body_asks_for_usage_too(self):
        """Without stream_options a streamed call reports no tokens at all,
        and a run that cannot price itself is not shippable."""
        m = Model("gpt-4o", api_key="k")
        _, body = m.client.build_stream_request(
            [{"role": "user", "parts": [{"type": "text", "text": "hi"}]}],
            {"format": {"type": "text"}}, m._params())
        self.assertTrue(body["stream"])
        self.assertEqual(body["stream_options"], {"include_usage": True})


class TestAProviderThatCannotStreamSaysSo(unittest.TestCase):
    """The important half. One chunk at the end is indistinguishable from a
    working stream unless the library says which it was."""

    def _run(self, name, **kw):
        model = Model(name, api_key="k", **kw)
        answer = {"choices": [{"message": {"content": "whole thing"}}],
                  "usage": {"prompt_tokens": 2, "completion_tokens": 2,
                            "total_tokens": 4}}
        with mock.patch.object(type(model.client), "send",
                               return_value=json.dumps(answer).encode()):
            skill = _skill(model)
            pieces = list(skill.stream())
        return pieces, skill, model

    def test_the_answer_still_arrives(self):
        """Providers stay swappable: asking a non-streaming provider to
        stream is not an error, it is a degraded delivery."""
        pieces, _, _ = self._run("gpt-5.5")
        self.assertEqual(pieces, ["whole thing"])

    def test_and_it_is_on_the_record(self):
        _, skill, _ = self._run("gpt-5.5")
        spoken = {(a.option, a.kind) for a in skill.last_adaptations}
        self.assertIn(("stream", "declined"), spoken)

    def test_a_caller_who_needs_a_real_stream_can_be_stopped(self):
        with self.assertRaises(UnsupportedOption):
            self._run("gpt-5.5", on_unsupported="raise")

    def test_a_streaming_model_records_nothing(self):
        resp = _sse(_delta("hi"))
        skill = _skill(Model("gpt-4o", api_key="k"))
        with mock.patch("urllib3.PoolManager.request", return_value=resp):
            list(skill.stream())
        self.assertEqual(skill.last_adaptations, [])




# ── One promise, three wires ────────────────────────────────────────────────
#
# The library's claim is that a stream means the same thing whichever provider
# is behind it. That claim breaks per family — each writes its own event
# shapes — so it has to be tested per family, from one set of assertions.
# Testing each family with the test its author wrote leaves the promise itself
# unchecked, which is how `stream_usage` came to return a bare block that two
# families understood and the third silently read as zero.

FAMILIES = {
    "openai": {
        "model": "gpt-4o",
        "text":  lambda t: {"choices": [{"delta": {"content": t}}]},
        "empty": {"choices": [{"delta": {"role": "assistant"}}]},
        "usage": [{"choices": [], "usage": {"prompt_tokens": 11,
                                            "completion_tokens": 3,
                                            "total_tokens": 14}}],
    },
    "anthropic": {
        "model": "claude-haiku-4-5-20251001",
        "text":  lambda t: {"type": "content_block_delta",
                            "delta": {"type": "text_delta", "text": t}},
        "empty": {"type": "content_block_start",
                  "content_block": {"type": "text", "text": ""}},
        # Two events, on purpose: the input count and the output count arrive
        # separately and the total is only right if both are kept.
        "usage": [{"type": "message_start",
                   "message": {"usage": {"input_tokens": 11}}},
                  {"type": "message_delta", "usage": {"output_tokens": 3}}],
    },
    "google": {
        "model": "gemini-2.5-flash",
        "text":  lambda t: {"candidates": [
            {"content": {"parts": [{"text": t}]}}]},
        "empty": {"candidates": [{"content": {"parts": []}}]},
        "usage": [{"usageMetadata": {"promptTokenCount": 11,
                                     "candidatesTokenCount": 3,
                                     "totalTokenCount": 14}}],
    },
}


class TestEveryFamilyKeepsTheSamePromise(unittest.TestCase):

    def _run(self, family, events):
        spec = FAMILIES[family]
        skill = _skill(Model(spec["model"], api_key="k"))
        with mock.patch("urllib3.PoolManager.request",
                        return_value=_sse(*events)):
            return list(skill.stream()), skill

    def test_text_reassembles(self):
        for family, spec in FAMILIES.items():
            with self.subTest(family=family):
                pieces, skill = self._run(
                    family, [spec["text"](t) for t in ("Hel", "lo", "!")])
                self.assertEqual("".join(pieces), "Hello!")
                self.assertEqual(skill.last_result, "Hello!")

    def test_an_event_with_no_text_yields_nothing(self):
        for family, spec in FAMILIES.items():
            with self.subTest(family=family):
                pieces, _ = self._run(
                    family, [spec["empty"], spec["text"]("x"), spec["empty"]])
                self.assertEqual(pieces, ["x"])

    def test_usage_arrives_and_is_counted_the_same_way(self):
        """11 in and 3 out, however the provider spells it — including
        Anthropic, which sends the two halves in different events."""
        for family, spec in FAMILIES.items():
            with self.subTest(family=family):
                _, skill = self._run(
                    family, [spec["text"]("hi")] + spec["usage"])
                self.assertIsNotNone(skill.last_usage, family)
                self.assertEqual(skill.last_usage.input_tokens, 11, family)
                self.assertEqual(skill.last_usage.output_tokens, 3, family)
                self.assertEqual(skill.last_usage.total_tokens, 14, family)

    def test_silence_about_usage_is_not_a_zero(self):
        for family, spec in FAMILIES.items():
            with self.subTest(family=family):
                _, skill = self._run(family, [spec["text"]("hi")])
                self.assertIsNone(skill.last_usage, family)


class TestTheTrapsPerFamily(unittest.TestCase):
    """Three places where a plausible implementation is wrong, and the wrong
    version returns something that looks right."""

    def test_anthropic_thinking_is_not_spliced_into_the_answer(self):
        """A thinking delta shares the event type with a text delta. Passing
        it through puts the model's private reasoning in front of the user."""
        events = [
            {"type": "content_block_delta",
             "delta": {"type": "thinking_delta", "thinking": "hmm, maybe"}},
            {"type": "content_block_delta",
             "delta": {"type": "text_delta", "text": "The answer."}},
        ]
        skill = _skill(Model("claude-haiku-4-5-20251001", api_key="k"))
        with mock.patch("urllib3.PoolManager.request",
                        return_value=_sse(*events)):
            pieces = list(skill.stream())
        self.assertEqual(pieces, ["The answer."])

    def test_anthropic_keeps_both_halves_of_the_bill(self):
        """Taking the last report is the obvious implementation and loses the
        whole prompt — and the number that comes back is still plausible."""
        events = [{"type": "message_start",
                   "message": {"usage": {"input_tokens": 900}}},
                  {"type": "content_block_delta",
                   "delta": {"type": "text_delta", "text": "hi"}},
                  {"type": "message_delta", "usage": {"output_tokens": 4}}]
        skill = _skill(Model("claude-haiku-4-5-20251001", api_key="k"))
        with mock.patch("urllib3.PoolManager.request",
                        return_value=_sse(*events)):
            list(skill.stream())
        self.assertEqual(skill.last_usage.input_tokens, 900)
        self.assertEqual(skill.last_usage.output_tokens, 4)

    def test_google_changes_the_verb_and_asks_for_sse(self):
        """`:streamGenerateContent` without `alt=sse` answers with one JSON
        array streamed as a single document — parseable only when complete,
        which is a slower non-stream wearing a stream's name."""
        m = Model("gemini-2.5-flash", api_key="k")
        path, _ = m.client.build_stream_request(
            [{"role": "user", "parts": [{"type": "text", "text": "hi"}]}],
            {"format": {"type": "text"}}, m._params())
        self.assertIn(":streamGenerateContent", path)
        self.assertIn("alt=sse", path)

    def test_google_image_generation_is_not_streamed(self):
        m = Model("gemini-2.5-flash-image", api_key="k")
        with self.assertRaises(NotImplementedError):
            m.client.build_stream_request(
                [{"role": "user", "parts": [{"type": "text", "text": "a cat"}]}],
                {"format": {"type": "image"}, "modalities": ["image"]},
                m._params())


if __name__ == "__main__":
    unittest.main()
