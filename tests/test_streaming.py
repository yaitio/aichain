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




class TestRecraftDoesNotClaimWhatItInherited(unittest.TestCase):

    def test_an_image_provider_does_not_stream(self):
        """It renders images and subclasses the client that streams. A
        capability arriving through the class hierarchy is the same defect
        the option layer was cleared of in 2.3.0, by another door."""
        self.assertFalse(Model("recraftv3", api_key="k").client.supports_streaming)


# ── Tool calls, reassembled ─────────────────────────────────────────────────
#
# A call does not arrive as a call. An id and a name land once, the arguments
# trickle in as fragments of a JSON string, and two calls in one turn
# interleave. Every trap below has the same shape: the wrong implementation
# produces a call that still runs, with the wrong arguments or none.

def _oa_tc(slot, *, id=None, name=None, args=None):
    d = {"index": slot}
    if id:
        d["id"] = id
    fn = {}
    if name:
        fn["name"] = name
    if args is not None:
        fn["arguments"] = args
    if fn:
        d["function"] = fn
    return {"choices": [{"delta": {"tool_calls": [d]}}]}


class TestToolCallsSurviveAStream(unittest.TestCase):

    def _stream(self, model_name, events):
        skill = Skill(model=Model(model_name, api_key="k"),
                      input={"messages": [{"role": "user", "parts": [
                          {"type": "text", "text": "hi"}]}]},
                      output={"format": {"type": "text"}},
                      _tools=[{"function": {"name": "echo",
                                            "parameters": {}}}])
        with mock.patch("urllib3.PoolManager.request",
                        return_value=_sse(*events)):
            pieces = list(skill.stream())
        return pieces, skill

    def test_openai_arguments_are_concatenated_then_parsed_once(self):
        """Parsing on the way sees truncated JSON on every fragment but the
        last, and unparseable arguments recover to an empty dict — so the
        tool runs, with nothing, and the run carries on."""
        _, skill = self._stream("gpt-4o", [
            _oa_tc(0, id="c1", name="echo", args=""),
            _oa_tc(0, args='{"value":'),
            _oa_tc(0, args=' "one"}'),
        ])
        call = skill.last_result.calls[0]
        self.assertEqual(call.name, "echo")
        self.assertEqual(call.id, "c1")
        self.assertEqual(call.arguments, {"value": "one"})

    def test_two_calls_in_one_turn_do_not_bleed_into_each_other(self):
        """Keyed on the provider's index, not on arrival order: the
        fragments interleave, and keying on order splices both argument
        strings into one that does not parse."""
        _, skill = self._stream("gpt-4o", [
            _oa_tc(0, id="c1", name="echo", args=""),
            _oa_tc(1, id="c2", name="echo", args=""),
            _oa_tc(0, args='{"value": "first"}'),
            _oa_tc(1, args='{"value": "second"}'),
        ])
        calls = skill.last_result.calls
        self.assertEqual([c.id for c in calls], ["c1", "c2"])
        self.assertEqual([c.arguments["value"] for c in calls],
                         ["first", "second"])

    def test_a_later_fragment_does_not_erase_the_name(self):
        """The name arrives once; the fragments after it carry arguments
        alone. Writing each fragment's empty name over the stored one leaves
        a call nothing can route."""
        _, skill = self._stream("gpt-4o", [
            _oa_tc(0, id="c1", name="echo", args="{}"),
            _oa_tc(0, args=""),
        ])
        self.assertEqual(skill.last_result.calls[0].name, "echo")

    def test_prose_beside_a_call_is_yielded_and_kept(self):
        """Some providers send both. The text is streamed as it arrives and
        rides along on the request, so it is neither shown twice nor lost."""
        pieces, skill = self._stream("gpt-4o", [
            _delta("Let me check. "),
            _oa_tc(0, id="c1", name="echo", args="{}"),
        ])
        self.assertEqual(pieces, ["Let me check. "])
        self.assertEqual(skill.last_result.text, "Let me check. ")
        self.assertEqual(skill.last_result.calls[0].name, "echo")

    def test_a_turn_with_no_calls_is_still_just_text(self):
        pieces, skill = self._stream("gpt-4o", [_delta("plain answer")])
        self.assertEqual(pieces, ["plain answer"])
        self.assertEqual(skill.last_result, "plain answer")

    def test_anthropic_streams_the_arguments_under_another_name(self):
        """The call's arguments arrive as `input_json_delta.partial_json`,
        not as text — which is why the text filter never saw them."""
        _, skill = self._stream("claude-haiku-4-5-20251001", [
            {"type": "content_block_start", "index": 1,
             "content_block": {"type": "tool_use", "id": "toolu_1",
                               "name": "echo", "input": {}}},
            {"type": "content_block_delta", "index": 1,
             "delta": {"type": "input_json_delta",
                       "partial_json": '{"value":'}},
            {"type": "content_block_delta", "index": 1,
             "delta": {"type": "input_json_delta",
                       "partial_json": ' "one"}'}},
            {"type": "content_block_stop", "index": 1},
        ])
        call = skill.last_result.calls[0]
        self.assertEqual((call.id, call.name), ("toolu_1", "echo"))
        self.assertEqual(call.arguments, {"value": "one"})

    def test_anthropic_keeps_a_text_block_out_of_the_call(self):
        pieces, skill = self._stream("claude-haiku-4-5-20251001", [
            {"type": "content_block_delta", "index": 0,
             "delta": {"type": "text_delta", "text": "Checking."}},
            {"type": "content_block_start", "index": 1,
             "content_block": {"type": "tool_use", "id": "toolu_1",
                               "name": "echo", "input": {}}},
            {"type": "content_block_delta", "index": 1,
             "delta": {"type": "input_json_delta", "partial_json": "{}"}},
        ])
        self.assertEqual(pieces, ["Checking."])
        self.assertEqual(skill.last_result.calls[0].name, "echo")

    def test_google_sends_a_call_whole_rather_than_in_pieces(self):
        """`args` is already an object, not a JSON string — the one family
        that needs no reassembly, handled by the same assembler rather than
        by a special case."""
        _, skill = self._stream("gemini-2.5-flash", [
            {"candidates": [{"content": {"parts": [
                {"functionCall": {"name": "echo",
                                  "args": {"value": "one"}}}]}}]},
        ])
        call = skill.last_result.calls[0]
        self.assertEqual(call.name, "echo")
        self.assertEqual(call.arguments, {"value": "one"})

    def test_a_call_is_not_rendered_into_the_stream(self):
        """The pieces are what a caller prints. A decision is not prose and
        must not appear there as one."""
        pieces, _ = self._stream("gpt-4o", [
            _oa_tc(0, id="c1", name="echo", args="{}")])
        self.assertEqual(pieces, [])


class TestTheFlagMeansWhatItSays(unittest.TestCase):
    """`supports_streaming` is a claim, and 2.3.0's rule is that a claim is
    established by effect. Every provider declaring one must build a
    streaming request for a plain text turn without raising."""

    def test_no_provider_claims_more_than_it_does(self):
        from yait_aichain.models._data import PROVIDERS
        from yait_aichain.models._base import _build_client, models as _models

        msgs = [{"role": "user", "parts": [{"type": "text", "text": "hi"}]}]
        for name in sorted(PROVIDERS):
            client = _build_client(name, "k", {})
            if not client.supports_streaming:
                continue
            with self.subTest(provider=name):
                # `private` lists no models on purpose — a self-hosted
                # server takes whatever id it was started with — so it is
                # probed with one rather than skipped.
                text_models = [n for n in _models(provider=name)
                               if "image" not in n] or ["any-local-model"]
                m = Model(f"{name}/{text_models[0]}", api_key="k")
                path, body = m.client.build_stream_request(
                    msgs, {"format": {"type": "text"}}, m._params())
                self.assertTrue(path)


if __name__ == "__main__":
    unittest.main()


class TestAStreamRetriesWhileItStillCan(unittest.TestCase):
    """2.5.0 refused to retry a stream at all, reasoning that a rule holding
    only before the first byte "holds sometimes". That was wrong: *before the
    first piece* is not a sometimes, it is a state the caller can see —
    nothing has been yielded — so there is no reason to be less reliable than
    `run()` while the attempt is still discardable."""

    def _skill(self, **kw):
        return Skill(model=Model("gpt-4o", api_key="k"),
                     input={"messages": [{"role": "user", "parts": [
                         {"type": "text", "text": "hi"}]}]},
                     output={"format": {"type": "text"}},
                     retry_delay=0, **kw)

    def test_a_rate_limit_before_the_first_piece_is_retried(self):
        from yait_aichain.clients._errors import RateLimitError
        attempts = []

        def flaky(*a, **k):
            attempts.append(1)
            if len(attempts) == 1:
                raise RateLimitError(429, "slow down")
            return _sse(_delta("hello"))

        skill = self._skill(max_retries=2)
        with mock.patch("urllib3.PoolManager.request", side_effect=flaky):
            pieces = list(skill.stream())
        self.assertEqual(pieces, ["hello"])
        self.assertEqual(len(attempts), 2)

    def test_a_failure_after_the_first_piece_is_not(self):
        """Starting over would replay or contradict what the caller has
        already seen."""
        from yait_aichain.clients._errors import RateLimitError

        class Broken:
            status, headers = 200, {}

            def stream(self, amt=None, decode_content=True):
                yield b'data: {"choices":[{"delta":{"content":"par"}}]}\n\n'
                raise RateLimitError(429, "mid-stream")

            def read(self):
                return b""

            def release_conn(self):
                pass

        skill = self._skill(max_retries=2)
        seen = []
        with mock.patch("urllib3.PoolManager.request", return_value=Broken()):
            with self.assertRaises(RateLimitError):
                for piece in skill.stream():
                    seen.append(piece)
        self.assertEqual(seen, ["par"])

    def test_a_permanent_error_is_not_retried_either(self):
        """A 400 arrives as a response, not as a transport exception — which
        is the shape worth testing, because the transport wraps anything it
        catches as a NetworkError and a NetworkError is transient."""
        from yait_aichain.clients._errors import APIError
        attempts = []

        class Refused:
            status, headers = 400, {}

            def __init__(self):
                attempts.append(1)

            def read(self):
                return b'{"error": "bad request"}'

            def release_conn(self):
                pass

        with mock.patch("urllib3.PoolManager.request",
                        side_effect=lambda *a, **k: Refused()):
            with self.assertRaises(APIError):
                list(self._skill(max_retries=2).stream())
        self.assertEqual(len(attempts), 1)
