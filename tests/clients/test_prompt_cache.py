"""
tests.clients.test_prompt_cache
===============================

Where the prompt cache is *placed*, asserted on the request body.

The three defects this pins were all silent: ``cache_control=True`` produced a
request with no mark at all for a one-system-plus-one-user skill, only the
5-minute lifetime was reachable, and the usage counters that would have shown
either problem were never populated. Nothing raised. A feature that does
nothing and cannot be observed doing nothing needs its wire format asserted,
not its return value.

Pure: no network.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from yait_aichain.models import Model
from yait_aichain.models._usage import (
    CACHE_READ_MULTIPLIER, CACHE_WRITE_MULTIPLIERS, Usage,
    estimate_cost, extract_usage,
)

_OUT = {"modalities": ["text"], "format": {"type": "text"}}

SYS = {"role": "system", "parts": [{"type": "text", "text": "You classify."}]}
U1 = {"role": "user", "parts": [{"type": "text", "text": "first"}]}
A1 = {"role": "assistant", "parts": [{"type": "text", "text": "answer"}]}
U2 = {"role": "user", "parts": [{"type": "text", "text": "second"}]}


def body_for(messages, **options):
    """The request body a Model with *options* would put on the wire."""
    m = Model("claude-sonnet-4-6", api_key="k", options=options)
    _, body = m.client.build_request(messages, _OUT, m._params())
    return body


def marks(body):
    """Every (where, cache_control) pair in a request body, in render order."""
    found = []
    system = body.get("system")
    if isinstance(system, list):
        for i, blk in enumerate(system):
            if isinstance(blk, dict) and "cache_control" in blk:
                found.append((f"system[{i}]", blk["cache_control"]))
    for i, msg in enumerate(body.get("messages", [])):
        for blk in msg.get("content", []):
            if isinstance(blk, dict) and "cache_control" in blk:
                found.append((f"messages[{i}]", blk["cache_control"]))
    return found


class TestPlacement(unittest.TestCase):

    def test_system_prompt_plus_one_user_message_is_marked(self):
        """
        The shape the feature exists for, and the one it silently skipped.

        A classification or extraction skill is a stable system prompt and one
        short variable message. There is no second-to-last message to mark, and
        an all-text system prompt had already been collapsed into a string with
        no block to carry the mark — so both branches missed and the request
        went out at full price.
        """
        body = body_for([SYS, U1], cache_control=True)
        self.assertEqual(marks(body), [("system[0]", {"type": "ephemeral"})])

    def test_the_mark_is_never_on_the_variable_message(self):
        """Marking the only user message would cache a prefix that never repeats."""
        body = body_for([SYS, U1], cache_control=True)
        self.assertEqual(body["messages"][0]["content"][0].get("cache_control"), None)

    def test_multi_turn_still_marks_the_second_to_last_message(self):
        body = body_for([SYS, U1, A1, U2], cache_control=True)
        self.assertEqual(marks(body), [("messages[1]", {"type": "ephemeral"})])

    def test_index_zero_is_a_real_answer(self):
        """
        Zero is false, and the mark used to be tested for truthiness — so a
        caller whose stable prefix ends at the very first message, the usual
        case for a fixed preamble, got no cache at all and no way to find out.
        """
        body = body_for([SYS, U1, A1, U2], cache_control=0)
        self.assertEqual(marks(body), [("messages[0]", {"type": "ephemeral"})])

    def test_a_later_index_names_that_message(self):
        body = body_for([SYS, U1, A1, U2], cache_control=1)
        self.assertEqual(marks(body), [("messages[1]", {"type": "ephemeral"})])

    def test_an_index_naming_no_message_marks_nothing(self):
        """
        ``-1`` means "nothing is stable yet", not "the last message" — Python
        would read it as the newest turn, the one guaranteed to differ next
        call. An explicit index is an instruction, so an index that names no
        message falls through to nothing rather than quietly to the system
        prompt; the caller who guesses gets ``True``, not an integer.
        """
        for bad in (-1, 99):
            with self.subTest(index=bad):
                self.assertEqual(marks(body_for([SYS, U1, A1, U2], cache_control=bad)), [])

    def test_two_system_messages_are_joined_and_marked_once(self):
        body = body_for([SYS, SYS, U1], cache_control=True)
        self.assertEqual(len(marks(body)), 1)
        self.assertEqual(marks(body)[0][0], "system[0]")

    def test_off_by_default(self):
        self.assertEqual(marks(body_for([SYS, U1])), [])
        self.assertEqual(marks(body_for([SYS, U1], cache_control=False)), [])


class TestLifetime(unittest.TestCase):

    def test_default_is_the_five_minute_cache(self):
        body = body_for([SYS, U1], cache_control=True)
        self.assertEqual(marks(body)[0][1], {"type": "ephemeral"})

    def test_one_hour_reaches_the_wire(self):
        body = body_for([SYS, U1], cache_control=True, cache_ttl="1h")
        self.assertEqual(marks(body)[0][1], {"type": "ephemeral", "ttl": "1h"})

    def test_an_unknown_lifetime_is_refused_at_construction(self):
        with self.assertRaises(ValueError):
            Model("claude-sonnet-4-6", api_key="k", options={"cache_ttl": "30m"})


class TestUsageAccounting(unittest.TestCase):

    def test_anthropic_counters_surface(self):
        u = extract_usage({"usage": {
            "input_tokens": 12, "output_tokens": 7,
            "cache_creation_input_tokens": 4000,
            "cache_read_input_tokens": 3800,
        }})
        self.assertEqual((u.cache_write_tokens, u.cache_read_tokens), (4000, 3800))
        self.assertEqual(u.input_tokens, 12)

    def test_openai_cached_tokens_are_taken_out_of_the_input_count(self):
        """
        The providers disagree, and taken at face value the disagreement bills
        a reused OpenAI prefix at 1.1x instead of 0.1x: Anthropic reports the
        cached prefix *beside* ``input_tokens``, OpenAI counts it *inside*
        ``prompt_tokens``.
        """
        u = extract_usage({"usage": {
            "prompt_tokens": 5000, "completion_tokens": 50,
            "prompt_tokens_details": {"cached_tokens": 4608},
        }})
        self.assertEqual(u.cache_read_tokens, 4608)
        self.assertEqual(u.input_tokens, 392)

    def test_a_read_costs_a_tenth_of_a_fresh_token(self):
        fresh = estimate_cost(Usage(input_tokens=1_000_000), "gpt-4o")
        cached = estimate_cost(Usage(cache_read_tokens=1_000_000), "gpt-4o")
        self.assertAlmostEqual(cached / fresh, CACHE_READ_MULTIPLIER, places=6)

    def test_the_hour_long_write_is_priced_above_the_five_minute_one(self):
        w = Usage(cache_write_tokens=1_000_000)
        short = estimate_cost(w, "gpt-4o", "5m")
        long = estimate_cost(w, "gpt-4o", "1h")
        self.assertAlmostEqual(
            long / short,
            CACHE_WRITE_MULTIPLIERS["1h"] / CACHE_WRITE_MULTIPLIERS["5m"],
            places=6)


if __name__ == "__main__":
    unittest.main()
