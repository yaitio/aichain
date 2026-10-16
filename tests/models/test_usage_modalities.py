"""
Not every token in a request costs the same, and the provider says so.

Both shapes below are verbatim from live calls on 2026-09-09 — an OpenAI
image edit and a Gemini vision turn — because a usage parser written against
a guessed shape is a guess about money.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from yait_aichain.models._usage import Usage, extract_usage, estimate_cost

# POST /v1/images/edits, gpt-image-2.5-flare, one reference image
OPENAI_EDIT = {"usage": {
    "input_tokens": 1043,
    "input_tokens_details": {"image_tokens": 1024, "text_tokens": 19},
    "output_tokens": 196,
    "output_tokens_details": {"image_tokens": 196, "text_tokens": 0},
    "total_tokens": 1239}}

# gemini-3.1-flash-lite asked the colour of a 64x64 square
GOOGLE_VISION = {"usageMetadata": {
    "promptTokenCount": 267, "candidatesTokenCount": 1, "totalTokenCount": 503,
    "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 9},
                            {"modality": "IMAGE", "tokenCount": 258}],
    "thoughtsTokenCount": 235, "serviceTier": "standard"}}


class TestImageInputIsSeparate(unittest.TestCase):

    def test_openai_reports_the_split_and_it_is_kept(self):
        u = extract_usage(OPENAI_EDIT)
        self.assertEqual(u.image_input_tokens, 1024)
        self.assertEqual(u.input_tokens, 19)      # text only, disjoint

    def test_google_reports_it_by_modality(self):
        u = extract_usage(GOOGLE_VISION)
        self.assertEqual(u.image_input_tokens, 258)
        self.assertEqual(u.input_tokens, 9)

    def test_an_image_is_billed_at_its_own_rate(self):
        """$8/M against text's $5/M — the whole reason for the split."""
        u = extract_usage(OPENAI_EDIT)
        cost = estimate_cost(u, "gpt-image-2.5-flare")
        self.assertAlmostEqual(
            cost, (19 * 5 + 1024 * 8 + 196 * 30) / 1_000_000, places=9)

    def test_one_rate_under_states_this_call_by_22_percent(self):
        """The number is worth pinning: it is what the gap actually was.

        On the input line alone the gap is 37%; across the whole call, output
        included, it is 22%. Quoting the first as if it were the second is how
        a figure computed in one frame ends up reported in another."""
        u = extract_usage(OPENAI_EDIT)
        honest = estimate_cost(u, "gpt-image-2.5-flare")
        naive = ((u.input_tokens + u.image_input_tokens) * 5
                 + u.output_tokens * 30) / 1_000_000
        self.assertAlmostEqual(honest / naive, 1.277, places=3)

    def test_a_model_without_the_field_falls_back_to_the_text_rate(self):
        """Most providers charge one rate; only those that do not carry it."""
        u = Usage(input_tokens=100, image_input_tokens=100, output_tokens=0)
        self.assertAlmostEqual(estimate_cost(u, "gpt-4o"),
                               200 * 2.5 / 1_000_000, places=9)


class TestThinkingTokensAreBilled(unittest.TestCase):

    def test_thoughts_join_the_output_count(self):
        """Google: "response pricing is the sum of output tokens and thinking
        tokens". Dropped, a reasoning call reads as 1 token instead of 236."""
        u = extract_usage(GOOGLE_VISION)
        self.assertEqual(u.output_tokens, 236)

    def test_a_reply_without_thinking_is_unchanged(self):
        u = extract_usage({"usageMetadata": {"promptTokenCount": 10,
                                             "candidatesTokenCount": 5,
                                             "totalTokenCount": 15}})
        self.assertEqual((u.input_tokens, u.output_tokens), (10, 5))


class TestAddition(unittest.TestCase):

    def test_image_tokens_survive_summation(self):
        """Pool and Chain sum usage; a field that does not add is lost there."""
        total = extract_usage(OPENAI_EDIT) + extract_usage(GOOGLE_VISION)
        self.assertEqual(total.image_input_tokens, 1024 + 258)


class TestRatesAModelCarriesItself(unittest.TestCase):
    """Prices the registry reads per model, not from a global rule (2026-10-10)."""

    def test_a_cache_read_cheaper_than_a_tenth(self):
        # Fable 5.1 reads at $0.25/M against $10/M fresh — a fortieth. The
        # global tenth would bill a million cached tokens at $1.00.
        u = Usage(cache_read_tokens=1_000_000)
        self.assertAlmostEqual(estimate_cost(u, "claude-fable-5-1"), 0.25)
        # A model without the field keeps the tenth.
        self.assertAlmostEqual(estimate_cost(u, "claude-opus-4-8"), 0.50)

    def test_past_the_long_context_line_the_whole_call_is_dearer(self):
        short = Usage(input_tokens=100_000, output_tokens=1_000)
        long_ = Usage(input_tokens=100_001, output_tokens=1_000)
        self.assertAlmostEqual(estimate_cost(short, "claude-haiku-5-5"), 0.0105)
        self.assertAlmostEqual(estimate_cost(long_, "claude-haiku-5-5"), 100_001 * 0.5e-6 + 0.0025)

    def test_a_model_priced_by_the_picture_has_no_token_cost(self):
        from yait_aichain.models._usage import price_per_image
        self.assertIsNone(estimate_cost(Usage(input_tokens=10, output_tokens=10), "grok-imagine-image"))
        self.assertEqual(price_per_image("grok-imagine-image"), 0.02)
        self.assertIsNone(price_per_image("claude-haiku-5-5"))


class TestACostTheProviderReports(unittest.TestCase):
    """OpenRouter bills in the response: `usage.cost`, verbatim from a live
    call on 2026-10-11. The bill wins over the registry's arithmetic, and a
    model the registry does not price is no longer `None`."""

    OPENROUTER = {"usage": {"prompt_tokens": 10, "completion_tokens": 13, "total_tokens": 23,
                            "cost": 5.04e-06, "is_byok": False,
                            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0}}}

    def test_it_is_read(self):
        from yait_aichain.models._usage import attach_cost
        u = attach_cost(extract_usage(self.OPENROUTER), "xiaomi/mimo-v2.6-flash")
        self.assertEqual(u.cost, 5.04e-06)

    def test_it_wins_over_the_registry(self):
        from yait_aichain.models._usage import attach_cost
        u = attach_cost(extract_usage(self.OPENROUTER), "claude-haiku-5-5")
        self.assertEqual(u.cost, 5.04e-06)

    def test_without_it_the_registry_prices(self):
        from yait_aichain.models._usage import attach_cost
        u = attach_cost(extract_usage({"usage": {"prompt_tokens": 10_000, "completion_tokens": 0}}),
                        "claude-haiku-5-5")
        self.assertAlmostEqual(u.cost, 0.001)


if __name__ == "__main__":
    unittest.main()
