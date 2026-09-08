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

# gemini-2.5-flash asked the colour of a 64x64 square
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


if __name__ == "__main__":
    unittest.main()
