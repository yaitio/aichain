"""
A universal name is only half the promise; the value has to travel too.

`reasoning="medium"` becomes 10000 budget tokens on Anthropic, 8192 on
Google, the string "medium" on OpenAI, `true` on Qwen and a different model
on DeepSeek. One word, five shapes, written once by the caller — that is what
"the same API whatever the provider" has to mean, or the name is portable and
the value is not.

The mapping existed for `reasoning` alone. It is the same need elsewhere:
Reve measures render quality as compute, a number from 1 to 15, and expresses
a transparent background as a step in a post-processing list.
"""

import sys
import unittest
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from yait_aichain import Model
from yait_aichain.models._options import value_map, to_provider_value

MSGS = [{"role": "user", "parts": [{"type": "text", "text": "hi"}]}]


def _body(name, options=None, fmt=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = Model(name, api_key="k", options=options or None)
        _, body = m.to_request(
            MSGS, {"format": {"type": "image" if fmt else "text", **(fmt or {})}})
    return m, body


class TestOneLevelManyScales(unittest.TestCase):

    def test_the_same_word_becomes_a_number_where_the_provider_counts(self):
        _, anthropic = _body("claude-sonnet-4-6", {"reasoning": "medium"})
        self.assertEqual(anthropic["thinking"]["budget_tokens"], 10000)

        _, google = _body("gemini-2.5-flash", {"reasoning": "medium"})
        self.assertEqual(
            google["generationConfig"]["thinkingConfig"]["thinkingBudget"], 8192)

    def test_a_word_where_the_provider_uses_words(self):
        _, openai = _body("gpt-5.5", {"reasoning": "medium"})
        self.assertEqual(openai["reasoning"]["effort"], "medium")

    def test_a_flag_where_the_provider_has_only_on_and_off(self):
        _, qwen = _body("qwen3-32b", {"reasoning": "medium"})
        self.assertIs(qwen["enable_thinking"], True)

    def test_a_model_name_where_reasoning_is_a_separate_model(self):
        _, deepseek = _body("deepseek-chat", {"reasoning": "high"})
        self.assertEqual(deepseek["model"], "deepseek-reasoner")


class TestTheMapIsNotOnlyForReasoning(unittest.TestCase):

    def test_quality_becomes_this_provider_s_compute_number(self):
        _, reve = _body("reve-image", fmt={"quality": "high"})
        self.assertEqual(reve["test_time_scaling"], 12)

    def test_a_lower_level_is_a_lower_number(self):
        _, reve = _body("reve-image", fmt={"quality": "low"})
        self.assertEqual(reve["test_time_scaling"], 3)

    def test_a_setting_can_become_a_step_in_a_list(self):
        """Not every provider has a field for every intent — here a
        transparent background is something done afterwards."""
        _, reve = _body("reve-image", fmt={"background": "transparent"})
        self.assertIn("remove_background", reve["postprocessing"])

    def test_the_caller_s_own_postprocessing_is_not_lost(self):
        _, reve = _body("reve-image", fmt={"background": "transparent",
                                           "postprocessing": ["upscale"]})
        self.assertEqual(sorted(reve["postprocessing"]),
                         ["remove_background", "upscale"])


class TestTheMechanism(unittest.TestCase):

    def test_the_older_reasoning_map_still_counts_as_a_map(self):
        """Provider files written before the generalisation must keep
        working: absence of the new spelling is not absence of the map."""
        self.assertEqual(value_map("reasoning", "anthropic")["medium"], 10000)

    def test_an_unmapped_value_passes_through(self):
        """The map names the levels a caller may ask for, not everything a
        provider will take."""
        sent, note = to_provider_value("quality", "high", "bfl")
        self.assertEqual((sent, note), ("high", None))


if __name__ == "__main__":
    unittest.main()


class TestOneAxisRunBothWays(unittest.TestCase):
    """`input_fidelity` and `strength` were the same axis, inverted.

    OpenAI asks how much of the original to preserve; Recraft asks how much to
    change. Measured on Recraft 2026-09-09 with a blue square and the prompt
    "a red circle": strength 0.05 came back blue, strength 0.95 came back a
    red circle — which is what fixes the direction rather than inferring it
    from the word.

    They are one option now, `fidelity`, meaning what survives. The canonical
    type is a number; `low`/`high` are shorthands resolved on the way in, so a
    range declared for one provider does not reject a word another requires.
    """

    def _edit(self, name, **fmt):
        import base64
        png = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()
        msgs = [{"role": "user", "parts": [
            {"type": "text", "text": "darker"},
            {"type": "image", "source": {"kind": "base64",
                                         "mime": "image/png", "data": png}}]}]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = Model(name, api_key="k")
            _, body = m.to_request(msgs, {"format": {"type": "image", **fmt}})
        return dict(body["fields"])

    def test_keeping_the_original_means_a_small_change_on_recraft(self):
        self.assertEqual(self._edit("recraftv3", fidelity=0.9)["strength"], "0.1")

    def test_and_a_large_one_when_little_is_kept(self):
        self.assertEqual(self._edit("recraftv3", fidelity=0.1)["strength"], "0.9")

    def test_the_same_number_becomes_a_word_on_openai(self):
        """Two levels there, a scale here."""
        self.assertEqual(
            self._edit("gpt-image-1.5", fidelity=0.9)["input_fidelity"], "high")
        self.assertEqual(
            self._edit("gpt-image-1.5", fidelity=0.1)["input_fidelity"], "low")

    def test_a_word_works_on_both(self):
        self.assertEqual(self._edit("recraftv3", fidelity="high")["strength"],
                         "0.1")
        self.assertEqual(
            self._edit("gpt-image-1.5", fidelity="high")["input_fidelity"],
            "high")

    def test_the_old_name_still_arrives(self):
        """`input_fidelity` shipped in 2.2.x and cannot stop working."""
        self.assertEqual(
            self._edit("gpt-image-1.5", input_fidelity="high")["input_fidelity"],
            "high")
