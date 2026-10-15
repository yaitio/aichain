"""
The forms a Chain step is written in, and what each says.

4.2.7: a step is named after its runner when no key is given, `(runner,
inputs)` is a form, and `Step` says a step's own error policy and whether a
dict it returns is spread among the variables.
"""

import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from yait_aichain.chain import Chain, Step


def fetch(url: str):
    return {"title": "T", "body": "B " + url}


def count(text: str) -> int:
    return len(text.split())


def broken(url: str):
    raise RuntimeError("down")


class TestForms(unittest.TestCase):

    def test_runner_and_inputs_is_named_after_the_runner(self):
        result = Chain([fetch, (count, {"text": "fetch.body"})]).run(variables={"url": "x"})
        self.assertEqual(result["count"], 2)

    def test_a_step_says_its_key_and_inputs(self):
        result = Chain([Step(fetch, out="page"),
                        Step(count, out="words", inputs={"text": "page.body"})]).run(
            variables={"url": "x"})
        self.assertEqual(result["words"], 2)

    def test_bad_values_are_refused_when_the_step_is_made(self):
        with self.assertRaises(ValueError):
            Step(fetch, on_error="ignore")
        with self.assertRaises(ValueError):
            Step(fetch, out=3)


class TestSpread(unittest.TestCase):

    def _run(self, spread):
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            result = Chain([Step(fetch, out="page", spread=spread)]).run(variables={"url": "x"})
        return result, [w for w in seen if "spread" in str(w.message)]

    def test_said_true_keeps_it_quietly(self):
        result, said = self._run(True)
        self.assertEqual((result["body"], said), ("B x", []))

    def test_said_false_keeps_the_dict_alone(self):
        result, said = self._run(False)
        self.assertNotIn("body", result)
        self.assertEqual((result["page.body"], said), ("B x", []))

    def test_unsaid_spreads_and_warns_once_per_chain(self):
        chain = Chain([(fetch, "page")])
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            chain.run(variables={"url": "x"})
            chain.run(variables={"url": "y"})
        self.assertEqual(len([w for w in seen if "spread" in str(w.message)]), 1)


class TestAStepsOwnErrorPolicy(unittest.TestCase):

    def test_it_wins_over_the_chains(self):
        result = Chain([Step(broken, on_error="skip"), (count, "words", {"text": "url"})],
                       on_step_error="raise").run(variables={"url": "a b"})
        self.assertEqual(result["words"], 2)

    def test_the_chains_holds_where_a_step_says_nothing(self):
        with self.assertRaises(RuntimeError):
            Chain([Step(broken)], on_step_error="raise").run(variables={"url": "a"})


class TestNamesThatCannotBeAKey(unittest.TestCase):

    def test_a_lambda_writes_result_with_a_warning(self):
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            chain = Chain([lambda url: url.upper()])
        self.assertTrue(any("no key" in str(w.message) for w in seen))
        self.assertEqual(chain.run(variables={"url": "x"})["result"], "X")


if __name__ == "__main__":
    unittest.main()


class TestAPathReadsAListByPosition(unittest.TestCase):

    def test_a_number_in_a_path_is_a_position(self):
        from yait_aichain import Chain
        from yait_aichain.chain import Step

        def pages() -> dict:
            return {"pages": [{"text": "first"}, {"text": "second"}]}

        def upper(text: str) -> str:
            return text.upper()

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            result = Chain([(pages, "page"), Step(upper, inputs={"text": "page.pages.1.text"})]).run()
        self.assertEqual(result.output, "SECOND")

    def test_a_position_past_the_end_says_how_many_there_are(self):
        from yait_aichain import Chain
        from yait_aichain.chain import Step

        def pages() -> dict:
            return {"pages": [{"text": "only"}]}

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            with self.assertRaisesRegex(ValueError, "has 1 item"):
                Chain([(pages, "page"), Step(len, inputs={"obj": "page.pages.3"})],
                      on_step_error="raise").run()
