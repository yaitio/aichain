"""
A chain with an agent in it must come back as the agent that went in.

It did not. `Chain.save()` read the agent through `getattr(runner, "...",
default)` under the pre-2.0 names — `orchestrator`, `max_steps`,
`max_attempts`, `max_tokens`, `persona` — every one of which stopped existing
in `2.0.0`. Each read fell back to its default, so a saved chain lost the
model, the instructions and the stop conditions, and gained three invented
budget numbers; `Chain.load()` then raised `TypeError` on the null model name.

The defaults are what made it quiet. `getattr` with a fallback turns a renamed
attribute into a plausible value, and plausible values do not fail tests —
which is why this shipped through four minor versions with a documented
"Persistence inside Chain.save()" section describing it.
"""

import os
import sys
import tempfile
import unittest
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

import yaml                                                        # noqa: E402

from yait_aichain import Agent, Chain, Model                       # noqa: E402
from yait_aichain.agent import check, step_count, token_budget     # noqa: E402


def _agent(**kw):
    return Agent(Model("gpt-4o", api_key="k"), name="researcher",
                 instructions="Be terse.", **kw)


def _round_trip(agent):
    path = os.path.join(tempfile.mkdtemp(), "chain.yaml")
    chain = Chain(steps=[(agent, "out")])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chain.save(path)
    return yaml.safe_load(open(path))["steps"][0]["agent"], \
        Chain.load(path, api_key="k")._steps[0][0]


class TestWhatGoesInComesBack(unittest.TestCase):

    def test_the_model_survives(self):
        """The one the old code could not carry: it wrote `orchestrator: null`
        and load died on it."""
        _, back = _round_trip(_agent())
        self.assertEqual(back.model.name, "gpt-4o")

    def test_the_instructions_survive(self):
        _, back = _round_trip(_agent())
        self.assertEqual(back.instructions, "Be terse.")

    def test_the_ceilings_survive(self):
        """A stop condition is a closure and a file cannot hold one, so each
        ceiling carries a `spec` naming itself and its number."""
        _, back = _round_trip(_agent(stop_when=[step_count(3),
                                                token_budget(9000)]))
        self.assertEqual(sorted(c.name for c in back.stop_when),
                         ["step_count", "token_budget"])

    def test_the_labels_survive(self):
        _, back = _round_trip(_agent())
        self.assertEqual(back.name, "researcher")

    def test_nothing_invented_is_written(self):
        """Three budget numbers used to be fabricated from `getattr`
        defaults, which is worse than losing them: a reader cannot tell an
        invented ceiling from one somebody chose."""
        block, _ = _round_trip(_agent())
        for gone in ("orchestrator", "max_steps", "max_attempts",
                     "max_tokens", "persona"):
            self.assertNotIn(gone, block)


class TestWhatCannotComeBackIsSaidOutLoud(unittest.TestCase):

    def test_a_check_warns_rather_than_vanishing(self):
        """Its predicate is the caller's own function. Pretending it
        round-trips would be worse than saying it does not — a chain that
        comes back quietly without its stop condition is the exact failure
        this whole repair is about."""
        path = os.path.join(tempfile.mkdtemp(), "chain.yaml")
        chain = Chain(steps=[(_agent(stop_when=[check(lambda s: False)]),
                              "out")])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            chain.save(path)
        self.assertTrue(any("cannot be serialised" in str(w.message)
                            for w in caught))

    def test_a_file_from_the_broken_versions_is_refused(self):
        """It carries `orchestrator: null` — a key that never held the model
        name. Papering over it would rebuild a different agent and say
        nothing."""
        path = os.path.join(tempfile.mkdtemp(), "old.yaml")
        with open(path, "w") as fh:
            yaml.safe_dump({"version": 1, "steps": [{
                "kind": "agent", "output_key": "out",
                "agent": {"class": "yait_aichain.agent._agent.Agent",
                          "orchestrator": None, "mode": "agile",
                          "max_steps": 10}}]}, fh)
        with self.assertRaises(ValueError) as caught:
            Chain.load(path, api_key="k")
        self.assertIn("model name was never", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
