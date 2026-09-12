"""
Create → save → load → run, for each primitive that can be saved.

`PLAN.md` has carried this item since the June audit with a blunt reason
attached: *their absence is precisely what let five dead features ship.* It
was still absent on 2026-09-11, when a chain containing an agent turned out
not to survive a round trip at all — the serialiser read the agent through
pre-2.0 attribute names with a `getattr` default for each, so the file lost
the model, the instructions and the stop conditions, and `load()` raised
`TypeError`. Found by hand, four minor versions late.

The shape of that defect is why these are end-to-end and not unit tests: every
individual piece worked. `save()` wrote a file, `load()` read one, the schema
matched its documentation. Only running the whole lifecycle in one breath
showed that what came back was not what went in.

Where a primitive does **not** serialise, that is stated rather than skipped:
a missing test and a deliberate absence look identical in a test run, and one
of them is a defect.
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

from yait_aichain import Agent, Chain, Model, Pool, Skill      # noqa: E402
from yait_aichain.agent import step_count                      # noqa: E402


def _answering(model, text="ANSWER"):
    """A model whose transport answers, so `run()` is a real call path."""
    model.client._post = MagicMock(return_value=json.dumps({
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2,
                  "total_tokens": 6},
    }).encode())
    model.client._auth_headers = MagicMock(return_value={})
    return model


def _skill(**kw):
    return Skill(model=_answering(Model("gpt-4o", api_key="k")),
                 input={"messages": [{"role": "user",
                                      "parts": ["Greet {name}"]}]},
                 name="greet", **kw)


def _path(stem):
    return os.path.join(tempfile.mkdtemp(), f"{stem}.yaml")


class TestSkill(unittest.TestCase):

    def test_the_whole_lifecycle(self):
        path = _path("skill")
        _skill().save(path)
        back = Skill.load(path, api_key="k")
        _answering(back.models[0])
        self.assertEqual(back.run(variables={"name": "Ada"}), "ANSWER")

    def test_what_went_in_is_what_comes_back(self):
        """The half a "did it not raise?" smoke misses: a round trip that
        loses a field still runs, and answers a different question."""
        path = _path("skill")
        _skill(variables={"name": "Ada"}, max_retries=2).save(path)
        back = Skill.load(path, api_key="k")
        self.assertEqual(back.name, "greet")
        self.assertEqual(back.models[0].name, "gpt-4o")
        self.assertEqual(back.variables, {"name": "Ada"})
        self.assertIn("Greet {name}",
                      json.dumps(back._input, ensure_ascii=False))


class TestChain(unittest.TestCase):

    def test_the_whole_lifecycle(self):
        path = _path("chain")
        Chain(steps=[(_skill(), "greeting")]).save(path)
        back = Chain.load(path, api_key="k")
        _answering(back._steps[0][0].models[0])
        # `run()` returns the last step's value, not the accumulated dict —
        # checked rather than assumed, because a smoke that asserts the wrong
        # shape fails on itself and teaches nothing.
        self.assertEqual(back.run(variables={"name": "Ada"}), "ANSWER")

    def test_an_agent_step_survives_it_too(self):
        """The case that was broken until 2.6.1. Kept here as well as in its
        own file, because this is the suite someone reads when they ask
        "does saving work?"."""
        path = _path("chain")
        agent = Agent(Model("gpt-4o", api_key="k"), instructions="Be terse.",
                      stop_when=[step_count(3)], name="worker")
        Chain(steps=[(agent, "out")]).save(path)
        back = Chain.load(path, api_key="k")._steps[0][0]
        self.assertEqual(back.model.name, "gpt-4o")
        self.assertEqual(back.instructions, "Be terse.")
        self.assertEqual([c.name for c in back.stop_when], ["step_count"])


class TestPoolAndAgentDoNotSerialise(unittest.TestCase):
    """Named rather than skipped. A test that does not exist and a capability
    that deliberately does not exist look identical in a test run, and only
    one of them is a defect — so the deliberate one is written down here, with
    what to use instead."""

    def test_a_pool_runs_but_does_not_save(self):
        pool = Pool(runner=_skill(), items=[{"name": "Ada"}, {"name": "Ann"}])
        self.assertEqual(pool.run(), ["ANSWER", "ANSWER"])
        self.assertFalse(hasattr(Pool, "save"),
                         "Pool gained serialisation — give it a lifecycle "
                         "smoke and delete this assertion.")

    def test_an_agent_saves_only_as_a_chain_step(self):
        """Its state is the conversation — a message list the caller can park
        and hand back — so there is nothing for the library to store. The
        configuration still travels, inside a Chain."""
        self.assertFalse(hasattr(Agent, "save"))
        self.assertFalse(hasattr(Agent, "resume"))


if __name__ == "__main__":
    unittest.main()
