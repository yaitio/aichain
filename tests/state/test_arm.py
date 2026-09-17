"""
A run that exists and has taken no step — the subscription.

`arm()` is the third thing that produces a `SuspendedResult`, beside a `Wait`
step and a tool the model called: a run created without being started. It has a
`run_id`, a document in the store, the variables it was seeded with, and a
declared shape for the signal that will start it — which is what a trigger
waits on and what a person is shown a form for. Whatever brings the signal, it
arrives the same way: `resume(run_id, signal)`.

What is pinned here:

  * the parked run is durable and falsy, and carries `reason`, `resume_with`
    and the scheduler `hint` untouched — the library still schedules nothing;
  * for a Chain the signal **seeds the variables**, because nothing has run and
    there is no suspended tool to hand it to;
  * for an Agent the signal **is the task**, as a string or under `"task"`;
  * a signal of the wrong shape is refused with the shape it was armed with;
  * resuming twice raises `KeyError`, so an at-least-once trigger is safe.
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from yait_aichain.agent import Agent
from yait_aichain.chain import Chain
from yait_aichain.models import Model
from yait_aichain.state import FileStore, InMemoryStore, SuspendedResult
from yait_aichain.tools._base import Tool


class Shout(Tool):
    name = "shout"
    description = "Upper-case the text."
    risk = "read"
    parameters = {"type": "object", "properties": {"text": {"type": "string"}},
                  "required": ["text"]}

    def run(self, text, options=None):
        return text.upper()


def _model(text="done"):
    m = Model("gpt-4o", api_key="k")
    body = json.dumps({
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }).encode()
    m.sent = []

    def side(path, b, h, *a, **k):
        m.sent.append(b)
        return body

    m.client._post = MagicMock(side_effect=side)
    m.client._auth_headers = MagicMock(return_value={})
    return m


class TestAnArmedChain(unittest.TestCase):

    def setUp(self):
        self.store = InMemoryStore()
        self.chain = Chain(steps=[(Shout(), "shout")], store=self.store)

    def test_it_is_parked_before_its_first_step(self):
        armed = self.chain.arm(reason="waiting for an order",
                               resume_with={"text": "str"},
                               hint={"on": "order.created"})

        self.assertIsInstance(armed, SuspendedResult)
        self.assertFalse(armed)
        self.assertEqual(armed.awaiting["reason"], "waiting for an order")
        self.assertEqual(armed.awaiting["resume_with"], {"text": "str"})
        # Recorded for whatever does the waking; the engine schedules nothing.
        self.assertEqual(armed.document["steps"][0]["suspend"]["hint"],
                         {"on": "order.created"})
        self.assertIsNotNone(self.store.load(armed.run_id))

    def test_the_signal_seeds_the_variables(self):
        armed  = self.chain.arm(resume_with={"text": "str"})
        result = self.chain.resume(armed.run_id, signal={"text": "new office hours"})

        self.assertTrue(result.success)
        self.assertEqual(result.output, "NEW OFFICE HOURS")
        self.assertEqual(result["shout"], "NEW OFFICE HOURS")

    def test_variables_given_when_arming_survive(self):
        armed  = self.chain.arm(variables={"text": "seeded at arming"})
        result = self.chain.resume(armed.run_id, signal={})
        self.assertEqual(result.output, "SEEDED AT ARMING")

    def test_a_signal_that_is_not_a_dict_is_refused_with_the_shape(self):
        armed = self.chain.arm(resume_with={"text": "str"})
        with self.assertRaises(TypeError) as caught:
            self.chain.resume(armed.run_id, signal="new office hours")
        self.assertIn("'text': 'str'", str(caught.exception))

    def test_a_second_delivery_is_a_key_error(self):
        armed = self.chain.arm()
        self.chain.resume(armed.run_id, signal={"text": "x"})
        with self.assertRaises(KeyError):
            self.chain.resume(armed.run_id, signal={"text": "x"})

    def test_another_process_resumes_it(self):
        """The trigger needs the run id, the store and the signal — not the
        object that armed it."""
        with tempfile.TemporaryDirectory() as tmp:
            armed = Chain(steps=[(Shout(), "shout")],
                          store=FileStore(tmp)).arm(resume_with={"text": "str"})
            fresh = Chain(steps=[(Shout(), "shout")], store=FileStore(tmp))
            self.assertEqual(fresh.resume(armed.run_id,
                                          signal={"text": "hello"}).output,
                             "HELLO")

    def test_a_chain_with_no_steps_has_nothing_to_arm(self):
        with self.assertRaises(ValueError):
            Chain(steps=[], store=InMemoryStore()).arm()


class TestAnArmedAgent(unittest.TestCase):

    def test_the_signal_is_the_task(self):
        agent = Agent(_model("refunded"), store=InMemoryStore())
        armed = agent.arm(reason="waiting for a support email",
                          resume_with={"task": "str"},
                          hint={"on": "email.received"})
        self.assertFalse(armed)

        result = agent.resume(armed.run_id, signal={"task": "refund order 4417"})
        self.assertTrue(result.success)
        self.assertEqual(result.output, "refunded")
        self.assertIn("refund order 4417", json.dumps(agent.model.sent[0]))

    def test_a_bare_string_is_the_task_too(self):
        agent = Agent(_model("ok"), store=InMemoryStore())
        armed = agent.arm()
        self.assertTrue(agent.resume(armed.run_id, signal="do the thing").success)

    def test_other_keys_seed_the_conversation(self):
        agent = Agent(_model("ok"), store=InMemoryStore())
        armed = agent.arm(variables={"tone": "formal"})
        agent.resume(armed.run_id, signal={"task": "write it", "city": "Lisbon"})

        sent = json.dumps(agent.model.sent[0])
        self.assertIn("Lisbon", sent)
        self.assertIn("formal", sent)

    def test_a_signal_without_a_task_is_refused_with_the_shape(self):
        agent = Agent(_model(), store=InMemoryStore())
        armed = agent.arm(reason="waiting for an email",
                          resume_with={"task": "str"})
        with self.assertRaises(ValueError) as caught:
            agent.resume(armed.run_id, signal={"subject": "hi"})
        message = str(caught.exception)
        self.assertIn("'task': 'str'", message)
        self.assertIn("waiting for an email", message)

    def test_arming_needs_a_store(self):
        with self.assertRaises(ValueError) as caught:
            Agent(_model()).arm()
        self.assertIn("store=", str(caught.exception))

    def test_a_second_delivery_is_a_key_error(self):
        agent = Agent(_model(), store=InMemoryStore())
        armed = agent.arm()
        agent.resume(armed.run_id, signal="go")
        with self.assertRaises(KeyError):
            agent.resume(armed.run_id, signal="go")

    def test_another_process_resumes_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            armed = Agent(_model("ok"), store=FileStore(tmp)).arm()
            fresh = Agent(_model("ok"), store=FileStore(tmp))
            self.assertTrue(fresh.resume(armed.run_id, signal="go").success)


if __name__ == "__main__":
    unittest.main()
