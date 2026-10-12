"""
The tenant in force stays in force.

`Chain.run()` given no `context=` called `using(None)`, which set the ambient
context to None for the run: the documented `with using(RunContext("acme")):
chain.run()` lost its tenant, and so did a chain inside a chain step and a
chain in a Pool — and `Model(api_key=callable)` then asked for the key of
tenant None. An agent parked under a tenant kept no context in its document,
so a resume in another process ran under nobody.
"""

import os
import unittest

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from yait_aichain import Agent, Chain, Pool
from yait_aichain.state import InMemoryStore, RunContext, current, using
from yait_aichain.tools import Wait, tool

seen: list = []


@tool(risk="read")
def who(x: str = "") -> str:
    """Say which tenant is current."""
    ctx = current()
    seen.append(ctx.tenant if ctx else None)
    return "ok"


class TestAChainKeepsTheTenant(unittest.TestCase):

    def setUp(self):
        seen.clear()

    def test_a_chain_run_without_context_keeps_the_ambient_one(self):
        with using(RunContext(tenant="acme")):
            Chain(steps=[(who, "out")]).run()
        self.assertEqual(seen, ["acme"])

    def test_a_chain_inside_a_chain_and_a_chain_in_a_pool(self):
        inner = Chain(steps=[(who, "out")])
        with using(RunContext(tenant="acme")):
            Chain(steps=[(inner, "inner")]).run()
            Pool(inner, items=[{}, {}]).run()
        self.assertEqual(seen, ["acme", "acme", "acme"])

    def test_using_none_changes_nothing(self):
        with using(RunContext(tenant="acme")), using(None) as ctx:
            self.assertEqual((ctx.tenant, current().tenant), ("acme", "acme"))

    def test_an_explicit_context_still_wins(self):
        with using(RunContext(tenant="acme")):
            Chain(steps=[(who, "out")]).run(context=RunContext(tenant="globex"))
        self.assertEqual(seen, ["globex"])


class TestAParkedAgentKeepsItsTenant(unittest.TestCase):

    def test_resume_runs_under_the_tenant_it_parked_under(self):
        import json
        from unittest.mock import MagicMock
        from yait_aichain import Model

        def reply(*specs):
            msg = ({"content": None, "tool_calls": [
                       {"id": i, "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}
                       for i, n, a in specs]} if specs else {"content": "done"})
            return json.dumps({"choices": [{"message": msg}],
                               "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}).encode()

        m = Model("gpt-4o", api_key="k")
        answers = iter([reply(("c1", "wait", {})), reply(("c2", "who", {})), reply()])
        m.client._post = MagicMock(side_effect=lambda *a, **k: next(answers))
        m.client._auth_headers = MagicMock(return_value={})
        store = InMemoryStore()
        agent = Agent(m, tools=[Wait(reason="ok?"), who], store=store)
        seen.clear()
        with using(RunContext(tenant="acme")):
            parked = agent.run("go")
        self.assertEqual(store.load(parked.run_id)["context"]["tenant"], "acme")
        agent.resume(parked.run_id, signal={"ok": True})        # outside any context
        self.assertEqual(seen, ["acme"])


if __name__ == "__main__":
    unittest.main()
