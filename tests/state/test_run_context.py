"""
RunContext is exposed as chain.context during a run, persisted in the run
document, and restored on resume — even from a fresh instance sharing the
store. (1.3.3)
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from chain import Chain
from tools import Wait
from state import InMemoryStore, RunContext, SuspendedResult


class TestRunContext(unittest.TestCase):

    def test_exposed_and_persisted(self):
        store = InMemoryStore()
        chain = Chain(steps=[Wait(reason="hold")], store=store)
        ctx = RunContext(tenant="acme", metadata={"req": "r-1"})

        res = chain.run(variables={}, context=ctx)
        self.assertIsInstance(res, SuspendedResult)
        # available on the instance during the run
        self.assertEqual(chain.context.tenant, "acme")
        self.assertEqual(chain.context.get("req"), "r-1")
        # persisted in the run document
        self.assertEqual(res.document["context"],
                         {"tenant": "acme", "metadata": {"req": "r-1"}})

    def test_restored_on_resume_cross_instance(self):
        store = InMemoryStore()
        c1 = Chain(steps=[Wait(reason="hold")], store=store)
        res = c1.run(variables={}, context=RunContext(tenant="acme", metadata={"k": 1}))

        # a fresh chain sharing only the store resumes without passing context
        c2 = Chain(steps=[Wait(reason="hold")], store=store)
        self.assertIsNone(c2.context)
        c2.resume(res.run_id, signal={"ok": True})
        self.assertEqual(c2.context.tenant, "acme")     # restored from the document
        self.assertEqual(c2.context.get("k"), 1)

    def test_no_context_is_none(self):
        chain = Chain(steps=[Wait(reason="hold")], store=InMemoryStore())
        res = chain.run(variables={})
        self.assertIsNone(chain.context)
        self.assertIsNone(res.document["context"])


if __name__ == "__main__":
    unittest.main()
