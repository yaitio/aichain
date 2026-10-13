"""
A stream runs in the context it was made in, every step.

A generator runs in whichever context advances it. Starlette advances each
``next()`` in a fresh copy, so the beacon board was gone from an agent's
second step on; a ``skill.stream()`` made inside ``with using(ctx)`` and read
after it resolved its key with no tenant; two streams advanced in turn in one
thread mixed their boards.
"""

import contextvars
import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from yait_aichain.models import Model
from yait_aichain.skills import Skill
from yait_aichain.state import RunContext, using


def _sse(*texts):
    def sse(path, body, headers, *a, **k):
        for t in texts:
            yield {"choices": [{"delta": {"content": t}}]}
        yield {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                                        "total_tokens": 2}}
    return sse


class TestTheContextIsTheMakers(unittest.TestCase):

    def _skill(self, seen):
        def key(ctx):
            seen.append(getattr(ctx, "tenant", None))
            return "k"
        model = Model("gpt-4o", api_key=key)
        model.client._post_sse = MagicMock(side_effect=_sse("a", "b"))
        return Skill(model, prompt="hi")

    def test_a_stream_made_under_a_tenant_reads_after_it_as_that_tenant(self):
        seen = []
        skill = self._skill(seen)
        with using(RunContext(tenant="acme")):
            stream = skill.stream()
        self.assertEqual("".join(stream), "ab")
        self.assertIn("acme", seen)
        self.assertNotIn(None, seen)

    def test_each_step_in_a_fresh_copy_still_sees_the_tenant(self):
        seen = []
        skill = self._skill(seen)
        with using(RunContext(tenant="acme")):
            stream = skill.stream()
        pieces = []
        while True:
            try:
                pieces.append(contextvars.Context().run(next, stream))   # Starlette's way
            except StopIteration:
                break
        self.assertEqual(pieces, ["a", "b"])
        self.assertNotIn(None, seen)


if __name__ == "__main__":
    unittest.main()
