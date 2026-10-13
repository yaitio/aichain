"""
A Pool stops when it says it stops, and refuses what it cannot run.

``on_error="raise"`` raised only after the whole queue had run — a thousand
paid calls after the first failed — and so did Ctrl-C. Items a stop never
started stayed PENDING, so a loop waiting for DONE + FAILED waited for ever.
A plain function as runner was a list of None, ``items=["a"]`` a bare
TypeError, and ``max_flows=0`` failed only in ``run()``.
"""

import os
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from yait_aichain.pool import CANCELLED, DONE, FAILED, PENDING, Pool   # noqa: E402
from yait_aichain.tools._base import Tool                               # noqa: E402


class Counted(Tool):
    name = "counted"
    parameters = {"type": "object", "properties": {"value": {"type": "string"}}}

    def __init__(self):
        self.ran = 0
        self.lock = threading.Lock()

    def run(self, value=None):
        with self.lock:
            self.ran += 1
        if value == "bad":
            raise RuntimeError("first")
        time.sleep(0.01)
        return value


class TestRaiseStops(unittest.TestCase):

    def test_raise_does_not_run_the_queue(self):
        tool = Counted()
        p = Pool(tool, [{"value": "bad"}] + [{"value": str(i)} for i in range(200)],
                 max_flows=2, on_error="raise")
        with self.assertRaisesRegex(RuntimeError, "first"):
            p.run()
        self.assertLess(tool.ran, 20)
        self.assertEqual(p.status[PENDING], 0)
        self.assertGreater(p.status[CANCELLED], 150)

    def test_every_item_ends_in_a_final_status(self):
        p = Pool(Counted(), [{"value": "bad"}] + [{"value": str(i)} for i in range(30)],
                 max_flows=1, on_error="stop")
        p.run()
        s = p.status
        self.assertEqual(s[DONE] + s[FAILED] + s[CANCELLED], 31)


class TestRefusedAtConstruction(unittest.TestCase):

    def test_a_function_is_not_a_runner(self):
        with self.assertRaisesRegex(TypeError, "@tool"):
            Pool(len, [{"x": 1}])

    def test_items_are_dicts(self):
        with self.assertRaisesRegex(TypeError, "item 0"):
            Pool(Counted(), ["a"])
        with self.assertRaises(TypeError):
            Pool(Counted(), "abc")

    def test_max_flows_is_at_least_one(self):
        for bad in (0, -1, True, 2.5):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                Pool(Counted(), [{"value": "a"}], max_flows=bad)


if __name__ == "__main__":
    unittest.main()
