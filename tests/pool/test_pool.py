"""
Pool — fan-out of one runner across many items.

The case that motivated this file: ``Pool`` matches item keys against the
tool's declared ``parameters``. A Tool that declares none therefore gets
called with nothing, its ``run()`` raises TypeError for the arguments it
needed, and ``on_error="collect"`` (the default) turns every one of those
into ``None``. The caller receives a full-length list of ``None`` with no
exception and no warning — indistinguishable from "every item legitimately
returned nothing".

That shipped: a cookbook recipe's first two stages silently did nothing, and
the ensemble degenerated to one model answering a prompt full of ``None``.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from yait_aichain.pool import Pool
from yait_aichain.tools._base import Tool


class Echo(Tool):
    """Well-formed: declares what it takes."""
    name        = "echo"
    description = "Echo a value back."
    parameters  = {"type": "object", "properties": {
        "value": {"type": "string"},
    }, "required": ["value"]}

    def run(self, value, options=None):
        return f"got {value}"


class Undeclared(Tool):
    """The trap: run() needs an argument the schema never mentions."""
    name        = "undeclared"
    description = "Needs a value but declares no parameters."
    parameters  = {"type": "object", "properties": {}}

    def run(self, value, options=None):
        return f"got {value}"


class NoArgs(Tool):
    """Declares nothing and needs nothing — legitimately callable with no kwargs."""
    name        = "noargs"
    description = "Takes nothing."
    parameters  = {"type": "object", "properties": {}}

    def run(self, options=None):
        return "fixed"


class Explodes(Tool):
    name        = "explodes"
    description = "Always raises."
    parameters  = {"type": "object", "properties": {"value": {"type": "string"}}}

    def run(self, value, options=None):
        raise RuntimeError(f"boom on {value}")


class TestFanOut(unittest.TestCase):

    def test_runs_every_item_and_preserves_order(self):
        out = Pool(Echo(), items=[{"value": v} for v in "abcd"], max_flows=4).run()
        self.assertEqual(out, ["got a", "got b", "got c", "got d"])

    def test_order_holds_when_workers_finish_out_of_order(self):
        # One flow per item, with the first item slowest: if the pool returned
        # completion order rather than item order, "a" would not be first.
        import time

        class Slow(Tool):
            name       = "slow"
            parameters = {"type": "object", "properties": {
                "value": {"type": "string"}, "delay": {"type": "number"}}}

            def run(self, value, delay=0, options=None):
                time.sleep(delay)
                return value

        items = [{"value": "a", "delay": 0.05}, {"value": "b", "delay": 0},
                 {"value": "c", "delay": 0}]
        self.assertEqual(Pool(Slow(), items=items, max_flows=3).run(),
                         ["a", "b", "c"])

    def test_tool_taking_no_arguments_is_called(self):
        out = Pool(NoArgs(), items=[{}, {}], max_flows=2).run()
        self.assertEqual(out, ["fixed", "fixed"])

    def test_empty_items_rejected(self):
        with self.assertRaises(ValueError):
            Pool(Echo(), items=[])

    def test_bad_on_error_rejected(self):
        with self.assertRaises(ValueError):
            Pool(Echo(), items=[{"value": "a"}], on_error="explode")


class TestUndeclaredParameters(unittest.TestCase):
    """The silent no-op this file exists for."""

    def test_undeclared_parameters_raise_instead_of_returning_none(self):
        pool = Pool(Undeclared(), items=[{"value": "a"}, {"value": "b"}])
        with self.assertRaises(TypeError) as ctx:
            pool.run()
        msg = str(ctx.exception)
        # The message must name the tool, what run() wanted, and what was
        # available — a bare TypeError sends the reader into the pool internals.
        self.assertIn("undeclared", msg)
        self.assertIn("value", msg)
        self.assertIn("parameters", msg)

    def test_does_not_silently_return_a_list_of_none(self):
        # The exact shipped symptom, asserted as the thing that must not happen.
        pool = Pool(Undeclared(), items=[{"value": "a"}, {"value": "b"}])
        try:
            out = pool.run()
        except TypeError:
            return                              # correct behaviour
        self.fail(f"fan-out silently produced {out!r} instead of raising")

    def test_no_items_variables_means_no_complaint(self):
        # Nothing was passed, so nothing was dropped — a tool that needs no
        # arguments must not be second-guessed.
        self.assertEqual(Pool(NoArgs(), items=[{}]).run(), ["fixed"])


class TestErrorPolicy(unittest.TestCase):

    def test_collect_turns_a_failing_item_into_none(self):
        out = Pool(Explodes(), items=[{"value": "a"}], on_error="collect").run()
        self.assertEqual(out, [None])

    def test_raise_propagates(self):
        with self.assertRaises(Exception):
            Pool(Explodes(), items=[{"value": "a"}], on_error="raise").run()

    def test_skip_keeps_the_slot_but_warns(self):
        # "skip" does not shorten the list — it leaves None like "collect" and
        # adds a warning. Order is positional, so dropping a slot would
        # silently re-align every later result with the wrong item.
        items = [{"value": "a"}, {"value": "b"}]
        with self.assertWarns(RuntimeWarning):
            out = Pool(Explodes(), items=items, on_error="skip").run()
        self.assertEqual(out, [None, None])


if __name__ == "__main__":
    unittest.main()
