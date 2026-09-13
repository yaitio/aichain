"""
Chain's contract, beyond variable flow: loading safely, saving an agent, hooks,
overrides, the budget it lends and the context it carries.

`tests/chain/test_chain.py` checks the spine — construction, variable flow,
history — and holds 28 tests. The rest of what a chain promises was either
tested elsewhere one property at a time or not at all: that an untrusted file
cannot name `subprocess.Popen`, that an agent step survives save/load with its
stop conditions, that hooks see step boundaries, that a lent budget is given
back. Coverage showed Chain at 81%, and the uncovered lines were exactly these.
"""

import os
import sys
import tempfile
import textwrap
import unittest
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from yait_aichain import Budget, Chain, Model, Tracer                  # noqa: E402
from yait_aichain.agent import Agent as RealAgent, check, step_count   # noqa: E402
from yait_aichain.chain._chain import _require_safe_tool_class         # noqa: E402
from yait_aichain.state import RunContext, current                     # noqa: E402
from yait_aichain.tools import VectorChunkTool, Wait                   # noqa: E402
from yait_aichain.tools._base import Tool                              # noqa: E402

try:
    import yaml  # noqa: F401
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


class Upper(Tool):
    name = "upper"
    parameters = {"type": "object", "properties": {"text": {"type": "string"}}}

    def run(self, text=None, options=None):
        return text.upper()


class Boom(Tool):
    name = "boom"
    parameters = {"type": "object", "properties": {"text": {"type": "string"}}}

    def run(self, text=None, options=None):
        raise RuntimeError("step broke")


class Spends(Tool):
    """Records the budget it held while running."""
    name = "spends"
    parameters = {"type": "object", "properties": {"text": {"type": "string"}}}

    def __init__(self):
        self.max_cost = None
        self.held = []

    def run(self, text=None, options=None):
        self.held.append(self.max_cost)
        return text


class Tenant(Tool):
    name = "tenant"
    parameters = {"type": "object", "properties": {"text": {"type": "string"}}}

    def run(self, text=None, options=None):
        return current().tenant if current() else None


class Records:
    """A Skill-shaped step: anything that is neither a Tool nor an Agent."""
    name = "records"

    def __init__(self):
        self.seen = None
        self.last_usage = None

    def run(self, variables=None):
        self.seen = dict(variables)
        return "recorded"


class _Result:
    def __init__(self, ok, output="", tokens=0, error=None):
        self.ok, self.output, self.tokens_used, self.error = ok, output, tokens, error

    def __bool__(self):
        return self.ok


class Agent:
    """Detected as an agent step by class name, as the real one is."""
    name = "fake-agent"

    def __init__(self, ok=True):
        self.ok = ok

    def run(self, task, variables=None):
        return _Result(self.ok, output=f"answered {task}", tokens=11, error="gave up")


class TestUntrustedFilesCannotNameArbitraryCode(unittest.TestCase):

    def test_a_class_outside_the_package_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            _require_safe_tool_class("subprocess.Popen", trusted=False)
        self.assertIn("trusted=True", str(ctx.exception))

    def test_a_lookalike_package_prefix_is_refused(self):
        with self.assertRaises(ValueError):
            _require_safe_tool_class("yait_aichain_evil.tools.Steal", trusted=False)

    def test_the_package_itself_is_allowed(self):
        _require_safe_tool_class("yait_aichain.tools.vectordb._chunk.VectorChunkTool", trusted=False)

    def test_trusted_allows_anything(self):
        _require_safe_tool_class("subprocess.Popen", trusted=True)


@unittest.skipUnless(HAS_YAML, "pyyaml not installed")
class TestLoad(unittest.TestCase):

    def _file(self, body):
        f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        f.write(textwrap.dedent(body)); f.close()
        return f.name

    def test_a_tool_step_naming_foreign_code_is_refused(self):
        path = self._file("""
            steps:
              - kind: tool
                output_key: out
                tool: {class: subprocess.Popen, init_args: {args: ["true"]}}
        """)
        with self.assertRaises(ValueError):
            Chain.load(path)

    def test_an_agent_step_naming_a_foreign_tool_is_refused(self):
        path = self._file("""
            steps:
              - kind: agent
                output_key: out
                agent: {model: gpt-4o, tools: [subprocess.Popen]}
        """)
        with self.assertRaises(ValueError):
            Chain.load(path)

    def test_an_unknown_step_kind_is_named(self):
        path = self._file("steps:\n  - kind: telepathy\n    output_key: out\n")
        with self.assertRaises(ValueError) as ctx:
            Chain.load(path)
        self.assertIn("telepathy", str(ctx.exception))

    def test_an_agent_step_saved_without_a_model_says_where_to_look(self):
        path = self._file("""
            steps:
              - kind: agent
                output_key: out
                agent: {orchestrator: null}
        """)
        with self.assertRaises(ValueError) as ctx:
            Chain.load(path)
        self.assertIn("2.6.1", str(ctx.exception))

    def test_a_missing_tool_class_is_an_import_error(self):
        path = self._file("""
            steps:
              - kind: tool
                output_key: out
                tool: {class: yait_aichain.tools.NoSuchTool}
        """)
        with self.assertRaises(ImportError):
            Chain.load(path)


@unittest.skipUnless(HAS_YAML, "pyyaml not installed")
class TestAnAgentStepSurvivesSaveAndLoad(unittest.TestCase):

    def test_round_trip(self):
        agent = RealAgent(Model("gpt-4o", api_key="k"),
                          tools=[VectorChunkTool()],
                          stop_when=[step_count(3), check(lambda s: True, name="done")],
                          instructions="be brief", name="researcher",
                          description="finds things")
        chain = Chain(steps=[(agent, "answer")], name="with-agent")
        path = os.path.join(tempfile.mkdtemp(), "chain.yaml")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            chain.save(path)
        # The closure-backed `check` cannot be written down, and says so.
        self.assertTrue(any("cannot be serialised" in str(w.message) for w in caught))

        loaded = Chain.load(path, api_key="k")
        step = loaded._steps[0][0]
        self.assertEqual(step.model.name, "gpt-4o")
        self.assertEqual(step.instructions, "be brief")
        self.assertEqual((step.name, step.description), ("researcher", "finds things"))
        self.assertEqual(len(step.stop_when), 1)
        self.assertEqual([type(t).__name__ for t in step.tools], ["VectorChunkTool"])


class TestStepShapes(unittest.TestCase):

    def test_a_three_tuple_needs_a_dict_input_map(self):
        with self.assertRaises(ValueError):
            Chain(steps=[(Upper(), "out", "text")])

    def test_a_four_tuple_needs_dict_options(self):
        with self.assertRaises(ValueError):
            Chain(steps=[(Upper(), "out", {}, "fast")])

    def test_a_five_tuple_is_refused(self):
        with self.assertRaises(ValueError):
            Chain(steps=[(Upper(), "out", {}, {}, "extra")])

    def test_the_output_key_must_be_a_string(self):
        with self.assertRaises(ValueError):
            Chain(steps=[(Upper(), 3)])

    def test_repr_shows_kind_name_and_a_non_default_policy(self):
        r = repr(Chain(steps=[(Upper(), "out")], name="shout", on_step_error="skip"))
        for part in ("[T]upper→out", "name='shout'", "on_step_error='skip'"):
            self.assertIn(part, r)

    def test_a_wait_without_a_store_warns_at_construction(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Chain(steps=[(Upper(), "out"), Wait(reason="approve?")])
        self.assertTrue(any("cannot be resumed from another process" in str(w.message)
                            for w in caught))


class TestHooksSeeStepBoundaries(unittest.TestCase):

    def test_started_and_ended_per_step(self):
        tracer = Tracer()
        Chain(steps=[(Upper(), "text"), (Upper(), "again", {"text": "text"})],
              hooks=[tracer]).run({"text": "hi"})
        self.assertEqual([(e.type, e.step) for e in tracer.events],
                         [("step.started", 0), ("step.ended", 0),
                          ("step.started", 1), ("step.ended", 1)])
        self.assertEqual(tracer.events[0].name, "upper")

    def test_a_failed_step_starts_and_never_ends(self):
        tracer = Tracer()
        Chain(steps=[(Boom(), "x"), (Upper(), "y")], hooks=[tracer],
              on_step_error="collect").run({"text": "hi"})
        self.assertEqual([(e.type, e.step) for e in tracer.events],
                         [("step.started", 0), ("step.started", 1), ("step.ended", 1)])


class TestPerCallPolicy(unittest.TestCase):

    def test_an_unknown_override_is_refused(self):
        with self.assertRaises(ValueError):
            Chain(steps=[(Upper(), "out")]).run({"text": "a"}, on_step_error="ignore")

    def test_an_override_applies_to_that_call_only(self):
        chain = Chain(steps=[(Boom(), "x"), (Upper(), "y")])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertEqual(chain.run({"text": "a"}, on_step_error="skip"), "A")
        with self.assertRaises(RuntimeError):
            chain.run({"text": "a"})

    def test_stop_returns_the_last_good_output(self):
        chain = Chain(steps=[(Upper(), "text"), (Boom(), "x"), (Upper(), "y")],
                      on_step_error="stop")
        self.assertEqual(chain.run({"text": "a"}), "A")
        self.assertEqual(chain.history[-1]["failure"]["type"], "RuntimeError")

    def test_resuming_an_unknown_run_is_a_key_error(self):
        with self.assertRaises(KeyError):
            Chain(steps=[(Upper(), "out")]).resume("run-does-not-exist")


class TestTheBudgetIsLentAndGivenBack(unittest.TestCase):

    def test_a_step_holds_the_chain_budget_only_while_it_runs(self):
        step, budget = Spends(), Budget(0.5)
        Chain(steps=[(step, "out")], max_cost=budget).run({"text": "a"})
        self.assertIs(step.held[0], budget)
        self.assertIsNone(step.max_cost)

    def test_it_is_given_back_even_when_the_run_raises(self):
        step = Spends()
        with self.assertRaises(RuntimeError):
            Chain(steps=[(step, "out"), (Boom(), "x")], max_cost=0.5).run({"text": "a"})
        self.assertIsNone(step.max_cost)


class TestContext(unittest.TestCase):

    def test_a_step_sees_the_run_context_and_it_ends_with_the_run(self):
        chain = Chain(steps=[(Tenant(), "who")])
        self.assertEqual(chain.run({"text": "a"}, context=RunContext(tenant="acme")), "acme")
        self.assertEqual(chain.context.tenant, "acme")
        self.assertIsNone(current())


class TestAgentAndSkillSteps(unittest.TestCase):

    def test_an_agent_step_answers_and_its_tokens_are_counted(self):
        chain = Chain(steps=[(Agent(), "answer")])
        self.assertEqual(chain.run({"task": "summarise"}), "answered summarise")
        self.assertEqual(chain.last_usage.total_tokens, 11)

    def test_an_agent_step_with_no_task_says_which_variable(self):
        with self.assertRaises(ValueError) as ctx:
            Chain(steps=[(Agent(), "answer")]).run({"topic": "x"})
        self.assertIn("'task'", str(ctx.exception))

    def test_a_failed_agent_result_is_an_error_not_an_empty_answer(self):
        with self.assertRaises(RuntimeError) as ctx:
            Chain(steps=[(Agent(ok=False), "answer")]).run({"task": "x"})
        self.assertIn("gave up", str(ctx.exception))

    def test_a_skill_step_input_map_renames_a_variable(self):
        skill = Records()
        Chain(steps=[(skill, "out", {"text": "draft"})]).run({"draft": "hello"})
        self.assertEqual(skill.seen["text"], "hello")
        self.assertEqual(skill.seen["draft"], "hello")

    def test_accumulated_is_a_copy(self):
        chain = Chain(steps=[(Upper(), "out")])
        chain.run({"text": "a"})
        chain.accumulated["out"] = "tampered"
        self.assertEqual(chain.accumulated["out"], "A")


if __name__ == "__main__":
    unittest.main()
