"""
A wrong name is refused with the right one named.

The names come from `evals/agent_builds`: what a model actually wrote when it
had only `SKILL.md` and `llms.txt`. Each test here would have seen Python's
bare `TypeError`, `AttributeError`, `ModuleNotFoundError` or an unrelated
crash before the hints existed.
"""

import importlib
import inspect
import unittest

from yait_aichain import Agent, Chain, ChainResult, Model, Pool, Skill, _hints
from yait_aichain.agent import AgentResult
from yait_aichain.state import SuspendedResult


def _model():
    return Model("gpt-5.4-mini", api_key="test-key")


class TestKeywords(unittest.TestCase):

    def test_no_hint_names_a_real_parameter(self):
        for owner, cls in (("Model", Model), ("Skill", Skill), ("Agent", Agent),
                           ("Chain", Chain), ("Pool", Pool)):
            with self.subTest(owner=owner):
                real = set(inspect.signature(cls.__init__).parameters)
                self.assertFalse(real & set(_hints.KEYWORDS[owner]))

    def test_the_model_options_are_the_librarys(self):
        from yait_aichain.models._options import UNIVERSAL_OPTIONS
        self.assertEqual(set(_hints.MODEL_OPTIONS), set(UNIVERSAL_OPTIONS))

    def test_the_signature_is_unchanged(self):
        self.assertIn("stop_when", inspect.signature(Agent.__init__).parameters)
        self.assertIn("prompt", inspect.signature(Skill.__init__).parameters)

    def test_removed_agent_arguments_name_their_replacement(self):
        with self.assertRaisesRegex(TypeError, r"max_steps.*stop_when=\[step_count\(n\)\]"):
            Agent(_model(), max_steps=5)
        with self.assertRaisesRegex(TypeError, r"orchestrator.*team="):
            Agent(_model(), orchestrator=_model())
        with self.assertRaisesRegex(TypeError, r"memory.*conversation"):
            Agent(_model(), memory=[])

    def test_a_sampling_option_points_at_the_model(self):
        with self.assertRaisesRegex(TypeError, r"options=\{'temperature'"):
            Model("gpt-5.4-mini", temperature=0.2)
        with self.assertRaisesRegex(TypeError, r"Skill\(Model\(name, options=\{'temperature'"):
            Skill(_model(), prompt="hi", temperature=0.2)

    def test_a_system_prompt_on_a_skill_shows_the_message_form(self):
        with self.assertRaisesRegex(TypeError, r"'role': 'system'"):
            Skill(_model(), prompt="hi", system="Be brief.")

    def test_chain_and_pool_policies_are_told_apart(self):
        with self.assertRaisesRegex(TypeError, r"on_step_error="):
            Chain(steps=[(Skill(_model(), prompt="hi"), "x")], on_error="skip")
        with self.assertRaisesRegex(TypeError, r"max_flows="):
            Pool(Skill(_model(), prompt="{x}"), items=[{"x": "1"}], concurrency=3)

    def test_an_unknown_name_without_a_hint_is_plain_python(self):
        with self.assertRaises(TypeError) as ctx:
            Agent(_model(), frobnicate=1)
        self.assertNotIn("—", str(ctx.exception))


class TestMessages(unittest.TestCase):

    def test_content_is_refused_with_parts_named(self):
        with self.assertRaisesRegex(ValueError, r"messages\[0\] has 'content'.*'parts'"):
            Skill(_model(), input={"messages": [{"role": "user", "content": "hi"}]})

    def test_an_assistant_turn_with_content_is_not_a_generate_marker(self):
        with self.assertRaisesRegex(ValueError, r"messages\[1\] has 'content'"):
            Skill(_model(), input={"messages": [{"role": "user", "parts": ["hi"]},
                                                {"role": "assistant", "content": "hello"}]})

    def test_a_bare_list_of_messages_names_the_wrapper(self):
        with self.assertRaisesRegex(ValueError, r"input=\{'messages'"):
            Skill(_model(), input=[{"role": "user", "parts": ["hi"]}])

    def test_the_generate_here_marker_still_works(self):
        Skill(_model(), input={"messages": [{"role": "user", "parts": ["a"]},
                                            {"role": "assistant"},
                                            {"role": "user", "parts": ["b"]}]})


class TestAttributes(unittest.TestCase):

    def test_agent_methods_from_other_libraries(self):
        agent = Agent(_model())
        with self.assertRaisesRegex(AttributeError, r"'chat'.*agent\.run\(task\)"):
            agent.chat("hi")
        with self.assertRaisesRegex(AttributeError, r"'resume'.*Chain pauses"):
            agent.resume()
        self.assertFalse(hasattr(agent, "chat"))
        self.assertTrue(hasattr(agent, "run"))

    def test_skill_invoke(self):
        with self.assertRaisesRegex(AttributeError, r"'invoke'.*skill\.run"):
            Skill(_model(), prompt="hi").invoke({})

    def test_a_paused_run_points_at_its_document(self):
        paused = SuspendedResult(run_id="r1", document={"variables": {"draft": "x"}})
        with self.assertRaisesRegex(AttributeError, r"'variables'.*document\['variables'\]"):
            paused.variables
        with self.assertRaisesRegex(AttributeError, r"'output'.*chain\.resume"):
            paused.output

    def test_results_name_their_output(self):
        agent_result = AgentResult(success=True, output="42", mode="agile", steps_taken=1, tokens_used=0)
        with self.assertRaisesRegex(AttributeError, r"'text'.*\.output"):
            agent_result.text
        with self.assertRaisesRegex(AttributeError, r"'content'.*\.output"):
            ChainResult(output="x", success=True).content

    def test_an_unknown_attribute_without_a_hint_is_plain(self):
        with self.assertRaises(AttributeError) as ctx:
            Agent(_model()).frobnicate
        self.assertNotIn("—", str(ctx.exception))


class TestModules(unittest.TestCase):

    def test_a_module_that_does_not_exist_names_where_its_contents_live(self):
        with self.assertRaisesRegex(ModuleNotFoundError, r"yait_aichain.hooks.*from yait_aichain import Tracer"):
            importlib.import_module("yait_aichain.hooks")
        with self.assertRaisesRegex(ModuleNotFoundError, r"yait_aichain.tracing.*Tracer"):
            importlib.import_module("yait_aichain.tracing")

    def test_an_unhinted_missing_module_is_plain(self):
        with self.assertRaises(ModuleNotFoundError) as ctx:
            importlib.import_module("yait_aichain.nothing_like_this")
        self.assertNotIn("—", str(ctx.exception))

    def test_no_hinted_module_exists(self):
        import importlib.util
        for name in _hints.MODULES:
            with self.subTest(module=name):
                with self.assertRaises(ModuleNotFoundError):
                    importlib.util.find_spec(name)


class TestAgentStep(unittest.TestCase):

    def test_a_missing_task_says_how_to_provide_it(self):
        chain = Chain(steps=[(Agent(_model()), "answer")])
        with self.assertRaisesRegex(Exception, r"store the previous step under the key 'task'"):
            chain.run(variables={"question": "Why is the sky blue?"})


if __name__ == "__main__":
    unittest.main()
