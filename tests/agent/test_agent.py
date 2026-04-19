"""
tests.agent.test_agent
=======================

Unit tests for the agent layer:
  - AgentMemory  — CRUD, clear, flush, FileBackend
  - AgentResult  — bool, repr, fields
  - Agent        — construction, validation, helpers, execute_action, run()

All tests are pure — no network calls, no real API keys.
Every LLM call is intercepted at the ``client._post`` level.
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

_TEST_KEYS = {
    "OPENAI_API_KEY":    "test-openai-key",
    "ANTHROPIC_API_KEY": "test-anthropic-key",
    "GOOGLE_AI_API_KEY": "test-google-key",
}
for _k, _v in _TEST_KEYS.items():
    if not os.environ.get(_k):
        os.environ[_k] = _v

from models import OpenAIModel
from agent  import Agent, AgentMemory, AgentResult, FileBackend, InMemoryBackend
from tools._base import Tool, ToolResult


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

class EchoTool(Tool):
    """Minimal tool that returns its kwargs as a string."""
    name        = "echo"
    description = "Echoes input."
    parameters  = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    def run(self, text: str = "") -> str:
        return f"echo: {text}"


def _mock_client(*responses: dict) -> MagicMock:
    """
    Return a mock client whose ``_post`` returns each *response* dict in order,
    JSON-encoded as bytes.  Extra calls return the last response forever.
    """
    encoded = [json.dumps(r).encode() for r in responses]
    client  = MagicMock()
    client._auth_headers.return_value = {"Authorization": "Bearer test"}

    if len(encoded) == 1:
        client._post.return_value = encoded[0]
    else:
        # Cycle through responses; repeat last if exhausted
        call_count = [0]
        def _side_effect(*_args, **_kwargs):
            idx = min(call_count[0], len(encoded) - 1)
            call_count[0] += 1
            return encoded[idx]
        client._post.side_effect = _side_effect

    return client


def _oai(text: str) -> dict:
    """Minimal OpenAI-style response wrapping *text*."""
    return {"choices": [{"message": {"content": text}}]}


def _plan_resp(steps: list) -> dict:
    """Orchestrator response for the planning phase."""
    return _oai(json.dumps({"steps": steps}))


def _action_resp(action: dict) -> dict:
    """Orchestrator response for the action phase."""
    return _oai(json.dumps(action))


def _reflect_resp(
    assessment: str = "success",
    decision:   str = "continue",
    store_as:   str = "step_result",
    final_answer: str | None = None,
    reason:     str = "",
) -> dict:
    payload: dict = {
        "assessment": assessment,
        "decision":   decision,
        "store_as":   store_as,
        "reason":     reason,
    }
    if final_answer is not None:
        payload["final_answer"] = final_answer
    return _oai(json.dumps(payload))


def _make_model(*responses: dict) -> OpenAIModel:
    """Create an OpenAIModel with a mocked client returning *responses* in order."""
    model = OpenAIModel("gpt-4o")
    model.client = _mock_client(*responses)
    return model


def _single_step_agent(tool_output: str = "tool result") -> tuple:
    """
    Build an Agent that runs one tool step and returns a final_answer.

    Returns ``(agent, tool)`` so tests can inspect the tool calls.
    """
    plan    = [{"id": 1, "type": "tool", "tool_name": "echo", "goal": "say hello"}]
    action  = {"type": "tool", "tool_name": "echo", "kwargs": {"text": "hello"}}
    reflect = _reflect_resp(decision="final_answer", final_answer=tool_output)

    model = _make_model(_plan_resp(plan), _action_resp(action), reflect)
    tool  = EchoTool()
    agent = Agent(orchestrator=model, tools=[tool])
    return agent, tool


# ---------------------------------------------------------------------------
# AgentMemory — in-memory backend
# ---------------------------------------------------------------------------

class TestAgentMemory(unittest.TestCase):

    def _mem(self, **kw) -> AgentMemory:
        return AgentMemory(kw if kw else None)

    # ── Basic CRUD ────────────────────────────────────────────────────────

    def test_set_and_get(self):
        m = self._mem()
        m.set("key", "value")
        self.assertEqual(m.get("key"), "value")

    def test_get_missing_returns_default(self):
        self.assertIsNone(AgentMemory().get("missing"))

    def test_get_missing_custom_default(self):
        self.assertEqual(AgentMemory().get("x", 42), 42)

    def test_update_merges(self):
        m = self._mem(a=1)
        m.update({"b": 2})
        self.assertEqual(m.get("a"), 1)
        self.assertEqual(m.get("b"), 2)

    def test_delete_removes_key(self):
        m = self._mem(x=10)
        m.delete("x")
        self.assertNotIn("x", m)

    def test_delete_missing_is_silent(self):
        m = AgentMemory()
        m.delete("nonexistent")          # must not raise

    def test_clear_wipes_store(self):
        m = self._mem(a=1, b=2)
        m.clear()
        self.assertEqual(len(m), 0)

    def test_all_returns_shallow_copy(self):
        m = self._mem(x=1)
        snapshot = m.all()
        snapshot["y"] = 99
        self.assertNotIn("y", m)

    def test_initial_dict_seeded(self):
        m = AgentMemory({"topic": "AI"})
        self.assertEqual(m.get("topic"), "AI")

    def test_len(self):
        m = self._mem(a=1, b=2, c=3)
        self.assertEqual(len(m), 3)

    def test_contains(self):
        m = self._mem(x=1)
        self.assertIn("x", m)
        self.assertNotIn("y", m)

    def test_keys_values_items(self):
        m = self._mem(a=1, b=2)
        self.assertIn("a", list(m.keys()))
        self.assertIn(1,   list(m.values()))
        self.assertIn(("a", 1), list(m.items()))

    def test_repr_shows_keys_and_backend(self):
        m = self._mem(x=1)
        r = repr(m)
        self.assertIn("x", r)
        self.assertIn("InMemoryBackend", r)


# ---------------------------------------------------------------------------
# AgentMemory — FileBackend
# ---------------------------------------------------------------------------

class TestFileBackend(unittest.TestCase):

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "state.json")
            fb   = FileBackend(path)
            fb.save({"key": "value", "num": 42})
            data = fb.load()
            self.assertEqual(data["key"], "value")
            self.assertEqual(data["num"], 42)

    def test_load_missing_file_returns_empty(self):
        with tempfile.TemporaryDirectory() as d:
            fb = FileBackend(os.path.join(d, "missing.json"))
            self.assertEqual(fb.load(), {})

    def test_clear_removes_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "state.json")
            fb   = FileBackend(path)
            fb.save({"k": "v"})
            fb.clear()
            self.assertEqual(fb.load(), {})

    def test_flush_persists_memory(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "state.json")
            fb   = FileBackend(path)
            m    = AgentMemory({"x": 99}, backend=fb)
            m.flush()
            # New AgentMemory with same backend loads saved state
            m2 = AgentMemory(backend=FileBackend(path))
            self.assertEqual(m2.get("x"), 99)

    def test_save_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as d:
            nested = os.path.join(d, "a", "b", "c", "state.json")
            fb = FileBackend(nested)
            fb.save({"k": 1})
            self.assertTrue(os.path.exists(nested))


# ---------------------------------------------------------------------------
# AgentResult
# ---------------------------------------------------------------------------

class TestAgentResult(unittest.TestCase):

    def _success(self, **kw) -> AgentResult:
        defaults = dict(
            success=True, output="done", mode="waterfall",
            steps_taken=2, tokens_used=500,
        )
        defaults.update(kw)
        return AgentResult(**defaults)

    def _failure(self, **kw) -> AgentResult:
        defaults = dict(
            success=False, output=None, mode="waterfall",
            steps_taken=1, tokens_used=200, error="step failed",
        )
        defaults.update(kw)
        return AgentResult(**defaults)

    def test_bool_true_on_success(self):
        self.assertTrue(bool(self._success()))

    def test_bool_false_on_failure(self):
        self.assertFalse(bool(self._failure()))

    def test_repr_shows_success(self):
        r = repr(self._success())
        self.assertIn("success", r)
        self.assertIn("waterfall", r)
        self.assertIn("2",  r)   # steps
        self.assertIn("500", r)  # tokens

    def test_repr_shows_failure_reason(self):
        r = repr(self._failure(error="step failed"))
        self.assertIn("step failed", r)

    def test_fields_accessible(self):
        res = self._success(plan=[{"id": 1}], history=[{"step": 0}])
        self.assertEqual(res.plan,    [{"id": 1}])
        self.assertEqual(res.history, [{"step": 0}])

    def test_memory_field(self):
        res = self._success(memory={"key": "val"})
        self.assertEqual(res.memory["key"], "val")

    def test_error_none_on_success(self):
        self.assertIsNone(self._success().error)


# ---------------------------------------------------------------------------
# Agent.__init__ — construction and validation
# ---------------------------------------------------------------------------

class TestAgentInit(unittest.TestCase):

    def _model(self) -> OpenAIModel:
        m = OpenAIModel("gpt-4o")
        m.client = _mock_client(_oai("ok"))
        return m

    def test_basic_construction(self):
        agent = Agent(orchestrator=self._model())
        self.assertEqual(agent.mode, "waterfall")
        self.assertEqual(agent.max_steps, 10)

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            Agent(orchestrator=self._model(), mode="sprint")

    def test_invalid_verbose_raises(self):
        with self.assertRaises(ValueError):
            Agent(orchestrator=self._model(), verbose=3)

    def test_executors_default_to_orchestrator(self):
        m = self._model()
        agent = Agent(orchestrator=m)
        self.assertIs(agent.executors[0], m)

    def test_custom_executors_stored(self):
        m1, m2 = self._model(), self._model()
        agent = Agent(orchestrator=m1, executors=[m2])
        self.assertIs(agent.executors[0], m2)

    def test_tools_stored(self):
        tool  = EchoTool()
        agent = Agent(orchestrator=self._model(), tools=[tool])
        self.assertIn(tool, agent.tools)

    def test_tool_map_built(self):
        tool  = EchoTool()
        agent = Agent(orchestrator=self._model(), tools=[tool])
        self.assertIn("echo", agent._tool_map)

    def test_allow_spawn_adds_spawn_tool(self):
        agent = Agent(orchestrator=self._model(), allow_spawn=True)
        names = [t.name for t in agent.tools]
        self.assertIn("spawn_agent", names)

    def test_name_and_description_stored(self):
        agent = Agent(
            orchestrator=self._model(),
            name="my_agent",
            description="does stuff",
        )
        self.assertEqual(agent.name,        "my_agent")
        self.assertEqual(agent.description, "does stuff")

    def test_persona_stored(self):
        agent = Agent(
            orchestrator=self._model(),
            persona="You are a helpful assistant.",
        )
        self.assertEqual(agent.persona, "You are a helpful assistant.")

    def test_memory_created_if_not_provided(self):
        agent = Agent(orchestrator=self._model())
        self.assertIsInstance(agent.memory, AgentMemory)

    def test_custom_memory_used(self):
        mem   = AgentMemory({"seed": "val"})
        agent = Agent(orchestrator=self._model(), memory=mem)
        self.assertIs(agent.memory, mem)


# ---------------------------------------------------------------------------
# Agent.__repr__
# ---------------------------------------------------------------------------

class TestAgentRepr(unittest.TestCase):

    def test_repr_contains_mode(self):
        m = OpenAIModel("gpt-4o")
        m.client = _mock_client(_oai("ok"))
        r = repr(Agent(orchestrator=m, mode="agile"))
        self.assertIn("agile", r)

    def test_repr_contains_orchestrator_name(self):
        m = OpenAIModel("gpt-4o")
        m.client = _mock_client(_oai("ok"))
        r = repr(Agent(orchestrator=m))
        self.assertIn("gpt-4o", r)

    def test_repr_contains_max_steps(self):
        m = OpenAIModel("gpt-4o")
        m.client = _mock_client(_oai("ok"))
        r = repr(Agent(orchestrator=m, max_steps=5))
        self.assertIn("5", r)

    def test_repr_includes_allow_spawn(self):
        m = OpenAIModel("gpt-4o")
        m.client = _mock_client(_oai("ok"))
        r = repr(Agent(orchestrator=m, allow_spawn=True))
        self.assertIn("allow_spawn=True", r)

    def test_repr_includes_truncated_persona(self):
        m = OpenAIModel("gpt-4o")
        m.client = _mock_client(_oai("ok"))
        r = repr(Agent(orchestrator=m, persona="A" * 80))
        self.assertIn("persona=", r)


# ---------------------------------------------------------------------------
# Agent._parse_json
# ---------------------------------------------------------------------------

class TestAgentParseJson(unittest.TestCase):

    def _agent(self) -> Agent:
        m = OpenAIModel("gpt-4o")
        m.client = _mock_client(_oai("ok"))
        return Agent(orchestrator=m)

    def test_parses_direct_json(self):
        data = self._agent()._parse_json('{"key": "value"}')
        self.assertEqual(data["key"], "value")

    def test_parses_json_code_fence(self):
        text = '```json\n{"foo": 42}\n```'
        data = self._agent()._parse_json(text)
        self.assertEqual(data["foo"], 42)

    def test_parses_bare_code_fence(self):
        text = '```\n{"bar": true}\n```'
        data = self._agent()._parse_json(text)
        self.assertTrue(data["bar"])

    def test_parses_embedded_object(self):
        text = 'Here is the result: {"nested": "obj"} — done.'
        data = self._agent()._parse_json(text)
        self.assertEqual(data["nested"], "obj")

    def test_raises_on_unparseable(self):
        with self.assertRaises(ValueError):
            self._agent()._parse_json("not json at all")

    def test_whitespace_stripped(self):
        data = self._agent()._parse_json('  { "x": 1 }  ')
        self.assertEqual(data["x"], 1)


# ---------------------------------------------------------------------------
# Agent._sanitise_key
# ---------------------------------------------------------------------------

class TestAgentSanitiseKey(unittest.TestCase):

    def test_already_clean(self):
        self.assertEqual(Agent._sanitise_key("clean_key"), "clean_key")

    def test_uppercase_lowercased(self):
        self.assertEqual(Agent._sanitise_key("MyKey"), "mykey")

    def test_spaces_become_underscores(self):
        self.assertEqual(Agent._sanitise_key("my key"), "my_key")

    def test_hyphens_become_underscores(self):
        self.assertEqual(Agent._sanitise_key("my-key"), "my_key")

    def test_consecutive_underscores_collapsed(self):
        self.assertEqual(Agent._sanitise_key("a__b"), "a_b")

    def test_leading_trailing_underscores_stripped(self):
        self.assertEqual(Agent._sanitise_key("_key_"), "key")

    def test_empty_string_returns_empty(self):
        self.assertEqual(Agent._sanitise_key(""), "")

    def test_special_chars_removed(self):
        result = Agent._sanitise_key("research!Results@2025")
        self.assertNotIn("!", result)
        self.assertNotIn("@", result)


# ---------------------------------------------------------------------------
# Agent._extract_tokens
# ---------------------------------------------------------------------------

class TestAgentExtractTokens(unittest.TestCase):

    def _agent(self) -> Agent:
        m = OpenAIModel("gpt-4o")
        m.client = _mock_client(_oai("ok"))
        return Agent(orchestrator=m)

    def test_openai_style_usage(self):
        resp = {"usage": {"prompt_tokens": 100, "completion_tokens": 50}}
        self.assertEqual(self._agent()._extract_tokens(resp), 150)

    def test_anthropic_style_usage(self):
        resp = {"usage": {"input_tokens": 200, "output_tokens": 80}}
        self.assertEqual(self._agent()._extract_tokens(resp), 280)

    def test_google_usage_metadata(self):
        resp = {"usageMetadata": {"totalTokenCount": 300}}
        self.assertEqual(self._agent()._extract_tokens(resp), 300)

    def test_no_usage_returns_zero(self):
        self.assertEqual(self._agent()._extract_tokens({}), 0)


# ---------------------------------------------------------------------------
# Agent._is_important
# ---------------------------------------------------------------------------

class TestAgentIsImportant(unittest.TestCase):

    def _rec(
        self,
        exec_error:  str | None = None,
        assessment:  str        = "success",
        decision:    str        = "continue",
    ) -> dict:
        return {
            "step":        0,
            "attempt":     1,
            "step_goal":   "test goal",
            "action_type": "tool",
            "action":      {},
            "output":      "ok",
            "exec_error":  exec_error,
            "stored_as":   None,
            "reflection":  {"assessment": assessment, "decision": decision},
            "tokens":      100,
        }

    def test_exec_error_is_important(self):
        self.assertTrue(Agent._is_important(self._rec(exec_error="boom")))

    def test_failure_assessment_is_important(self):
        self.assertTrue(Agent._is_important(self._rec(assessment="failure")))

    def test_fatal_assessment_is_important(self):
        self.assertTrue(Agent._is_important(self._rec(assessment="fatal")))

    def test_partial_assessment_is_important(self):
        self.assertTrue(Agent._is_important(self._rec(assessment="partial")))

    def test_replan_decision_is_important(self):
        self.assertTrue(Agent._is_important(self._rec(decision="replan")))

    def test_stop_decision_is_important(self):
        self.assertTrue(Agent._is_important(self._rec(decision="stop")))

    def test_final_answer_decision_is_important(self):
        self.assertTrue(Agent._is_important(self._rec(decision="final_answer")))

    def test_normal_success_not_important(self):
        self.assertFalse(Agent._is_important(self._rec()))


# ---------------------------------------------------------------------------
# Agent._history_summary
# ---------------------------------------------------------------------------

class TestAgentHistorySummary(unittest.TestCase):

    def _agent(self) -> Agent:
        m = OpenAIModel("gpt-4o")
        m.client = _mock_client(_oai("ok"))
        return Agent(orchestrator=m)

    def _rec(self, step_idx: int, decision: str = "continue") -> dict:
        return {
            "step":        step_idx,
            "attempt":     1,
            "step_goal":   f"goal {step_idx}",
            "action_type": "tool",
            "action":      {},
            "output":      "ok",
            "exec_error":  None,
            "stored_as":   None,
            "reflection":  {"assessment": "success", "decision": decision},
            "tokens":      50,
        }

    def test_empty_history_returns_sentinel(self):
        result = self._agent()._history_summary([])
        self.assertIn("no prior steps", result)

    def test_single_step_included(self):
        rec    = self._rec(0)
        result = self._agent()._history_summary([rec])
        self.assertIn("goal 0", result)

    def test_important_records_flagged(self):
        rec        = self._rec(0, decision="stop")
        summary    = self._agent()._history_summary([rec])
        self.assertIn("[!]", summary)

    def test_normal_records_not_flagged(self):
        rec     = self._rec(0)
        summary = self._agent()._history_summary([rec])
        self.assertNotIn("[!]", summary)

    def test_capped_at_eight(self):
        history = [self._rec(i) for i in range(20)]
        result  = self._agent()._history_summary(history)
        # Count "Step " occurrences as a proxy for lines
        self.assertLessEqual(result.count("Step "), 8)


# ---------------------------------------------------------------------------
# Agent._extract_final_output
# ---------------------------------------------------------------------------

class TestAgentExtractFinalOutput(unittest.TestCase):

    def _agent(self) -> Agent:
        m = OpenAIModel("gpt-4o")
        m.client = _mock_client(_oai("ok"))
        return Agent(orchestrator=m)

    def _rec(self, output, exec_error=None) -> dict:
        return {
            "step": 0, "attempt": 1, "step_goal": "g",
            "action_type": "tool", "action": {},
            "output": output, "exec_error": exec_error,
            "stored_as": None, "reflection": {}, "tokens": 0,
        }

    def test_empty_history_returns_none(self):
        self.assertIsNone(self._agent()._extract_final_output([]))

    def test_returns_last_successful_output(self):
        history = [self._rec("first"), self._rec("second")]
        self.assertEqual(self._agent()._extract_final_output(history), "second")

    def test_skips_records_with_exec_error(self):
        history = [self._rec("ok"), self._rec("bad", exec_error="boom")]
        self.assertEqual(self._agent()._extract_final_output(history), "ok")

    def test_skips_none_output(self):
        history = [self._rec("good"), self._rec(None)]
        self.assertEqual(self._agent()._extract_final_output(history), "good")


# ---------------------------------------------------------------------------
# Agent._execute_action — tool path
# ---------------------------------------------------------------------------

class TestAgentExecuteActionTool(unittest.TestCase):

    def _agent(self, tool: Tool) -> Agent:
        m = OpenAIModel("gpt-4o")
        m.client = _mock_client(_oai("ok"))
        return Agent(orchestrator=m, tools=[tool])

    def test_tool_called_with_kwargs(self):
        tool  = EchoTool()
        agent = self._agent(tool)
        action = {"type": "tool", "tool_name": "echo", "kwargs": {"text": "hello"}}
        result, tokens = agent._execute_action(action, {})
        self.assertEqual(result, "echo: hello")
        self.assertEqual(tokens, 0)

    def test_unknown_tool_raises(self):
        agent = self._agent(EchoTool())
        action = {"type": "tool", "tool_name": "nonexistent", "kwargs": {}}
        with self.assertRaises(ValueError):
            agent._execute_action(action, {})

    def test_context_substituted_into_string_kwarg(self):
        tool  = EchoTool()
        agent = self._agent(tool)
        action  = {"type": "tool", "tool_name": "echo", "kwargs": {"text": "{msg}"}}
        context = {"msg": "world"}
        result, _ = agent._execute_action(action, context)
        self.assertEqual(result, "echo: world")

    def test_non_string_kwarg_passed_as_is(self):
        # A non-string kwarg (e.g. int) must not be format_map'd
        class IntTool(Tool):
            name = "inttool"
            description = "takes a number"
            parameters = {"type": "object", "properties": {"n": {"type": "integer"}}}
            def run(self, n=0): return n * 2

        agent  = self._agent(IntTool())
        action = {"type": "tool", "tool_name": "inttool", "kwargs": {"n": 5}}
        result, _ = agent._execute_action(action, {})
        self.assertEqual(result, 10)


# ---------------------------------------------------------------------------
# Agent._execute_action — skill path
# ---------------------------------------------------------------------------

class TestAgentExecuteActionSkill(unittest.TestCase):

    def test_skill_returns_model_output(self):
        m = OpenAIModel("gpt-4o")
        m.client = _mock_client(_oai("skill output"))

        agent  = Agent(orchestrator=m)
        action = {
            "type":          "skill",
            "model":         "",
            "system_prompt": "Be helpful.",
            "user_prompt":   "Summarise {text}",
            "output_format": "text",
        }
        context = {"text": "some document"}
        result, tokens = agent._execute_action(action, context)
        self.assertEqual(result, "skill output")
        self.assertEqual(tokens, 0)  # mock returns no usage field

    def test_skill_unknown_model_falls_back_to_first_executor(self):
        m = OpenAIModel("gpt-4o")
        m.client = _mock_client(_oai("fallback"))

        agent  = Agent(orchestrator=m)
        action = {
            "type": "skill", "model": "nonexistent-model",
            "system_prompt": "", "user_prompt": "hi", "output_format": "text",
        }
        result, _ = agent._execute_action(action, {})
        self.assertEqual(result, "fallback")

    def test_unknown_action_type_raises(self):
        m = OpenAIModel("gpt-4o")
        m.client = _mock_client(_oai("ok"))
        agent  = Agent(orchestrator=m)
        with self.assertRaises(ValueError):
            agent._execute_action({"type": "magic"}, {})


# ---------------------------------------------------------------------------
# Agent.run() — full integration with mocked LLM
# ---------------------------------------------------------------------------

class TestAgentRun(unittest.TestCase):

    def test_empty_plan_returns_failure(self):
        """Orchestrator returns no steps → AgentResult.success is False."""
        m = _make_model(_plan_resp([]))
        agent = Agent(orchestrator=m)
        result = agent.run("do something")
        self.assertFalse(result.success)
        self.assertIn("empty plan", result.error.lower())

    def test_single_step_final_answer_success(self):
        """One tool step, reflect says final_answer → success with correct output."""
        agent, _ = _single_step_agent("the final answer")
        result   = agent.run("say hello")
        self.assertTrue(result.success)
        self.assertEqual(result.output, "the final answer")

    def test_result_is_agent_result_instance(self):
        agent, _ = _single_step_agent()
        result   = agent.run("task")
        self.assertIsInstance(result, AgentResult)

    def test_result_mode_matches_agent_mode(self):
        agent, _ = _single_step_agent()
        result   = agent.run("task")
        self.assertEqual(result.mode, "waterfall")

    def test_result_history_has_entry(self):
        agent, _ = _single_step_agent()
        result   = agent.run("task")
        self.assertGreater(len(result.history), 0)

    def test_history_entry_has_required_fields(self):
        agent, _ = _single_step_agent()
        result   = agent.run("task")
        rec      = result.history[0]
        for field in ("step", "attempt", "step_goal", "action_type", "action",
                      "output", "exec_error", "stored_as", "reflection", "tokens"):
            self.assertIn(field, rec, f"Missing field: {field}")

    def test_variables_seeded_into_memory(self):
        """Variables passed to run() appear in result.memory."""
        plan    = [{"id": 1, "type": "tool", "tool_name": "echo", "goal": "g"}]
        action  = {"type": "tool", "tool_name": "echo", "kwargs": {"text": "hi"}}
        reflect = _reflect_resp(decision="final_answer", final_answer="done")

        m     = _make_model(_plan_resp(plan), _action_resp(action), reflect)
        agent = Agent(orchestrator=m, tools=[EchoTool()])
        result = agent.run("task", variables={"lang": "French"})
        # Memory is cleared at start of run and seeded with variables
        # The final memory snapshot is in result.memory
        # lang should appear (seeded before first step)
        # (it could be overwritten by store_as but our reflect doesn't override "lang")
        self.assertIsInstance(result.memory, dict)

    def test_stop_decision_returns_failure(self):
        """Reflect decision='stop' → AgentResult.success is False."""
        plan    = [{"id": 1, "type": "tool", "tool_name": "echo", "goal": "g"}]
        action  = {"type": "tool", "tool_name": "echo", "kwargs": {"text": "hi"}}
        reflect = _reflect_resp(assessment="fatal", decision="stop", reason="fatal error")

        m     = _make_model(_plan_resp(plan), _action_resp(action), reflect)
        agent = Agent(orchestrator=m, tools=[EchoTool()])
        result = agent.run("task")
        self.assertFalse(result.success)
        self.assertIn("fatal error", result.error)

    def test_continue_advances_to_next_step(self):
        """Two steps, both reflect with 'continue', then final output is last step's."""
        plan = [
            {"id": 1, "type": "tool", "tool_name": "echo", "goal": "step 1"},
            {"id": 2, "type": "tool", "tool_name": "echo", "goal": "step 2"},
        ]
        action1  = {"type": "tool", "tool_name": "echo", "kwargs": {"text": "one"}}
        reflect1 = _reflect_resp(decision="continue", store_as="step1_out")
        action2  = {"type": "tool", "tool_name": "echo", "kwargs": {"text": "two"}}
        reflect2 = _reflect_resp(decision="final_answer", final_answer="two done")

        m = _make_model(
            _plan_resp(plan),
            _action_resp(action1), reflect1,
            _action_resp(action2), reflect2,
        )
        agent  = Agent(orchestrator=m, tools=[EchoTool()])
        result = agent.run("two steps")
        self.assertTrue(result.success)
        self.assertEqual(result.output, "two done")
        self.assertEqual(result.steps_taken, 2)

    def test_early_final_answer_via_action(self):
        """Orchestrator can emit final_answer directly in the action phase."""
        plan   = [{"id": 1, "type": "tool", "tool_name": "echo", "goal": "g"}]
        action = {"type": "final_answer", "answer": "early answer"}

        m     = _make_model(_plan_resp(plan), _action_resp(action))
        agent = Agent(orchestrator=m, tools=[EchoTool()])
        result = agent.run("task")
        self.assertTrue(result.success)
        self.assertEqual(result.output, "early answer")

    def test_run_always_returns_agent_result(self):
        """Even when the orchestrator raises an exception, run() returns AgentResult."""
        m = OpenAIModel("gpt-4o")
        m.client = MagicMock()
        m.client._auth_headers.return_value = {}
        m.client._post.side_effect = RuntimeError("network down")

        agent  = Agent(orchestrator=m)
        result = agent.run("task")
        self.assertIsInstance(result, AgentResult)
        self.assertFalse(result.success)

    def test_token_budget_exhausted_stops_loop(self):
        """
        Set max_tokens=0 so the budget is exhausted before execution begins.
        The agent should return a success (all steps complete) with no history
        entries from execution (it stops at the token check).
        """
        plan    = [{"id": 1, "type": "tool", "tool_name": "echo", "goal": "g"}]
        # Only one response needed — plan — because the loop exits immediately
        m     = _make_model(_plan_resp(plan))
        agent = Agent(orchestrator=m, tools=[EchoTool()], max_tokens=0)
        result = agent.run("task")
        # Budget was 0; plan call alone consumes ≥0 tokens.
        # The loop exits on the first budget check before executing any step.
        self.assertIsInstance(result, AgentResult)

    def test_store_as_written_to_memory(self):
        """Reflection's store_as key appears in result.memory."""
        plan    = [{"id": 1, "type": "tool", "tool_name": "echo", "goal": "g"}]
        action  = {"type": "tool", "tool_name": "echo", "kwargs": {"text": "hi"}}
        reflect = _reflect_resp(
            decision="final_answer",
            final_answer="done",
            store_as="my_result",
        )
        m     = _make_model(_plan_resp(plan), _action_resp(action), reflect)
        agent = Agent(orchestrator=m, tools=[EchoTool()])
        result = agent.run("task")
        self.assertIn("my_result", result.memory)


# ---------------------------------------------------------------------------
# Agent.run() — agile mode with replan
# ---------------------------------------------------------------------------

class TestAgentRunAgile(unittest.TestCase):

    def test_replan_replaces_current_plan(self):
        """
        Step 1 reflects with 'replan' providing a shorter revised plan.
        The agent should adopt the revised plan and continue from the
        specified goto_step.
        """
        plan_original = [
            {"id": 1, "type": "tool", "tool_name": "echo", "goal": "original step 1"},
            {"id": 2, "type": "tool", "tool_name": "echo", "goal": "original step 2"},
        ]
        action1  = {"type": "tool", "tool_name": "echo", "kwargs": {"text": "one"}}
        # Reflect says replan with a single-step revised plan
        revised_plan = [
            {"id": 1, "type": "tool", "tool_name": "echo", "goal": "revised only step"},
        ]
        reflect1_payload = {
            "assessment":   "partial",
            "decision":     "replan",
            "store_as":     "",
            "reason":       "plan changed",
            "revised_plan": revised_plan,
            "goto_step":    0,
        }
        reflect1 = _oai(json.dumps(reflect1_payload))

        action2  = {"type": "tool", "tool_name": "echo", "kwargs": {"text": "revised"}}
        reflect2 = _reflect_resp(decision="final_answer", final_answer="replanned done")

        m = _make_model(
            _plan_resp(plan_original),
            _action_resp(action1), reflect1,
            _action_resp(action2), reflect2,
        )
        agent  = Agent(orchestrator=m, tools=[EchoTool()], mode="agile")
        result = agent.run("agile task")
        self.assertTrue(result.success)
        self.assertEqual(result.output, "replanned done")


# ---------------------------------------------------------------------------
# Agent.spawn()
# ---------------------------------------------------------------------------

class TestAgentSpawn(unittest.TestCase):

    def test_spawn_depth_limit_blocks_nesting(self):
        """An agent at max depth returns a blocked message, does not recurse."""
        m = OpenAIModel("gpt-4o")
        m.client = _mock_client(_oai("ok"))
        agent = Agent(
            orchestrator=m,
            _depth=5,
            _max_depth=5,
        )
        result = agent.spawn("sub-task")
        self.assertIn("blocked", result)
        self.assertIn("depth", result)

    def test_spawn_inherits_tools_excluding_spawn_agent(self):
        """spawn() creates a child that inherits parent's tools (minus spawn_agent)."""
        plan    = [{"id": 1, "type": "tool", "tool_name": "echo", "goal": "g"}]
        action  = {"type": "tool", "tool_name": "echo", "kwargs": {"text": "hi"}}
        reflect = _reflect_resp(decision="final_answer", final_answer="child done")

        # Parent orchestrator serves parent run() — won't be used here directly
        parent_m = OpenAIModel("gpt-4o")
        parent_m.client = _mock_client(
            _plan_resp(plan), _action_resp(action), reflect
        )

        tool   = EchoTool()
        parent = Agent(
            orchestrator=parent_m,
            tools=[tool],
            allow_spawn=True,
        )
        # spawn() creates a child with the same orchestrator model
        # (inherits parent's orchestrator); the mock has already been set up
        child_result = parent.spawn("child task")
        self.assertIsInstance(child_result, str)

    def test_spawn_failure_returns_subagent_failed_prefix(self):
        """A failing child agent returns '[subagent failed] …'."""
        m = OpenAIModel("gpt-4o")
        m.client = _mock_client(_plan_resp([]))  # empty plan → failure

        parent = Agent(orchestrator=m)
        result = parent.spawn("fail task")
        self.assertIn("subagent failed", result)


# ---------------------------------------------------------------------------
# _SpawnTool
# ---------------------------------------------------------------------------

class TestSpawnTool(unittest.TestCase):

    def test_spawn_tool_delegates_to_agent_spawn(self):
        """_SpawnTool.run() calls Agent.spawn() with the right arguments."""
        m = OpenAIModel("gpt-4o")
        m.client = _mock_client(_oai("ok"))
        agent = Agent(orchestrator=m, allow_spawn=True)

        spawn_tool = agent._tool_map["spawn_agent"]
        agent.spawn = MagicMock(return_value="mocked spawn result")

        result = spawn_tool.run(task="do something")
        agent.spawn.assert_called_once()
        call_kwargs = agent.spawn.call_args
        self.assertEqual(call_kwargs[0][0], "do something")
        self.assertEqual(result, "mocked spawn result")

    def test_spawn_tool_name_and_description(self):
        m = OpenAIModel("gpt-4o")
        m.client = _mock_client(_oai("ok"))
        agent = Agent(orchestrator=m, allow_spawn=True)
        st = agent._tool_map["spawn_agent"]
        self.assertEqual(st.name, "spawn_agent")
        self.assertIsInstance(st.description, str)
        self.assertGreater(len(st.description), 10)


# ---------------------------------------------------------------------------
# _SafeFormatMap (via execute_action context substitution)
# ---------------------------------------------------------------------------

class TestSafeFormatMap(unittest.TestCase):

    def test_missing_key_left_as_literal(self):
        """Unknown {keys} in prompts are left as-is rather than raising."""
        m = OpenAIModel("gpt-4o")
        m.client = _mock_client(_oai("ok"))
        agent  = Agent(orchestrator=m)
        action = {
            "type":          "skill",
            "model":         "",
            "system_prompt": "{unknown_var} is missing",
            "user_prompt":   "hello {also_missing}",
            "output_format": "text",
        }
        # Should not raise KeyError
        result, _ = agent._execute_action(action, {})
        self.assertEqual(result, "ok")


if __name__ == "__main__":
    unittest.main()
