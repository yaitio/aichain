"""
tests.test_audit_fixes
======================

Regression tests for the Stage-1 audit fixes (June 2026).
Each test pins a bug that previously made a feature dead-on-arrival:

1. Broken absolute imports in Chain.load() and the Qwen embedder/reranker.
2. Agent steps in Chain never detected (``_is_agent`` checked ``__module__``).
3. POST requests excluded from the urllib3 retry policy.
4. ``delete(ids=[])`` interpreted as a full-collection wipe by backends.
5. ``Agent.run()`` wiping persistent memory (``clear()`` erased the backend).
6. False ``success=True`` on token-budget exhaustion / exhausted retries.
7. ``str.format_map`` raising on literal braces (JSON examples) in prompts.

All tests are pure — no network calls, no real API keys.
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_TEST_KEYS = {
    "OPENAI_API_KEY":    "test-openai-key",
    "DASHSCOPE_API_KEY": "test-dashscope-key",
}
for _k, _v in _TEST_KEYS.items():
    if not os.environ.get(_k):
        os.environ[_k] = _v


# ---------------------------------------------------------------------------
# Fix 1 — broken absolute imports
# ---------------------------------------------------------------------------

class TestBrokenImports(unittest.TestCase):

    def test_chain_load_roundtrip(self):
        """Chain.save() → Chain.load() must not raise ModuleNotFoundError."""
        try:
            import yaml  # noqa: F401
        except ImportError:
            self.skipTest("PyYAML not installed")

        import tempfile
        from yait_aichain.models import Model
        from yait_aichain.skills import Skill
        from yait_aichain.chain import Chain

        skill = Skill(
            model = Model("gpt-4o", api_key="test-key"),
            input = {"messages": [{"role": "user", "parts": ["Say {word}"]}]},
        )
        chain = Chain(steps=[(skill, "result", {}, {})])

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "chain.yaml")
            chain.save(path)
            loaded = Chain.load(path, api_key="test-key")

        self.assertEqual(len(loaded._steps), 1)

    def test_qwen_embedder_internal_import(self):
        """EmbeddingQwen._embed_request must resolve its lazy import."""
        from yait_aichain.tools.embedding._qwen import EmbeddingQwen

        emb = EmbeddingQwen("text-embedding-v4", api_key="test-key")
        emb._http = MagicMock()
        emb._http.request.return_value = MagicMock(
            status=200,
            data=json.dumps({
                "data":  [{"index": 0, "embedding": [0.1, 0.2]}],
                "usage": {"total_tokens": 3},
            }).encode(),
        )
        # Before the fix this raised ModuleNotFoundError: 'clients'
        vectors, tokens, _meta = emb._embed_chunk(
            ["hello"], input_type=None, dimensions=None
        )
        self.assertEqual(vectors, [[0.1, 0.2]])

    def test_qwen_reranker_internal_import(self):
        """RerankQwen._rerank_request must resolve its lazy import."""
        from yait_aichain.tools.reranking._qwen import RerankQwen

        rr = RerankQwen("gte-rerank-v2", api_key="test-key")
        rr._http = MagicMock()
        rr._http.request.return_value = MagicMock(
            status=200,
            data=json.dumps({
                "output": {"results": [
                    {"index": 0, "relevance_score": 0.9,
                     "document": {"text": "doc"}},
                ]},
                "usage": {"total_tokens": 3},
            }).encode(),
        )
        results, _tokens, _meta = rr._rerank_request("q", ["doc"], top_n=None)
        self.assertEqual(len(results), 1)


# ---------------------------------------------------------------------------
# Fix 2 — Agent detection in Chain
# ---------------------------------------------------------------------------

class TestAgentDetection(unittest.TestCase):

    def test_real_agent_detected(self):
        from yait_aichain.agent import Agent
        from yait_aichain.chain._chain import _is_agent

        agent = Agent.__new__(Agent)   # no orchestrator needed for the check
        self.assertTrue(_is_agent(agent))

    def test_user_subclass_detected(self):
        from yait_aichain.agent import Agent
        from yait_aichain.chain._chain import _is_agent

        class MyAgent(Agent):
            pass

        self.assertTrue(_is_agent(MyAgent.__new__(MyAgent)))

    def test_skill_not_detected_as_agent(self):
        from yait_aichain.chain._chain import _is_agent

        class NotAnAgent:
            pass

        self.assertFalse(_is_agent(NotAnAgent()))

    def test_chain_classifies_agent_step(self):
        """Chain.__init__ must register an Agent runner with kind='agent'."""
        from yait_aichain.agent import Agent
        from yait_aichain.chain import Chain

        agent = Agent.__new__(Agent)
        agent.name = "test-agent"
        chain = Chain(steps=[(agent, "out", {}, {})])
        kinds = [step[3] for step in chain._steps]
        self.assertEqual(kinds, ["agent"])


# ---------------------------------------------------------------------------
# Fix 3 — retry policy covers POST
# ---------------------------------------------------------------------------

class TestRetryPolicy(unittest.TestCase):

    def test_post_is_retryable(self):
        from yait_aichain.clients._constants import DEFAULT_RETRIES
        self.assertTrue(DEFAULT_RETRIES._is_method_retryable("POST"))

    def test_post_retries_only_safe_statuses(self):
        """POST retries pre-inference codes only — no double-billing risk."""
        from yait_aichain.clients._constants import DEFAULT_RETRIES
        self.assertEqual(set(DEFAULT_RETRIES.status_forcelist), {429, 503})

    def test_idempotent_policy_covers_full_5xx(self):
        from yait_aichain.clients._constants import DEFAULT_IDEMPOTENT_RETRIES
        self.assertEqual(
            set(DEFAULT_IDEMPOTENT_RETRIES.status_forcelist),
            {429, 500, 502, 503, 504},
        )
        self.assertFalse(DEFAULT_IDEMPOTENT_RETRIES._is_method_retryable("POST"))


# ---------------------------------------------------------------------------
# Fix 4 — delete(ids=[]) must not wipe the collection
# ---------------------------------------------------------------------------

class TestDeleteEmptyIds(unittest.TestCase):

    def _store(self):
        from yait_aichain.tools.vectordb._base import VectorStore
        backend = MagicMock()
        return VectorStore(backend, "test-collection"), backend

    def test_empty_ids_rejected(self):
        store, backend = self._store()
        with self.assertRaises(ValueError):
            store.delete(ids=[])
        backend.delete.assert_not_called()

    def test_empty_filter_rejected(self):
        store, backend = self._store()
        with self.assertRaises(ValueError):
            store.delete(filter={})
        backend.delete.assert_not_called()

    def test_empty_both_rejected(self):
        store, backend = self._store()
        with self.assertRaises(ValueError):
            store.delete(ids=[], filter={})
        backend.delete.assert_not_called()

    def test_none_rejected(self):
        store, backend = self._store()
        with self.assertRaises(ValueError):
            store.delete()
        backend.delete.assert_not_called()

    def test_valid_ids_pass_through(self):
        store, backend = self._store()
        store.delete(ids=["doc_1"])
        backend.delete.assert_called_once_with(
            "test-collection", ids=["doc_1"], filter=None
        )

    def test_empty_ids_with_valid_filter_normalised(self):
        """ids=[] alongside a real filter is normalised to ids=None."""
        store, backend = self._store()
        store.delete(ids=[], filter={"lang": "en"})
        backend.delete.assert_called_once_with(
            "test-collection", ids=None, filter={"lang": "en"}
        )

    def test_delete_tool_rejects_empty_input(self):
        from yait_aichain.tools.vectordb._delete import VectorDeleteTool
        from yait_aichain.tools.vectordb._base import VectorStore

        backend = MagicMock()
        store = VectorStore(backend, "test-collection")
        tool = VectorDeleteTool(store)
        with self.assertRaises(ValueError):
            tool.run(input=[])
        backend.delete.assert_not_called()


# ---------------------------------------------------------------------------
# Helpers for Agent-level tests (fixes 5 and 6)
# ---------------------------------------------------------------------------

def _stub_transport(model, *responses: dict):
    """
    Stub only the TRANSPORT on *model*'s real family client, keeping
    ``build_request`` / ``parse_response`` real.  The wire format moved into
    the client, so replacing the whole client would also stub the format.
    ``_post`` returns each response in order (last repeats forever).
    """
    encoded = [json.dumps(r).encode() for r in responses] or [b"{}"]
    c = model.client
    c._auth_headers = MagicMock(return_value={"Authorization": "Bearer test"})
    c._get = MagicMock(return_value=b'{"data": []}')
    if len(encoded) == 1:
        c._post = MagicMock(return_value=encoded[0])
    else:
        call_count = [0]
        def _side_effect(*_args, **_kwargs):
            idx = min(call_count[0], len(encoded) - 1)
            call_count[0] += 1
            return encoded[idx]
        c._post = MagicMock(side_effect=_side_effect)
    return c


def _mock_orchestrator(*responses: dict):
    """OpenAI-style model whose client returns *responses* in order."""
    from yait_aichain.models import Model

    model = Model("gpt-4o", api_key="test-key")
    _stub_transport(model, *responses)
    return model


def _oai(text: str, total_tokens: int = 10, usage: dict | None = None) -> dict:
    return {
        "choices": [{"message": {"content": text}}],
        "usage":   usage or {"prompt_tokens": total_tokens, "completion_tokens": 0},
    }


# ---------------------------------------------------------------------------
# Fix 5 — Agent.run() must not wipe persistent memory
# ---------------------------------------------------------------------------

class TestPersistentMemory(unittest.TestCase):

    def test_reset_keeps_baseline_and_backend(self):
        import tempfile
        from yait_aichain.agent import AgentMemory, FileBackend

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "state.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"fact": "42"}, f)

            memory = AgentMemory(backend=FileBackend(path))
            memory.set("scratch", "tmp")
            memory.reset()                       # what run() now calls

            self.assertEqual(memory.get("fact"), "42")   # baseline survives
            self.assertNotIn("scratch", memory)          # scratch discarded
            self.assertTrue(os.path.exists(path))        # file untouched

    def test_initial_seed_survives_reset(self):
        from yait_aichain.agent import AgentMemory

        memory = AgentMemory(initial={"language": "French"})
        memory.set("scratch", "tmp")
        memory.reset()
        self.assertEqual(memory.get("language"), "French")
        self.assertNotIn("scratch", memory)

    def test_flush_updates_baseline(self):
        import tempfile
        from yait_aichain.agent import AgentMemory, FileBackend

        with tempfile.TemporaryDirectory() as tmp:
            path   = os.path.join(tmp, "state.json")
            memory = AgentMemory(backend=FileBackend(path))
            memory.set("result", "done")
            memory.flush()
            memory.reset()                       # next run() on same instance
            self.assertEqual(memory.get("result"), "done")

    def test_run_sees_persisted_state(self):
        """End-to-end: run() must expose file-backed state to the agent."""
        import tempfile
        from yait_aichain.agent import Agent, AgentMemory, FileBackend

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "state.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"fact": "42"}, f)

            orchestrator = _mock_orchestrator(
                _oai(json.dumps({"steps": [
                    {"id": 1, "type": "skill", "goal": "answer"},
                ]})),
                _oai(json.dumps({"type": "final_answer", "answer": "ok"})),
            )
            agent  = Agent(
                orchestrator=orchestrator,
                memory=AgentMemory(backend=FileBackend(path)),
                verbose=0,
            )
            result = agent.run("What is the fact?")

            self.assertTrue(result.success)
            self.assertEqual(result.memory.get("fact"), "42")
            self.assertTrue(os.path.exists(path))        # file not deleted


# ---------------------------------------------------------------------------
# Fix 6 — honest success on budget exhaustion / exhausted retries
# ---------------------------------------------------------------------------

class TestHonestSuccess(unittest.TestCase):

    def test_budget_exhaustion_is_failure(self):
        from yait_aichain.agent import Agent

        # Planning alone consumes 100 tokens against a budget of 50.
        orchestrator = _mock_orchestrator(
            _oai(json.dumps({"steps": [
                {"id": 1, "type": "skill", "goal": "step one"},
                {"id": 2, "type": "skill", "goal": "step two"},
            ]}), total_tokens=100),
        )
        agent  = Agent(orchestrator=orchestrator, max_tokens=50, verbose=0)
        result = agent.run("do work")

        self.assertFalse(result.success)
        self.assertIn("budget", (result.error or "").lower())
        self.assertEqual(result.steps_taken, 0)

    def test_exhausted_retries_is_failure(self):
        from yait_aichain.agent import Agent

        plan    = _oai(json.dumps({"steps": [
            {"id": 1, "type": "skill", "goal": "the only step"},
        ]}))
        action  = _oai(json.dumps({
            "type": "skill", "model": "", "system_prompt": "",
            "user_prompt": "try", "output_format": "text",
        }))
        reflect = _oai(json.dumps({
            "assessment": "failure", "decision": "retry", "store_as": "",
        }))
        skill_answer = _oai("attempt output")

        # plan, then per attempt: action, skill execution, reflection
        orchestrator = _mock_orchestrator(
            plan,
            action, skill_answer, reflect,
            action, skill_answer, reflect,
        )
        agent  = Agent(orchestrator=orchestrator, max_attempts=2, verbose=0)
        result = agent.run("do work")

        self.assertFalse(result.success)
        self.assertIn("after 2 attempt", result.error or "")
        self.assertIsNone(result.output)


# ---------------------------------------------------------------------------
# Stage 2 block A — LLM response parsing
# ---------------------------------------------------------------------------

class TestOpenAICompatParser(unittest.TestCase):

    def _parse(self, response, output=None):
        from yait_aichain.clients._families._openai_compat import _parse_openai_compat_response
        return _parse_openai_compat_response(
            response, output or {"format": {"type": "text"}}
        )

    def test_empty_choices_raises_with_context(self):
        with self.assertRaises(ValueError) as ctx:
            self._parse({"choices": []})
        self.assertIn("no choices", str(ctx.exception))

    def test_missing_choices_raises(self):
        with self.assertRaises(ValueError):
            self._parse({})

    def test_provider_error_message_included(self):
        with self.assertRaises(ValueError) as ctx:
            self._parse({"choices": [], "error": {"message": "content filtered"}})
        self.assertIn("content filtered", str(ctx.exception))

    def test_refusal_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self._parse({"choices": [{"message": {"refusal": "I cannot help"}}]})
        self.assertIn("I cannot help", str(ctx.exception))

    def test_truncated_json_mentions_max_tokens(self):
        response = {"choices": [{
            "message": {"content": '{"a": 1, "b'},
            "finish_reason": "length",
        }]}
        with self.assertRaises(ValueError) as ctx:
            self._parse(response, {"format": {"type": "json"}})
        self.assertIn("max_tokens", str(ctx.exception))

    def test_invalid_json_clear_error(self):
        response = {"choices": [{
            "message": {"content": "not json"}, "finish_reason": "stop",
        }]}
        with self.assertRaises(ValueError) as ctx:
            self._parse(response, {"format": {"type": "json"}})
        self.assertIn("invalid JSON", str(ctx.exception))

    def test_happy_path_text_and_json(self):
        ok = {"choices": [{"message": {"content": '{"x": 1}'}}]}
        self.assertEqual(self._parse(ok), '{"x": 1}')
        self.assertEqual(self._parse(ok, {"format": {"type": "json"}}), {"x": 1})

    def test_empty_image_data_raises(self):
        from yait_aichain.clients._families._openai_compat import _parse_image_generations_response
        with self.assertRaises(ValueError):
            _parse_image_generations_response({"data": []})


class TestAgentJsonParsing(unittest.TestCase):

    def _agent(self):
        from yait_aichain.agent import Agent
        return Agent(orchestrator=_mock_orchestrator(_oai("{}")), verbose=0)

    def test_parse_json_rejects_non_dict(self):
        agent = self._agent()
        for bad in ('[1, 2, 3]', '"a string"', '42', 'null'):
            with self.assertRaises(ValueError, msg=bad):
                agent._parse_json(bad)

    def test_parse_json_recovers_object_after_list(self):
        agent = self._agent()
        # Fence contains a list (rejected) but the text has an object too
        text = 'Result: {"steps": []}'
        self.assertEqual(agent._parse_json(text), {"steps": []})

    def test_llm_call_json_retries_once_on_invalid_json(self):
        from yait_aichain.agent import Agent

        orchestrator = _mock_orchestrator(
            _oai("here is prose, no json at all"),
            _oai(json.dumps({"steps": [{"id": 1}]})),
        )
        agent = Agent(orchestrator=orchestrator, verbose=0)
        data, _tokens = agent._llm_call_json(
            agent.orchestrator,
            [{"role": "user", "parts": [{"type": "text", "text": "plan"}]}],
        )
        self.assertEqual(data, {"steps": [{"id": 1}]})
        self.assertEqual(orchestrator.client._post.call_count, 2)

    def test_plan_step_without_id_does_not_crash_logging(self):
        """A plan whose steps lack 'id' must not abort the run via _log."""
        from yait_aichain.agent import Agent

        orchestrator = _mock_orchestrator(
            _oai(json.dumps({"steps": [{"type": "skill", "goal": "g"}]})),
            _oai(json.dumps({"type": "final_answer", "answer": "done"})),
        )
        agent  = Agent(orchestrator=orchestrator, verbose=1)
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            result = agent.run("task")
        self.assertTrue(result.success)

    def test_non_numeric_goto_step_does_not_abort(self):
        """goto_step='two' in a replan must not raise ValueError."""
        from yait_aichain.agent import Agent

        plan    = _oai(json.dumps({"steps": [
            {"id": 1, "type": "skill", "goal": "step"},
        ]}))
        action  = _oai(json.dumps({
            "type": "skill", "model": "", "system_prompt": "",
            "user_prompt": "go", "output_format": "text",
        }))
        skill_answer = _oai("output")
        replan  = _oai(json.dumps({
            "assessment": "failure", "decision": "replan", "store_as": "",
            "revised_plan": [{"id": 1, "type": "skill", "goal": "retry step"}],
            "goto_step": "two",
        }))
        final_reflect = _oai(json.dumps({
            "assessment": "success", "decision": "final_answer",
            "store_as": "", "final_answer": "done",
        }))
        orchestrator = _mock_orchestrator(
            plan,
            action, skill_answer, replan,
            action, skill_answer, final_reflect,
        )
        agent  = Agent(orchestrator=orchestrator, mode="agile", verbose=0)
        result = agent.run("task")
        self.assertTrue(result.success)
        self.assertEqual(result.output, "done")


# ---------------------------------------------------------------------------
# Stage 2 block B — timeouts for search tools and the MCP bridge
# ---------------------------------------------------------------------------

class TestSearchNetworkPolicy(unittest.TestCase):

    def test_all_search_tools_have_timeout_and_retries(self):
        os.environ.setdefault("SERPAPI_API_KEY", "k")
        os.environ.setdefault("PERPLEXITY_API_KEY", "k")
        os.environ.setdefault("BRAVE_SEARCH_API_KEY", "k")

        from yait_aichain.tools.search.serp       import searchSerp
        from yait_aichain.tools.search.perplexity import searchPerplexity
        from yait_aichain.tools.search.brave      import searchBrave
        from yait_aichain.tools.search.openai     import searchOpenAI

        for tool in (searchSerp(), searchPerplexity(),
                     searchBrave(api_key="k"), searchOpenAI(api_key="k")):
            kw      = tool._http.connection_pool_kw
            timeout = kw.get("timeout")
            retries = kw.get("retries")
            name    = type(tool).__name__
            self.assertIsNotNone(timeout, f"{name}: no timeout")
            self.assertIsNotNone(timeout.read_timeout, f"{name}: no read timeout")
            self.assertIsNotNone(retries, f"{name}: no retry policy")
            self.assertTrue(retries._is_method_retryable("POST"), name)


class TestMcpBridgeTimeout(unittest.TestCase):

    def test_hung_coroutine_raises_timeout(self):
        import asyncio
        from yait_aichain.tools.mcp import _run_sync

        async def _hang():
            await asyncio.sleep(60)

        with self.assertRaises(TimeoutError) as ctx:
            _run_sync(_hang(), timeout=0.1)
        self.assertIn("did not complete", str(ctx.exception))

    def test_fast_coroutine_unaffected(self):
        from yait_aichain.tools.mcp import _run_sync

        async def _quick():
            return 42

        self.assertEqual(_run_sync(_quick(), timeout=5.0), 42)

    def test_no_timeout_still_works(self):
        from yait_aichain.tools.mcp import _run_sync

        async def _quick():
            return "ok"

        self.assertEqual(_run_sync(_quick()), "ok")


# ---------------------------------------------------------------------------
# Stage 2 block C — provider parameter correctness
# ---------------------------------------------------------------------------

_MSG = [{"role": "user", "parts": [{"type": "text", "text": "hi"}]}]
_OUT = {"modalities": ["text"], "format": {"type": "text"}}


class TestAnthropicReasoningBudget(unittest.TestCase):

    def test_high_reasoning_raises_max_tokens(self):
        from yait_aichain.models import Model

        m = Model(
            "claude-sonnet-4-6", api_key="k", options={"reasoning": "high"}
        )
        _path, body = m.to_request(_MSG, _OUT)
        budget = body["thinking"]["budget_tokens"]
        self.assertEqual(budget, 20000)
        self.assertGreater(body["max_tokens"], budget)

    def test_explicit_max_tokens_respected_when_sufficient(self):
        from yait_aichain.models import Model

        m = Model(
            "claude-sonnet-4-6", api_key="k",
            options={"reasoning": "low", "max_tokens": 30000},
        )
        _path, body = m.to_request(_MSG, _OUT)
        self.assertEqual(body["max_tokens"], 30000)


class TestOpenAIReasoningModels(unittest.TestCase):

    def test_gpt5_omits_sampling_params_and_sends_reasoning(self):
        from yait_aichain.models import Model

        m = Model("gpt-5", api_key="k", options={"reasoning": "high"})
        path, body = m.to_request(_MSG, _OUT)
        self.assertEqual(path, "/v1/responses")
        self.assertNotIn("temperature", body)
        self.assertNotIn("top_p", body)
        self.assertEqual(body["reasoning"], {"effort": "high"})

    def test_o_series_omits_sampling_params(self):
        from yait_aichain.models import Model

        m = Model("o3-mini", api_key="k", options={"reasoning": "medium"})
        path, body = m.to_request(_MSG, _OUT)
        self.assertEqual(path, "/v1/chat/completions")
        self.assertNotIn("temperature", body)
        self.assertNotIn("top_p", body)
        self.assertEqual(body["reasoning_effort"], "medium")

    def test_regular_gpt_keeps_sampling_params(self):
        from yait_aichain.models import Model

        m = Model("gpt-4o", api_key="k")
        _path, body = m.to_request(_MSG, _OUT)
        self.assertIn("temperature", body)

    def test_o_series_detection(self):
        from yait_aichain.clients._families._openai_compat import _is_o_series_model

        for name in ("o1", "o1-mini", "o3", "o3-mini", "o4-mini"):
            self.assertTrue(_is_o_series_model(name), name)
        for name in ("gpt-4o", "gpt-4o-mini", "omni-x", "o1x"):
            self.assertFalse(_is_o_series_model(name), name)


class TestGoogleKeyInHeader(unittest.TestCase):

    def test_model_path_has_no_key(self):
        from yait_aichain.models import Model

        m = Model("gemini-2.5-pro", api_key="SECRET")
        path, _body = m.to_request(_MSG, _OUT)
        self.assertNotIn("SECRET", path)
        self.assertIn("x-goog-api-key",
                      m.client._auth_headers())
        self.assertEqual(m.client._auth_headers()["x-goog-api-key"], "SECRET")

    def test_embedder_uses_header(self):
        from yait_aichain.tools.embedding._google import EmbeddingGoogle

        emb = EmbeddingGoogle("gemini-embedding-001", api_key="SECRET")
        captured: dict = {}

        def _fake_request(url, body, extra_headers=None):
            captured["url"]     = url
            captured["headers"] = extra_headers or {}
            return {"embeddings": [{"values": [0.1]}]}

        emb._request = _fake_request
        emb._embed_chunk(["hello"], input_type=None, dimensions=None)
        self.assertNotIn("SECRET", captured["url"])
        self.assertEqual(captured["headers"].get("x-goog-api-key"), "SECRET")


# ---------------------------------------------------------------------------
# Stage 2 block D — chunker contract and vectordb providers
# ---------------------------------------------------------------------------

class TestChunkerContract(unittest.TestCase):

    def _tool(self, **kw):
        from yait_aichain.tools.vectordb._chunk import VectorChunkTool
        return VectorChunkTool(**kw)

    def test_code_respects_max_chars(self):
        doc    = "```python\nx = " + "a" * 200 + "\n```"
        chunks = self._tool(max_chars=30, overlap_chars=0).run(doc)
        self.assertTrue(all(c["chars"] <= 30 for c in chunks))

    def test_table_respects_max_chars(self):
        doc    = "| a | b |\n|---|---|\n| " + "x" * 200 + " | y |"
        chunks = self._tool(max_chars=30, overlap_chars=0).run(doc)
        self.assertTrue(all(c["chars"] <= 30 for c in chunks))

    def test_sentence_separator_preserved(self):
        text   = "First sentence here. Second sentence there. Third one closes."
        chunks = self._tool(
            max_chars=40, overlap_chars=0, merge_peers=False
        ).run(text)
        self.assertGreater(len(chunks), 1)
        # No characters may be lost: every period must survive
        combined = " ".join(c["text"] for c in chunks)
        self.assertEqual(combined.count("."), text.count("."))

    def test_overlap_is_applied(self):
        text   = " ".join(f"word{i}" for i in range(100))
        chunks = self._tool(
            max_chars=80, overlap_chars=20, merge_peers=False
        ).run(text)
        self.assertGreater(len(chunks), 2)
        for i in range(1, len(chunks)):
            first_word = chunks[i]["text"].split()[0]
            self.assertIn(first_word, chunks[i - 1]["text"],
                          f"no overlap at boundary {i}")
            self.assertLessEqual(chunks[i]["chars"], 80)


class TestQdrantStringIds(unittest.TestCase):

    def test_point_id_mapping(self):
        from yait_aichain.tools.vectordb.providers._qdrant import _point_id
        import uuid as _uuid

        self.assertEqual(_point_id("42"), 42)            # digits → int
        u = str(_uuid.uuid4())
        self.assertEqual(_point_id(u), u)                # UUID → unchanged
        mapped = _point_id("doc_1")                      # string → UUIDv5
        _uuid.UUID(mapped)                               # must be valid UUID
        self.assertEqual(mapped, _point_id("doc_1"))     # deterministic

    def test_upsert_and_query_roundtrip_original_id(self):
        from yait_aichain.tools.vectordb.providers._qdrant import QdrantBackend

        backend = QdrantBackend(url="http://localhost:6333")
        sent: dict = {}
        backend._put  = lambda path, body: sent.update(body)
        backend._post = lambda path, body: {"result": [{
            "id":      sent["points"][0]["id"],
            "score":   0.9,
            "payload": sent["points"][0]["payload"],
        }]}

        from yait_aichain.tools.vectordb._base import VectorRecord
        rec = VectorRecord(id="doc_1", text="hello", metadata={"k": "v"})
        backend.upsert("col", [rec], [[0.1, 0.2]])

        point = sent["points"][0]
        import uuid as _uuid
        _uuid.UUID(point["id"])                      # Qdrant-valid ID
        self.assertEqual(point["payload"]["_aichain_id"], "doc_1")

        results = backend.query("col", [0.1, 0.2])
        self.assertEqual(results[0].id, "doc_1")     # original restored
        self.assertNotIn("_aichain_id", results[0].metadata)

    def test_delete_empty_raises(self):
        from yait_aichain.tools.vectordb.providers._qdrant import QdrantBackend

        backend = QdrantBackend(url="http://localhost:6333")
        with self.assertRaises(ValueError):
            backend.delete("col")


class TestVectordbProviders(unittest.TestCase):

    def test_pinecone_document_key_not_eaten(self):
        """User metadata key 'document' must survive when 'text' exists."""
        from yait_aichain.tools.vectordb.providers._pinecone import PineconeBackend

        backend = PineconeBackend.__new__(PineconeBackend)
        backend._index_url = "https://h"
        backend._ctrl_http = MagicMock()
        backend._ctrl_http.request.return_value = MagicMock(
            status=200,
            data=json.dumps({"matches": [{
                "id":       "1",
                "score":    0.5,
                "metadata": {"text": "body", "document": "contract.pdf"},
            }]}).encode(),
        )
        backend._headers = lambda: {}
        records = backend.query("col", [0.1])
        self.assertEqual(records[0].metadata.get("document"), "contract.pdf")

    def test_upsert_batches_large_ingests(self):
        from yait_aichain.tools.vectordb._base import VectorStore

        backend = MagicMock()
        store   = VectorStore(backend, "col")
        records = [
            {"id": str(i), "text": f"t{i}", "vector": [0.1]}
            for i in range(120)
        ]
        n = store.upsert(records)
        self.assertEqual(n, 120)
        self.assertEqual(backend.upsert.call_count, 3)   # 50 + 50 + 20
        sizes = [len(c.args[1]) for c in backend.upsert.call_args_list]
        self.assertEqual(sizes, [50, 50, 20])

    def test_chroma_uses_v2_endpoints(self):
        from yait_aichain.tools.vectordb.providers._chroma import ChromaBackend

        backend = ChromaBackend(url="http://localhost:8000")
        called: list = []
        backend._post = lambda path, body: (
            called.append(path) or {"id": "cid-1"}
        )
        backend.create_collection("docs", dimension=3)
        self.assertTrue(called[0].startswith(
            "/api/v2/tenants/default_tenant/databases/default_database/"
        ), called[0])
        self.assertNotIn("/api/v1", called[0])


# ---------------------------------------------------------------------------
# Stage 2 block E — __all__, Chain thread-safety, sttGoogle routing
# ---------------------------------------------------------------------------

class TestToolsAll(unittest.TestCase):

    def test_star_import_works(self):
        import yait_aichain.tools as t
        missing = [n for n in t.__all__ if not hasattr(t, n)]
        self.assertEqual(missing, [])


class TestChainConcurrency(unittest.TestCase):

    def test_concurrent_runs_do_not_interleave_history(self):
        import threading
        from yait_aichain.chain import Chain
        from yait_aichain.tools._base import Tool

        class SlowEcho(Tool):
            name        = "slow_echo"
            description = "echo with a delay"
            parameters  = {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            }
            def run(self, value: str = "") -> str:
                import time
                time.sleep(0.01)
                return value

        chain   = Chain(steps=[(SlowEcho(), "out", {}, {})])
        results = {}

        def _worker(tag):
            results[tag] = chain.run(variables={"value": tag})

        threads = [
            threading.Thread(target=_worker, args=(f"item{i}",))
            for i in range(8)
        ]
        for t in threads: t.start()
        for t in threads: t.join()

        # Every run returns its own output (no cross-item interleaving)
        for tag, out in results.items():
            self.assertEqual(out, tag)
        # Instance attributes reflect exactly one finished run, not a mix
        self.assertEqual(len(chain.history), 1)

    def test_accumulated_updated_on_stop(self):
        """accumulated must reflect the failed run, not the previous one."""
        from yait_aichain.chain import Chain
        from yait_aichain.tools._base import Tool

        class Boom(Tool):
            name        = "boom"
            description = "always fails"
            parameters  = {"type": "object", "properties": {
                "x": {"type": "string"}}, "required": []}
            def run(self, x: str = "") -> str:
                raise RuntimeError("boom")

        chain = Chain(steps=[(Boom(), "out", {}, {})], on_step_error="stop")
        chain.run(variables={"x": "first"})
        self.assertEqual(chain.accumulated.get("x"), "first")
        chain.run(variables={"x": "second"})
        self.assertEqual(chain.accumulated.get("x"), "second")


class TestSttGoogleRouting(unittest.TestCase):

    def test_duration_probe_used_for_routing(self):
        """A small-but-long file must go to the chunked path."""
        from yait_aichain.tools.convert.to_text import sttGoogle

        tool = sttGoogle.__new__(sttGoogle)   # skip __init__ (no credentials)
        self.assertEqual(tool._MAX_SYNC_SECONDS, 55.0)
        # _probe_duration_seconds returns None gracefully without pydub/file
        self.assertIsNone(tool._probe_duration_seconds("/nonexistent.mp3"))


# ---------------------------------------------------------------------------
# 1.2-A — exception hierarchy
# ---------------------------------------------------------------------------

class TestExceptionHierarchy(unittest.TestCase):

    def test_factory_maps_status_to_subclass(self):
        from yait_aichain.clients._errors import (
            error_from_status, APIError, NetworkError, RateLimitError,
            AuthenticationError, InvalidRequestError, NotFoundError, ServerError,
        )
        self.assertIsInstance(error_from_status(0, "x"),   NetworkError)
        self.assertIsInstance(error_from_status(429, "x"), RateLimitError)
        self.assertIsInstance(error_from_status(401, "x"), AuthenticationError)
        self.assertIsInstance(error_from_status(403, "x"), AuthenticationError)
        self.assertIsInstance(error_from_status(400, "x"), InvalidRequestError)
        self.assertIsInstance(error_from_status(422, "x"), InvalidRequestError)
        self.assertIsInstance(error_from_status(404, "x"), NotFoundError)
        self.assertIsInstance(error_from_status(500, "x"), ServerError)
        self.assertIsInstance(error_from_status(503, "x"), ServerError)
        # unmapped status falls back to the base class exactly
        self.assertIs(type(error_from_status(418, "x")), APIError)

    def test_all_subclasses_are_apierror(self):
        """Backward compat: except APIError still catches every failure."""
        from yait_aichain.clients import (
            APIError, NetworkError, RateLimitError, AuthenticationError,
            InvalidRequestError, NotFoundError, ServerError,
        )
        for cls in (NetworkError, RateLimitError, AuthenticationError,
                    InvalidRequestError, NotFoundError, ServerError):
            self.assertTrue(issubclass(cls, APIError), cls.__name__)

    def test_retry_after_parsed_for_rate_limit(self):
        from yait_aichain.clients._errors import error_from_status, RateLimitError
        e = error_from_status(429, "slow down", {"Retry-After": "30"})
        self.assertIsInstance(e, RateLimitError)
        self.assertEqual(e.retry_after, 30.0)

    def test_retry_after_absent_or_unparseable_is_none(self):
        from yait_aichain.clients._errors import error_from_status
        self.assertIsNone(error_from_status(429, "x").retry_after)
        self.assertIsNone(
            error_from_status(429, "x", {"Retry-After": "Wed, 21 Oct"}).retry_after
        )

    def test_status_and_message_preserved(self):
        from yait_aichain.clients._errors import error_from_status
        e = error_from_status(401, "invalid key")
        self.assertEqual(e.status, 401)
        self.assertEqual(e.message, "invalid key")
        self.assertIn("401", str(e))

    def test_base_import_path_still_works(self):
        """Legacy `from clients._base import APIError` must keep working."""
        from yait_aichain.clients._base import APIError as FromBase
        from yait_aichain.clients._errors import APIError as FromErrors
        self.assertIs(FromBase, FromErrors)

    def test_exported_at_top_level(self):
        import yait_aichain
        for name in ("APIError", "RateLimitError", "AuthenticationError",
                     "NetworkError", "ServerError"):
            self.assertTrue(hasattr(yait_aichain, name), name)


# ---------------------------------------------------------------------------
# 1.2-B — Usage accounting
# ---------------------------------------------------------------------------

class TestUsage(unittest.TestCase):

    def test_extract_openai_shape(self):
        from yait_aichain.models._usage import extract_usage
        u = extract_usage({"usage": {
            "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})
        self.assertEqual((u.input_tokens, u.output_tokens, u.total_tokens), (10, 5, 15))

    def test_extract_anthropic_shape(self):
        from yait_aichain.models._usage import extract_usage
        u = extract_usage({"usage": {"input_tokens": 7, "output_tokens": 3}})
        self.assertEqual((u.input_tokens, u.output_tokens), (7, 3))
        self.assertEqual(u.total_tokens, 10)   # derived when absent

    def test_extract_google_shape(self):
        from yait_aichain.models._usage import extract_usage
        u = extract_usage({"usageMetadata": {
            "promptTokenCount": 20, "candidatesTokenCount": 8, "totalTokenCount": 28}})
        self.assertEqual((u.input_tokens, u.output_tokens, u.total_tokens), (20, 8, 28))

    def test_extract_missing_is_zero_and_never_raises(self):
        from yait_aichain.models._usage import extract_usage
        self.assertFalse(extract_usage({}))
        self.assertFalse(extract_usage({"choices": []}))
        self.assertFalse(extract_usage("not a dict"))

    def test_usage_is_additive(self):
        from yait_aichain.models._usage import Usage
        a = Usage(10, 5, 15)
        b = Usage(2, 3, 5)
        c = a + b
        self.assertEqual((c.input_tokens, c.output_tokens, c.total_tokens), (12, 8, 20))
        # sum() starts from 0 (int) → __radd__
        self.assertEqual(sum([a, b]).total_tokens, 20)

    def test_skill_records_last_usage(self):
        from yait_aichain.models import Model
        from yait_aichain.skills import Skill

        model = Model("gpt-4o", api_key="k")
        model.client = _stub_transport(model, _oai("hi", usage={
            "prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15}))
        skill = Skill(model=model,
                      input={"messages": [{"role": "user", "parts": ["x"]}]})
        self.assertIsNone(skill.last_usage)   # before run
        skill.run()
        self.assertEqual(skill.last_usage.input_tokens, 11)
        self.assertEqual(skill.last_usage.total_tokens, 15)

    def test_chain_sums_step_usage(self):
        from yait_aichain.models import Model
        from yait_aichain.skills import Skill
        from yait_aichain.chain import Chain

        def _mk(toks):
            m = Model("gpt-4o", api_key="k")
            m.client = _stub_transport(m, _oai("ok", usage={
                "prompt_tokens": toks, "completion_tokens": 1,
                "total_tokens": toks + 1}))
            return Skill(model=m,
                         input={"messages": [{"role": "user", "parts": ["x"]}]})

        chain = Chain(steps=[(_mk(10), "a", {}, {}), (_mk(20), "b", {}, {})])
        chain.run()
        self.assertEqual(chain.last_usage.input_tokens, 30)
        self.assertEqual(chain.last_usage.total_tokens, 32)

    def test_usage_exported_top_level(self):
        import yait_aichain
        self.assertTrue(hasattr(yait_aichain, "Usage"))


# ---------------------------------------------------------------------------
# 1.2-C — cost from pricing data
# ---------------------------------------------------------------------------

class TestPricing(unittest.TestCase):

    def test_estimate_known_model(self):
        from yait_aichain.models._usage import Usage
        from yait_aichain.models._usage import estimate_cost
        # gpt-4o: 2.5 in / 10 out per Mtok → 1M in + 1M out = 2.5 + 10 = 12.5
        cost = estimate_cost(Usage(1_000_000, 1_000_000, 2_000_000), "gpt-4o")
        self.assertAlmostEqual(cost, 12.5)

    def test_unknown_model_is_none(self):
        from yait_aichain.models._usage import Usage
        from yait_aichain.models._usage import estimate_cost
        self.assertIsNone(estimate_cost(Usage(100, 100), "totally-unknown-xyz"))

    def test_attach_cost_fills_field(self):
        from yait_aichain.models._usage import Usage
        from yait_aichain.models._usage import attach_cost
        u = attach_cost(Usage(1_000_000, 0, 1_000_000), "gpt-4o")
        self.assertAlmostEqual(u.cost, 2.5)

    def test_skill_last_usage_has_cost(self):
        from yait_aichain.models import Model
        from yait_aichain.skills import Skill

        model = Model("gpt-4o", api_key="k")
        model.client = _stub_transport(model, _oai("hi", usage={
            "prompt_tokens": 1_000_000, "completion_tokens": 0,
            "total_tokens": 1_000_000}))
        skill = Skill(model=model,
                      input={"messages": [{"role": "user", "parts": ["x"]}]})
        skill.run()
        self.assertAlmostEqual(skill.last_usage.cost, 2.5)

    def test_chain_sums_cost(self):
        from yait_aichain.models._usage import Usage
        a = Usage(0, 0, 0, cost=1.5)
        b = Usage(0, 0, 0, cost=2.0)
        self.assertAlmostEqual((a + b).cost, 3.5)
        # None + value keeps the known part
        self.assertAlmostEqual((Usage(0, 0, 0) + b).cost, 2.0)
        # both None stays None
        self.assertIsNone((Usage(0, 0, 0) + Usage(0, 0, 0)).cost)


# ---------------------------------------------------------------------------
# 1.2-D — provider/model routing
# ---------------------------------------------------------------------------

class TestProviderRouting(unittest.TestCase):

    def test_explicit_prefix_selects_provider(self):
        from yait_aichain.models import Model
        self.assertEqual(Model("openai/gpt-4o", api_key="k")._provider, "openai")
        self.assertEqual(Model("anthropic/claude-x", api_key="k")._provider, "anthropic")

    def test_prefix_stripped_from_wire_name(self):
        from yait_aichain.models import Model
        m = Model("openai/gpt-4o", api_key="k")
        self.assertEqual(m.name, "gpt-4o")   # prefix never reaches the API

    def test_bare_name_still_works(self):
        from yait_aichain.models import Model
        m = Model("gpt-4o", api_key="k")
        self.assertEqual(m._provider, "openai")
        self.assertEqual(m.name, "gpt-4o")

    def test_explicit_prefix_allows_custom_model_name(self):
        """A name the regex can't recognise works via an explicit prefix."""
        from yait_aichain.models import Model
        m = Model("openai/ft:gpt-4o:org:abc", api_key="k")
        self.assertEqual(m._provider, "openai")
        self.assertEqual(m.name, "ft:gpt-4o:org:abc")

    def test_unknown_prefix_falls_through_to_regex(self):
        from yait_aichain.models._base import _split_provider_prefix
        self.assertEqual(_split_provider_prefix("foo/bar"), (None, "foo/bar"))

    def test_split_helper(self):
        from yait_aichain.models._base import _split_provider_prefix
        self.assertEqual(_split_provider_prefix("openai/gpt-4o"), ("openai", "gpt-4o"))
        self.assertEqual(_split_provider_prefix("gpt-4o"), (None, "gpt-4o"))


# ---------------------------------------------------------------------------
# 1.2-E — registry model refresh
# ---------------------------------------------------------------------------

class TestRegistryRefresh(unittest.TestCase):

    class _FakeClient:
        def __init__(self, models): self._models = models
        def list_models(self): return self._models

    def test_diff_new_and_removed(self):
        from yait_aichain.models import registry
        # live has one new model and is missing one previously-registered one
        registered = registry.models(provider="openai")
        live = [m for m in registered if m != registered[0]] + ["gpt-6-new"]
        d = registry.refresh("openai", client=self._FakeClient(live))
        self.assertIn("gpt-6-new", d["new"])
        self.assertIn(registered[0], d["removed"])
        self.assertNotIn("gpt-6-new", d["registered"])

    def test_does_not_mutate_registry(self):
        from yait_aichain.models import registry
        before = registry.models(provider="openai")
        registry.refresh("openai", client=self._FakeClient(["only-this"]))
        after = registry.models(provider="openai")
        self.assertEqual(before, after)   # registry stays a reference

    def test_unknown_provider_raises(self):
        from yait_aichain.models import registry
        with self.assertRaises(ValueError):
            registry.refresh("nope", client=self._FakeClient([]))

    def test_missing_key_raises_when_no_client(self):
        from yait_aichain.models import registry
        import os
        saved = os.environ.pop("OPENAI_API_KEY", None)
        try:
            with self.assertRaises(ValueError):
                registry.refresh("openai")   # no client, no key
        finally:
            if saved is not None:
                os.environ["OPENAI_API_KEY"] = saved


# ---------------------------------------------------------------------------
# 1.2-F — model fallback chain
# ---------------------------------------------------------------------------

class TestFallbackChain(unittest.TestCase):

    def _ok(self, text):
        from yait_aichain.models import Model
        m = Model("gpt-4o", api_key="k")
        m.client = _stub_transport(m, _oai(text))
        return m

    def _failing(self, exc):
        from yait_aichain.models import Model
        m = Model("gpt-4o", api_key="k")
        m.client._auth_headers = MagicMock(return_value={})
        m.client._post = MagicMock(side_effect=exc)
        return m

    def test_single_model_unchanged(self):
        from yait_aichain.skills import Skill
        s = Skill(model=self._ok("solo"),
                  input={"messages": [{"role": "user", "parts": ["x"]}]})
        self.assertEqual(s.run(), "solo")
        self.assertEqual(len(s.models), 1)

    def test_transient_failure_falls_through(self):
        from yait_aichain.skills import Skill
        from yait_aichain.clients._errors import RateLimitError
        s = Skill(model=[self._failing(RateLimitError(429, "slow")),
                         self._ok("second")],
                  input={"messages": [{"role": "user", "parts": ["x"]}]})
        self.assertEqual(s.run(), "second")

    def test_server_and_network_errors_fall_through(self):
        from yait_aichain.skills import Skill
        from yait_aichain.clients._errors import ServerError, NetworkError
        for exc in (ServerError(503, "down"), NetworkError(0, "no route")):
            s = Skill(model=[self._failing(exc), self._ok("recovered")],
                      input={"messages": [{"role": "user", "parts": ["x"]}]})
            self.assertEqual(s.run(), "recovered")

    def test_non_transient_does_not_fall_through(self):
        """Auth error must propagate immediately, not hide behind fallback."""
        from yait_aichain.skills import Skill
        from yait_aichain.clients._errors import AuthenticationError
        s = Skill(model=[self._failing(AuthenticationError(401, "bad key")),
                         self._ok("unreached")],
                  input={"messages": [{"role": "user", "parts": ["x"]}]})
        with self.assertRaises(AuthenticationError):
            s.run()

    def test_all_models_exhausted_raises_last(self):
        from yait_aichain.skills import Skill
        from yait_aichain.clients._errors import RateLimitError
        s = Skill(model=[self._failing(RateLimitError(429, "a")),
                         self._failing(RateLimitError(429, "b"))],
                  input={"messages": [{"role": "user", "parts": ["x"]}]})
        with self.assertRaises(RateLimitError):
            s.run()

    def test_empty_model_list_rejected(self):
        from yait_aichain.skills import Skill
        with self.assertRaises(ValueError):
            Skill(model=[], input={"messages": [{"role": "user", "parts": ["x"]}]})


# ---------------------------------------------------------------------------
# Fix 7 — safe templating
# ---------------------------------------------------------------------------

class TestSafeTemplating(unittest.TestCase):

    def test_json_example_in_prompt_does_not_raise(self):
        from yait_aichain.skills._adapters import substitute

        messages = [{"role": "user", "parts": [
            {"type": "text",
             "text": 'Return JSON like {"a": 1, "b": []} for {topic}'},
        ]}]
        out = substitute(messages, {"topic": "cats"})
        self.assertEqual(
            out[0]["parts"][0]["text"],
            'Return JSON like {"a": 1, "b": []} for cats',
        )

    def test_unbalanced_braces_survive(self):
        from yait_aichain._template import substitute_placeholders

        self.assertEqual(
            substitute_placeholders("closing } and { opening", {"x": "y"}),
            "closing } and { opening",
        )

    def test_empty_braces_survive(self):
        from yait_aichain._template import substitute_placeholders

        self.assertEqual(
            substitute_placeholders("set() is {} in maths", {"x": "y"}),
            "set() is {} in maths",
        )

    def test_unknown_placeholder_left_intact(self):
        from yait_aichain._template import substitute_placeholders

        self.assertEqual(
            substitute_placeholders("hello {name}", {"other": "x"}),
            "hello {name}",
        )

    def test_doubled_braces_stay_literal(self):
        # A doubled-brace ``{{name}}`` must NOT be substituted even when a
        # variable of that name exists — doubled braces stay literal.
        from yait_aichain._template import substitute_placeholders

        self.assertEqual(
            substitute_placeholders("emit {{name}} and {name}", {"name": "X"}),
            "emit {{name}} and X",
        )

    def test_known_placeholder_substituted(self):
        from yait_aichain._template import substitute_placeholders

        self.assertEqual(
            substitute_placeholders("hello {name}", {"name": "world"}),
            "hello world",
        )

    def test_non_string_value_stringified(self):
        from yait_aichain._template import substitute_placeholders

        self.assertEqual(
            substitute_placeholders("n = {n}", {"n": 7}),
            "n = 7",
        )

    def test_format_spec_left_intact(self):
        """Format specs are no longer interpreted — documented behaviour."""
        from yait_aichain._template import substitute_placeholders

        self.assertEqual(
            substitute_placeholders("{x:>10}", {"x": "y"}),
            "{x:>10}",
        )


if __name__ == "__main__":
    unittest.main()
