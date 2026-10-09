"""
3.5.0 — embed, rerank and chunk are tools; speech, transcribe, wait and gate
are made by their new names; six families moved to folders of their own.

What is pinned:

  * each factory makes its tool by the provider's name, takes ``name=`` and
    ``description=``, and names the services that exist when given another;
  * an embedder and a reranker are ``Tool``s an agent accepts — until 3.5.0
    they were plain classes and ``Agent(tools=[Reranker(...)])`` failed with
    ``AttributeError``; an embedding reaches a model as a one-line summary,
    never as the vectors;
  * every module path from before the move still resolves, by every import
    form and by ``mock.patch`` — a saved chain names its tools by these paths;
  * a saved chain brings back what each tool is: the model of an embedder or
    a reranker, the sizes of a chunker, and any name or description given to a
    factory. The sizes and the model were lost before this.
"""

import importlib
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from yait_aichain import Agent, Chain, Model
from yait_aichain.tools import (Embedder, RerankBase, Tool, chunk, embed, gate, rerank, speech,
                                transcribe, wait)
from yait_aichain.tools.embed._base import EmbeddingResult

KEYS = {"OPENAI_API_KEY": "k", "COHERE_API_KEY": "k", "VOYAGE_API_KEY": "k",
        "GOOGLE_AI_API_KEY": "k", "DASHSCOPE_API_KEY": "k", "XAI_API_KEY": "k"}


@patch.dict(os.environ, KEYS)
class TestFactories(unittest.TestCase):

    def test_each_service_by_name(self):
        from yait_aichain.tools import embed as e, rerank as r, speech as s, transcribe as t
        for factory, kwargs in ((e, {"model": "m"}), (r, {"model": "m"}), (s, {}), (t, {})):
            for service, cls in factory.SERVICES.items():
                with self.subTest(factory=factory.__name__, service=service):
                    self.assertIs(type(factory(service, **kwargs)), cls)

    def test_an_unknown_service_names_the_ones_that_exist(self):
        for call, kind in ((lambda: embed("openia", model="m"), "embed"),
                           (lambda: rerank("coher", model="m"), "rerank"),
                           (lambda: speech("open"), "speech"), (lambda: transcribe("x"), "transcribe")):
            with self.subTest(kind), self.assertRaisesRegex(ValueError, f"unknown {kind} service .*one of: "):
                call()

    def test_the_model_is_a_setting_and_required(self):
        with self.assertRaises(TypeError):
            embed("openai")
        with self.assertRaises(TypeError):
            rerank("cohere")

    def test_region_is_qwens(self):
        self.assertEqual(speech("qwen", region="intl")._region, "intl")
        with self.assertRaisesRegex(TypeError, "region is speech\\('qwen'\\)'s"):
            speech("openai", region="intl")

    def test_name_and_description_override_on_every_factory(self):
        for t in (embed("openai", model="m", name="n", description="d"),
                  rerank("cohere", model="m", name="n", description="d"),
                  speech("openai", name="n", description="d"), transcribe("openai", name="n", description="d"),
                  chunk(name="n", description="d"), wait(name="n", description="d"),
                  gate(chunk(), name="n", description="d")):
            with self.subTest(type(t).__name__):
                self.assertEqual((t.name, t.description), ("n", "d"))

    def test_chunk_and_wait_and_gate(self):
        c = chunk(max_chars=800, overlap_chars=80)
        self.assertEqual((c._max_chars, c._overlap_chars), (800, 80))
        self.assertEqual(wait(reason="ok").name, "wait")
        self.assertTrue(gate(c, reason="approve?").suspends)


@patch.dict(os.environ, KEYS)
class TestEmbedAndRerankAreTools(unittest.TestCase):

    def test_they_are_tools_that_read(self):
        for t in (embed("openai", model="m"), rerank("cohere", model="m")):
            with self.subTest(type(t).__name__):
                self.assertIsInstance(t, Tool)
                self.assertEqual(t.risk, "read")
                self.assertTrue(callable(t.schema))
        self.assertTrue(issubclass(Embedder, Tool) and issubclass(RerankBase, Tool))

    def test_names_follow_the_provider(self):
        self.assertEqual(embed("openai", model="m").name, "embedOpenAI")
        self.assertEqual(embed("voyage", model="m").name, "embedVoyage")
        self.assertEqual(rerank("cohere", model="m").name, "rerankCohere")

    def test_an_agent_accepts_them(self):
        Agent(Model("gpt-5.4-mini", api_key="k"),
              tools=[embed("openai", model="m"), rerank("cohere", model="m")])

    def test_a_model_reads_a_summary_not_the_vectors(self):
        from yait_aichain.agent._prompts import result_message
        r = EmbeddingResult(embeddings=[[0.1] * 1536, [0.2] * 1536], model="text-embedding-3-small",
                            provider="openai", dimensions=1536, total_tokens=7)
        text = result_message(r)
        self.assertEqual(text, "2 vector(s) of 1536 dimensions from openai text-embedding-3-small, 7 tokens")
        self.assertNotIn("0.1", text)


class TestOldPathsStillResolve(unittest.TestCase):

    OLD = {
        "yait_aichain.tools.embedding":          "yait_aichain.tools.embed",
        "yait_aichain.tools.embedding._openai":  "yait_aichain.tools.embed._openai",
        "yait_aichain.tools.reranking":          "yait_aichain.tools.rerank",
        "yait_aichain.tools.reranking._cohere":  "yait_aichain.tools.rerank._cohere",
        "yait_aichain.tools.convert.to_speech":  "yait_aichain.tools.speech._tts",
        "yait_aichain.tools.convert.to_text":    "yait_aichain.tools.transcribe._stt",
        "yait_aichain.tools.vectordb._chunk":    "yait_aichain.tools.chunk._chunk",
        "yait_aichain.tools._wait":              "yait_aichain.tools.general._wait",
    }

    def test_by_import_module_as_chain_load_does(self):
        for old, new in self.OLD.items():
            with self.subTest(old):
                self.assertIs(importlib.import_module(old), importlib.import_module(new))

    def test_by_from_import_and_import_as_and_attribute(self):
        import yait_aichain
        from yait_aichain.tools.convert.to_speech import ttsOpenAI
        from yait_aichain.tools.embedding import Embedding
        import yait_aichain.tools.reranking._cohere as cohere_module
        self.assertEqual(ttsOpenAI.__module__, "yait_aichain.tools.speech._tts")
        self.assertTrue(callable(Embedding))
        self.assertIs(cohere_module, sys.modules["yait_aichain.tools.rerank._cohere"])
        self.assertIs(yait_aichain.tools.convert.to_text, sys.modules["yait_aichain.tools.transcribe._stt"])

    def test_by_mock_patch(self):
        # 3.10 and 3.11 resolve a patch path through attributes.
        with patch("yait_aichain.tools.convert.to_speech.os") as p:
            self.assertIs(sys.modules["yait_aichain.tools.speech._tts"].os, p)
        with patch("yait_aichain.tools.embedding._base.make_http") as p:
            self.assertIs(sys.modules["yait_aichain.tools.embed._base"].make_http, p)

    def test_a_chain_saved_with_an_old_path_loads(self):
        path = os.path.join(tempfile.mkdtemp(), "old.yaml")
        with open(path, "w") as fh:
            fh.write("steps:\n- kind: tool\n  output_key: pieces\n  tool:\n"
                     "    class: yait_aichain.tools.vectordb._chunk.VectorChunkTool\n"
                     "    init_args: {max_chars: 700}\n")
        tool = Chain.load(path)._steps[0][0]
        self.assertEqual((type(tool).__name__, tool._max_chars), ("VectorChunkTool", 700))


@patch.dict(os.environ, KEYS)
class TestASavedChainKeepsWhatEachToolIs(unittest.TestCase):

    def _round_trip(self, tool):
        path = os.path.join(tempfile.mkdtemp(), "c.yaml")
        Chain(steps=[(tool, "o")]).save(path)
        return Chain.load(path)._steps[0][0], open(path).read()

    def test_an_embedder_keeps_its_model_and_settings(self):
        from yait_aichain.tools import Embedding
        back, text = self._round_trip(Embedding("qwen/text-embedding-v4", api_key="secret",
                                                region="intl", dimensions=512))
        self.assertEqual((back.model, back._region, back.dimensions), ("text-embedding-v4", "intl", 512))
        self.assertNotIn("secret", text)

    def test_a_reranker_keeps_its_model(self):
        back, _ = self._round_trip(rerank("cohere", model="rerank-v3.5"))
        self.assertEqual(back.model, "rerank-v3.5")

    def test_a_chunker_keeps_its_sizes(self):
        back, _ = self._round_trip(chunk(max_chars=800, overlap_chars=80, merge_peers=False))
        self.assertEqual((back._max_chars, back._overlap_chars, back._merge_enabled), (800, 80, False))

    def test_a_name_or_description_given_to_any_factory_comes_back(self):
        for tool in (speech("openai", name="say"), transcribe("openai", description="Write down a call."),
                     chunk(name="cut"), embed("openai", model="m", name="vec")):
            with self.subTest(type(tool).__name__):
                back, _ = self._round_trip(tool)
                self.assertEqual((back.name, back.description), (tool.name, tool.description))



@patch.dict(os.environ, KEYS)
class TestTheReadmeRagChainRuns(unittest.TestCase):
    """The README shows this chain as working; until 3.4.0 its templates were
    dropped, and until 3.5.0 the reranker was not a tool."""

    def test_query_then_rerank(self):
        import json
        from unittest.mock import MagicMock
        from yait_aichain.tools.vectordb import vectorQuery
        from yait_aichain.tools.vectordb._base import VectorRecord
        store = MagicMock()
        store._collection = "docs"
        store.query.return_value = [VectorRecord(id=f"c{i}", text=t, score=0.5, metadata={}) for i, t in
                                    enumerate(["KV cache stores keys", "Bananas are yellow", "Attention reuses keys"])]
        reranker = rerank("cohere", model="rerank-v3.5")
        reply = MagicMock(status=200, data=json.dumps({"results": [{"index": 2, "relevance_score": 0.91},
                                                                   {"index": 0, "relevance_score": 0.84}]}).encode())
        reranker._http = MagicMock()
        reranker._http.request.return_value = reply

        res = Chain(steps=[
            (vectorQuery(store), "candidates", {"input": "{question}", "options": {"n": 20}}),
            (reranker,           "context",    {"input": "{candidates}",
                                                "options": {"query": "{question}", "top_n": 2}}),
        ]).run(variables={"question": "How does KV caching work?"})

        self.assertEqual(store.query.call_args.args, ("How does KV caching work?",))
        sent = json.loads(reranker._http.request.call_args.kwargs["body"])
        self.assertEqual((sent["query"], sent["top_n"], len(sent["documents"])), ("How does KV caching work?", 2, 3))
        self.assertEqual([c["id"] for c in res["context"]], ["c2", "c0"])

if __name__ == "__main__":
    unittest.main()
