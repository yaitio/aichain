"""
3.7.0 — a setting is accepted in exactly one place.

Each was accepted at construction and again with a call, the call winning in
silence: an embedder's length (a store filled with vectors of two lengths), a
TTS region and model, a vector store's collection (a model switching an agent
to another collection), a chunker's sizes, REST headers and timeout, MCP
headers, env and timeout. Now each lives in the tool, and given to a call is an
error that says where it belongs. Alongside: a reranker that dropped any
keyword, a dimension the model could not honour sent nowhere, a REST token in
the schema a model sees, and a chunk overlap that swallowed small pieces.
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from yait_aichain import Agent, Chain, Model
from yait_aichain.tools import Embedding, Reranker, chunk, embed, speech, transcribe

KEYS = {"OPENAI_API_KEY": "k", "COHERE_API_KEY": "k", "DASHSCOPE_API_KEY": "k", "XAI_API_KEY": "k",
        "GOOGLE_AI_API_KEY": "k", "VOYAGE_API_KEY": "k"}


@patch.dict(os.environ, KEYS)
class TestEmbedder(unittest.TestCase):

    def test_length_and_batch_are_settings(self):
        e = embed("openai", model="text-embedding-3-small", dimensions=512, batch_size=10)
        self.assertEqual((e.dimensions, e.batch_size), (512, 10))
        self.assertNotIn("dimensions", e.parameters["properties"]["options"]["properties"])
        with self.assertRaisesRegex(ValueError, "dimensions is a setting of the embedder"):
            e.run(["a"], {"dimensions": 256})
        with self.assertRaisesRegex(ValueError, "dimensions is a setting of the embedder"):
            e.embed(["a"], dimensions=256)

    def test_the_settings_reach_the_request(self):
        e = embed("openai", model="text-embedding-3-small", dimensions=512, batch_size=1)
        e._embed_chunk = MagicMock(return_value=([[0.0] * 512], 1, {}))
        e.run(["a", "b"], {"input_type": "query"})
        self.assertEqual(e._embed_chunk.call_count, 2)                      # one text a request
        self.assertEqual(e._embed_chunk.call_args.kwargs, {"input_type": "query", "dimensions": 512})

    def test_an_unknown_setting_or_option_is_refused(self):
        with self.assertRaises(TypeError):
            Embedding("openai/text-embedding-3-small", input_type="document")
        with self.assertRaisesRegex(ValueError, "unknown option"):
            embed("openai", model="text-embedding-3-small").run(["a"], {"truncate": "END"})

    def test_a_length_the_model_cannot_take_is_refused_where_it_is_set(self):
        with self.assertRaisesRegex(ValueError, "text-embedding-ada-002 has a fixed length"):
            embed("openai", model="text-embedding-ada-002", dimensions=512)
        with self.assertRaisesRegex(ValueError, "text-embedding-v2 has a fixed length"):
            embed("qwen", model="text-embedding-v2", dimensions=512)
        embed("qwen", model="text-embedding-v4", dimensions=512)


@patch.dict(os.environ, KEYS)
class TestSpeech(unittest.TestCase):

    def test_the_model_is_a_setting(self):
        self.assertEqual(speech("xai", model="grok-tts-2").model, "grok-tts-2")
        self.assertEqual(speech("xai").model, "grok-tts")                    # the provider's default
        self.assertEqual(transcribe("qwen", model="paraformer-v2").model, "paraformer-v2")
        for t in (speech("xai"), transcribe("xai")):
            self.assertNotIn("model", t.parameters["properties"]["options"]["properties"])
        with self.assertRaisesRegex(ValueError, "model is a setting of the tool"):
            speech("xai").run("hi", {"model": "x"})
        with self.assertRaisesRegex(TypeError, "has one model"):
            speech("google", model="x")

    def test_qwens_region_is_a_setting(self):
        t = speech("qwen", region="intl")
        with self.assertRaisesRegex(ValueError, "region is a setting of the tool"):
            t.run("hi", {"region": "cn"})


class TestVectorsAndChunk(unittest.TestCase):

    def test_the_collection_is_the_stores(self):
        from yait_aichain.utilities.vectors import VectorQueryTool
        q = VectorQueryTool(MagicMock())
        self.assertNotIn("collection", q.parameters["properties"]["options"]["properties"])
        with self.assertRaisesRegex(ValueError, "collection is the store's setting"):
            q.run("caching", {"collection": "someone_elses"})

    def test_the_sizes_are_the_chunkers(self):
        with self.assertRaisesRegex(ValueError, "max_chars is a setting of the chunker"):
            chunk().run("text", {"max_chars": 300})
        self.assertNotIn("options", chunk().parameters["properties"])

    def test_the_overlap_scales_with_the_size(self):
        self.assertEqual(chunk()._overlap_chars, 150)                        # as before, at the default
        self.assertEqual(chunk(max_chars=200)._overlap_chars, 20)
        text = "# Title\n\n" + "A sentence about caching. " * 30
        self.assertLessEqual(len(chunk(max_chars=200).run(text)), 6)          # was 30 windows


@patch.dict(os.environ, KEYS)
class TestReranker(unittest.TestCase):

    def test_an_unknown_keyword_is_refused(self):
        with self.assertRaisesRegex(TypeError, "top_n and query are options of the call"):
            Reranker("rerank-v3.5", top_n=3)
        with self.assertRaisesRegex(TypeError, "got bogus"):
            Reranker("rerank-v3.5", bogus=1)


@patch.dict(os.environ, {"AICHAIN_ALLOW_PRIVATE_URLS": "1"})
class TestREST(unittest.TestCase):

    def _tool(self, **kw):
        from yait_aichain.utilities.rest import RestApiTool
        t = RestApiTool("get_booking", "One booking.", "GET", "https://example.com/booking/{id}",
                        path_params=["id"], **kw)
        t._http = MagicMock()
        t._http.request.return_value = MagicMock(status=200, data=b'{"ok": 1}',
                                                 headers={"content-type": "application/json"})
        return t

    def test_the_token_is_not_in_what_a_model_sees_but_a_chain_fills_it(self):
        t = self._tool(auth={"type": "bearer", "token_field": "token"})
        self.assertEqual(list(t.schema()["function"]["parameters"]["properties"]), ["id"])
        self.assertIn("token", t.parameters["properties"])                   # the schema itself is whole
        self.assertIn("credential", t.check_args({"id": "1", "token": "x"}))
        Chain(steps=[(t, "r", {"id": "booking", "token": "tok"})]).run(
            variables={"booking": "7", "tok": "from-login"})
        self.assertEqual(t._http.request.call_args.kwargs["headers"]["Authorization"], "Bearer from-login")

    def test_the_token_has_one_source(self):
        with self.assertRaisesRegex(ValueError, "one source of its token; got token_field, env"):
            self._tool(auth={"type": "bearer", "token_field": "t", "env": "TOKEN"})
        t = self._tool(auth={"type": "bearer", "token": lambda: "from-a-callable"})
        t.run(id="1")
        self.assertEqual(t._http.request.call_args.kwargs["headers"]["Authorization"], "Bearer from-a-callable")

    def test_headers_and_timeout_are_the_tools(self):
        t = self._tool(headers={"Accept": "application/json"}, timeout=5)
        t.run(id="1")
        kw = t._http.request.call_args.kwargs
        self.assertEqual((kw["headers"]["Accept"], kw["timeout"].read_timeout), ("application/json", 5.0))
        with self.assertRaisesRegex(ValueError, "headers is a setting of the tool"):
            t.run(id="1", options={"headers": {"X": "1"}})
        with self.assertRaisesRegex(ValueError, "timeout is a setting of the tool"):
            t.run(id="1", options={"timeout": 1})

    def test_a_field_is_given_once(self):
        with self.assertRaisesRegex(ValueError, r"\['id'\] given both"):
            self._tool().run({"id": "1"}, id="2")


class TestMCP(unittest.TestCase):

    def test_headers_env_cwd_and_transport_have_one_source(self):
        from yait_aichain.utilities.mcp._tools import _build_client
        with self.assertRaisesRegex(ValueError, "MCP headers given twice"):
            _build_client({"url": "https://example.com/mcp", "headers": {"A": "1"}}, headers={"B": "2"},
                          env=None, cwd=None, transport=None)
        with self.assertRaisesRegex(ValueError, "MCP env given twice"):
            _build_client({"command": "server", "env": {"A": "1"}}, headers=None, env={"B": "2"},
                          cwd=None, transport=None)

    def test_the_utility_names_its_two_timeouts(self):
        from yait_aichain.utilities import MCP
        m = MCP("https://example.com/mcp", connect_timeout=10, call_timeout=30)
        self.assertEqual((m._settings["timeout"], m._settings["call_timeout"]), (10, 30))


if __name__ == "__main__":
    unittest.main()
