"""
Utilities (3.6.0) — what holds a connection or a place, and gives an agent the
tools it should have.

A utility's methods are called directly, by code and by a chain; its tools are
called with ``.run`` and handed to an agent one by one. An agent given a
utility, or one of its methods, refuses it by name and lists the utility's
tools. A chain takes a method as a step, filled from the variables its
signature names, and cannot be saved with one. And every module path the
utilities had under ``tools/`` still imports — including the vector store's
providers, which no table lists.
"""

import importlib
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from yait_aichain import Agent, Chain, Model
from yait_aichain.tools import Tool, chunk
from yait_aichain.utilities import MCP, REST, Shell, Utility, Vectors


def _vectors():
    """A Vectors over a fake store — no database needed."""
    kb = Vectors.__new__(Vectors)
    kb.store = MagicMock()
    from yait_aichain.utilities.vectors import VectorQueryTool
    kb.query = VectorQueryTool(kb.store)
    return kb


class TestVectors(unittest.TestCase):

    def test_methods_reach_the_store(self):
        kb = _vectors()
        kb.store.upsert.return_value = 2
        self.assertEqual(kb.upsert([{"id": "a", "text": "x"}, {"id": "b", "text": "y"}]), 2)
        kb.fetch(["a"]); kb.delete(["a"]); kb.count(); kb.create(1536)
        kb.store.fetch.assert_called_once_with(["a"])
        kb.store.delete.assert_called_once_with(["a"], None)
        kb.store.create.assert_called_once_with(1536, "cosine")

    def test_query_is_its_tool_and_the_rest_are_not(self):
        kb = _vectors()
        self.assertIsInstance(kb.query, Tool)
        self.assertEqual(kb.TOOLS, ("query",))
        self.assertEqual(kb.tools, [kb.query])
        for method in (kb.upsert, kb.delete, kb.fetch, kb.count, kb.create):
            self.assertFalse(hasattr(method, "run"), method)      # only a tool has .run

    def test_it_is_made_on_vectordb(self):
        with patch("yait_aichain.utilities.vectors.VectorDB") as vdb:
            kb = Vectors("qdrant", "docs", description="HR policies, 2024–2026", embedder="E",
                         url="http://q", index="i")
        vdb.assert_called_once_with("qdrant", "docs", embedder="E", url="http://q", api_key=None, index="i")
        self.assertIs(kb.store, vdb.return_value)

    def test_the_search_says_what_it_searches(self):
        with patch("yait_aichain.utilities.vectors.VectorDB"):
            kb = Vectors("qdrant", "docs", description="HR policies, 2024–2026")
            self.assertTrue(kb.query.description.startswith("Search HR policies, 2024–2026 — "))
            with self.assertRaises(TypeError):
                Vectors("qdrant", "docs")
            with self.assertRaisesRegex(ValueError, "description of what the database holds"):
                Vectors("qdrant", "docs", description=" ")


class TestMCP(unittest.TestCase):

    def _mcp(self):
        tools = [MagicMock(spec=Tool), MagicMock(spec=Tool)]
        tools[0].name, tools[1].name = "ask", "read"
        tools[0].run.return_value = "answer"
        return MCP("https://example.com/mcp", filter=["ask", "read"]), tools

    def test_nothing_is_asked_until_used(self):
        with patch("yait_aichain.utilities.mcp.MCPTools") as discover:
            m = MCP("https://example.com/mcp")
            discover.assert_not_called()
            self.assertEqual(m.TOOLS, ())

    def test_tools_names_tool_and_call(self):
        m, tools = self._mcp()
        with patch("yait_aichain.utilities.mcp.MCPTools", return_value=tools) as discover:
            self.assertEqual(m.names, ["ask", "read"])
            self.assertIs(m.tool("ask"), tools[0])
            self.assertEqual(m.call("ask", question="q"), "answer")
            tools[0].run.assert_called_once_with(question="q")
            self.assertEqual(discover.call_count, 1)                    # asked once
            self.assertEqual(discover.call_args.kwargs["filter"], ["ask", "read"])
            with self.assertRaisesRegex(KeyError, r"no tool 'write'; it has \['ask', 'read'\]"):
                m.tool("write")


class TestREST(unittest.TestCase):

    def test_endpoints_share_the_apis_url_auth_and_headers(self):
        api = REST("https://example.com/api/", description="Example bookings API",
                   auth={"type": "bearer", "env": "EXAMPLE_TOKEN"}, headers={"Accept": "application/json"})
        get = api.endpoint("get_booking", "GET", "/booking/{id}", path_params=["id"], description="One booking.")
        put = api.endpoint("update_booking", "PUT", "booking/{id}", path_params=["id"], body_params=["name"],
                           description="Rename a booking.")
        self.assertEqual(get._url_template, "https://example.com/api/booking/{id}")
        self.assertEqual(put._url_template, "https://example.com/api/booking/{id}")
        for t in (get, put):
            self.assertEqual(t._static_headers, {"Accept": "application/json"})
            self.assertEqual(t._auth["env"], "EXAMPLE_TOKEN")
        self.assertIs(api.get_booking, get)
        # the model reads which API, then what the endpoint does
        self.assertEqual(get.description, "Example bookings API — One booking.")
        self.assertEqual(api.tools, [get, put])
        self.assertEqual(api.TOOLS, ("get_booking", "update_booking"))

    def test_the_api_must_say_what_it_is(self):
        with self.assertRaises(TypeError):
            REST("https://example.com")
        with self.assertRaisesRegex(ValueError, "needs a description of the API"):
            REST("https://example.com", description="  ")
        with self.assertRaisesRegex(ValueError, "endpoint 'x' needs a description"):
            REST("https://example.com", description="Example").endpoint("x", "GET", "/x", description="")

    def test_mistakes_are_said(self):
        api = REST("https://example.com", description="Example")
        api.endpoint("x", "GET", "/x", description="X.")
        with self.assertRaisesRegex(ValueError, "already has an endpoint named 'x'"):
            api.endpoint("x", "GET", "/y", description="Y.")
        with self.assertRaisesRegex(ValueError, "http or https base URL"):
            REST("ftp://example.com", description="Example")
        with self.assertRaisesRegex(AttributeError, "no attribute or endpoint 'y'"):
            api.y


class TestShell(unittest.TestCase):

    def test_four_tools_confined_to_the_folder(self):
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "a.txt"), "w") as fh:
            fh.write("hello\n")
        sh = Shell(d)
        self.assertEqual(sh.TOOLS, ("browse", "read", "write", "run"))
        self.assertTrue(all(isinstance(t, Tool) for t in sh.tools))
        self.assertEqual(sh.folder, os.path.realpath(d))
        self.assertIn("hello", sh.read.run("a.txt"))
        with self.assertRaises(Exception):
            sh.read.run("../../etc/passwd")


class TestAnAgentTakesTheToolsNotTheUtility(unittest.TestCase):

    def setUp(self):
        self.model = Model("gpt-5.4-mini", api_key="k")

    def test_a_method_is_refused_and_the_tools_named(self):
        kb = _vectors()
        with self.assertRaisesRegex(TypeError, r"Vectors\.upsert is a method of a utility.*Its tools are query"):
            Agent(self.model, tools=[kb.upsert])

    def test_a_utility_is_refused_and_the_tools_named(self):
        with self.assertRaisesRegex(TypeError, r"Shell is a utility, not a tool.*browse, read, write, run"):
            Agent(self.model, tools=[Shell(tempfile.mkdtemp())])

    def test_its_tools_are_taken(self):
        sh = Shell(tempfile.mkdtemp())
        Agent(self.model, tools=[_vectors().query, sh.browse, sh.read])
        Agent(self.model, tools=sh.tools)

    def test_every_utility_is_a_utility(self):
        for cls in (Vectors, MCP, REST, Shell):
            self.assertTrue(issubclass(cls, Utility))


class TestAChainTakesAMethod(unittest.TestCase):

    def test_chunk_then_upsert(self):
        kb = _vectors()
        kb.store.upsert.return_value = 3
        res = Chain(steps=[(chunk(max_chars=400, overlap_chars=0), "pieces", {"input": "text"}),
                           (kb.upsert, "count", {"records": "pieces"})]).run(
            variables={"text": "# T\n\n" + "word " * 200})
        self.assertEqual(res["count"], 3)
        self.assertEqual(len(kb.store.upsert.call_args.args[0]), 3)

    def test_a_parameter_takes_its_namesake_and_a_missing_one_is_said(self):
        def add(a, b=1):
            return a + b
        self.assertEqual(Chain(steps=[(add, "s")]).run(variables={"a": 2})["s"], 3)
        with self.assertRaisesRegex(ValueError, "add needs 'a'"):
            Chain(steps=[(add, "s")]).run(variables={})
        with self.assertRaisesRegex(ValueError, r"maps \['c'\] into 'add'"):
            Chain(steps=[(add, "s", {"c": "a"})]).run(variables={"a": 2})

    def test_it_cannot_be_saved(self):
        path = os.path.join(tempfile.mkdtemp(), "c.yaml")
        with self.assertRaisesRegex(TypeError, "'Vectors.upsert' is a function or a utility's method"):
            Chain(steps=[(_vectors().upsert, "n", {"records": "x"})]).save(path)




if __name__ == "__main__":
    unittest.main()
