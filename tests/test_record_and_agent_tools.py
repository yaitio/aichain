"""
A tool's result as one object for three readers, and an agent that refuses
what cannot be a tool — both 3.3.0.

Record: code reads the fields, an agent shows the model ``str(record)``, a
Skill prompt substitutes the same text, and a Chain keeps the record under its
step's key where a plain dict is merged.

Agent: a non-tool used to pass construction and fail on the first turn with an
``AttributeError``; two tools with one name used to be kept as one, the second
replacing the first in silence. Both now fail at ``Agent(...)``, by name.
"""

import copy
import json
import os
import pickle
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from yait_aichain import Agent, Chain, Model, Skill
from yait_aichain.agent import step_count
from yait_aichain.agent._prompts import result_message
from yait_aichain.tools import Record, Reranker, Tool, search


def _render(d):
    return "found: " + ", ".join(d["results"])


class Find(Tool):
    name = "find"; description = "Find."; risk = "read"
    parameters = {"type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"]}

    def run(self, input, options=None):
        return Record({"results": [input], "cost": 0.01}, _render)


class Fields(Tool):
    name = "fields"; description = "Fields."; risk = "read"
    parameters = {"type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"]}

    def run(self, input, options=None):
        return {"city": input}


class TestRecord(unittest.TestCase):

    def test_a_dict_for_code_and_text_for_a_reader(self):
        r = Record({"results": ["a", "b"]}, _render)
        self.assertEqual(r["results"], ["a", "b"])
        self.assertEqual(str(r), "found: a, b")
        self.assertEqual(json.loads(json.dumps(r)), {"results": ["a", "b"]})

    def test_copies_keep_the_rendering_and_pickle_is_the_plain_dict(self):
        r = Record({"results": ["a"]}, _render)
        self.assertEqual(str(copy.copy(r)), "found: a")
        self.assertEqual(str(copy.deepcopy(r)), "found: a")
        back = pickle.loads(pickle.dumps(r))
        self.assertIs(type(back), dict)               # a renderer is not data
        self.assertEqual(back, {"results": ["a"]})

    def test_an_agent_shows_the_model_its_text_not_its_json(self):
        self.assertEqual(result_message(Record({"results": ["a"]}, _render)), "found: a")
        self.assertIn('"city"', result_message({"city": "Paris"}))   # a plain dict is still JSON

    def test_a_chain_keeps_a_record_under_its_key_and_merges_a_dict(self):
        res = Chain(steps=[(Find(), "one", {"input": "q"}), (Find(), "two", {"input": "q"}),
                           (Fields(), "ignored", {"input": "q"})]).run(variables={"q": "fox"})
        self.assertEqual(res["one"]["results"], ["fox"])
        self.assertEqual(res["two"]["results"], ["fox"])          # not overwritten by step one
        self.assertEqual(res["city"], "fox")                       # a plain dict, merged as before

    def test_a_skill_prompt_gets_the_text(self):
        model = Model("gpt-4o", api_key="k")
        sent = []

        def post(path, body, headers, *a, **k):
            sent.append(body)
            return json.dumps({"choices": [{"message": {"content": "ok"}}],
                               "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}).encode()

        model.client._post = MagicMock(side_effect=post)
        model.client._auth_headers = MagicMock(return_value={})
        Chain(steps=[(Find(), "found", {"input": "q"}),
                     (Skill(model, prompt="Summarise: {found}"), "summary")]).run(variables={"q": "fox"})
        self.assertIn("Summarise: found: fox", json.dumps(sent[-1]))


class TestAgentRefusesWhatCannotBeATool(unittest.TestCase):

    def setUp(self):
        self.model = Model("gpt-5.4-mini", api_key="k")

    def test_two_tools_with_one_name(self):
        with self.assertRaisesRegex(ValueError, "two tools are named 'searchExa'.*name="):
            Agent(self.model, tools=[search("exa", api_key="k"), search("exa", api_key="k")])

    def test_named_apart_they_are_two_tools(self):
        agent = Agent(self.model, tools=[search("exa", api_key="k"),
                                         search("exa", api_key="k", name="search_again")])
        self.assertEqual(sorted(agent._tool_map), ["searchExa", "search_again"])

    def test_a_non_tool_by_its_class(self):
        with self.assertRaisesRegex(TypeError, "RerankCohere is not a tool"):
            Agent(self.model, tools=[Reranker("rerank-v3.5", api_key="k")])

    def test_a_skill_says_where_it_belongs(self):
        with self.assertRaisesRegex(TypeError, "a step for a Chain, not a tool for an Agent"):
            Agent(self.model, tools=[Skill(self.model, prompt="x {input}")])

    def test_a_duck_typed_tool_is_still_a_tool(self):
        class Duck:
            name = "duck"
            parameters = {"type": "object", "properties": {}}
            def schema(self): return {"type": "function", "function": {"name": "duck", "parameters": self.parameters}}
            def run(self, **kw): return "quack"
            def check_args(self, kw): return None
        Agent(self.model, tools=[Duck()], stop_when=[step_count(1)])


if __name__ == "__main__":
    unittest.main()
