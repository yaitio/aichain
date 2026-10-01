"""
Live: a model on its provider's own endpoint calls a function tool.

Two of these failed on 2026-10-01 in the yait agents tool evals, each through a
different gap in this library, and each was papered over in the product:

  * gpt-6 was routed to Chat Completions, which answers function tools together
    with reasoning with HTTP 400 — fixed by routing every generation after
    gpt-4 to the Responses API;
  * Qwen on DashScope raised ``TypeError`` because its client's
    ``build_request`` took no ``tools`` — fixed in 3.1.0, and this is the live
    check the eval asked for.

The offline suite pins the wire shapes. What only a real call can show is that
the provider accepts them and the reply comes back as a tool call. Skipped
without the key; the class names end in ``Live``, so CI deselects them and the
nightly ``live`` job runs them.

    python -m pytest tests/clients/test_tools_live.py -m live -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from yait_aichain.agent import Agent, step_count
from yait_aichain.models import Model
from yait_aichain.tools._base import Tool


class Weather(Tool):
    name = "weather"
    description = "Current weather for a city."
    risk = "read"
    parameters = {"type": "object", "properties": {"city": {"type": "string"}},
                  "required": ["city"]}

    def __init__(self):
        self.asked = []

    def run(self, city, options=None):
        self.asked.append(city)
        return {"city": city, "celsius": 14, "sky": "rain"}


TASK = "What is the weather in Lisbon? Use the tool, then answer in one sentence."


class _ToolCallLive:
    MODEL = ""
    KEY = ""

    def setUp(self):
        # Several offline modules `setdefault` a fake key at import time, and a
        # live run collects them too — so "a key is set" is not "a real key is
        # set". Found on the first run: the fake key reached the network and the
        # test failed on authentication instead of skipping.
        if os.environ.get(self.KEY, "") in ("", "test-key", "k"):
            self.skipTest(f"{self.KEY} not set to a real key")

    def test_the_model_calls_the_tool_and_answers(self):
        weather = Weather()
        agent = Agent(Model(self.MODEL), tools=[weather], stop_when=[step_count(4)])
        result = agent.run(TASK)

        self.assertTrue(result.success, result.error)
        self.assertEqual(result.stopped_by, "answered")
        self.assertEqual([c.lower() for c in weather.asked], ["lisbon"])
        self.assertIn("14", result.output)


class TestGpt6ToolsLive(_ToolCallLive, unittest.TestCase):
    MODEL = "gpt-6-luna"
    KEY = "OPENAI_API_KEY"


class TestQwenToolsLive(_ToolCallLive, unittest.TestCase):
    MODEL = "qwen3.8-flash"
    KEY = "DASHSCOPE_API_KEY"


if __name__ == "__main__":
    unittest.main()
