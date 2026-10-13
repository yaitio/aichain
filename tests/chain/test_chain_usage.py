"""
What a chain reports it spent is what its steps spent.

An agent step added its tokens and not its cost, so `ChainResult.cost` was
None or short where a Pool counting the same agent was right; a Skill shared
by a Pool's copies of one chain had its `last_usage` read after another
thread overwrote it; and a pause kept four of `Usage`'s seven fields, losing
the cache and image counts.
"""

import os
import threading
import unittest

from yait_aichain import Chain, Pool
from yait_aichain.agent._result import AgentResult
from yait_aichain.agent import Agent
from yait_aichain.models._usage import Usage
from yait_aichain.chain._chain import _usage_to_dict


class _Agent(Agent):
    def __init__(self):
        self.max_cost = None

    def run(self, task, variables=None):
        return AgentResult(success=True, output="done", mode="agile", steps_taken=1,
                           tokens_used=100, cost=0.25)


class _Counting:
    """A Skill-shaped step whose usage depends on its input, slowly."""
    name = "counting"

    def __init__(self):
        self.last_usage = None
        self.max_cost = None

    def run(self, variables=None):
        n = int(variables["n"])
        self.last_usage = Usage(total_tokens=n)
        threading.Event().wait(0.002)            # another thread gets in here
        return n


class TestUsage(unittest.TestCase):

    def test_an_agent_steps_cost_is_counted(self):
        self.assertEqual(Chain(steps=[(_Agent(), "a")]).run({"task": "x"}).cost, 0.25)

    def test_a_shared_skill_in_a_pool_of_chains_counts_its_own(self):
        chain = Chain(steps=[(_Counting(), "out")])
        result = Pool(chain, [{"n": i} for i in range(1, 41)], max_flows=8).run()
        self.assertEqual(result.usage.total_tokens, sum(range(1, 41)))

    def test_every_field_survives_a_pause(self):
        u = Usage(input_tokens=1, output_tokens=2, total_tokens=3, cost=0.1,
                  cache_write_tokens=4, cache_read_tokens=5, image_input_tokens=6)
        self.assertEqual(Usage(**_usage_to_dict(u)), u)


if __name__ == "__main__":
    unittest.main()
