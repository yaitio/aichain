"""
Two runs of one agent do not see each other.

The run id, the parked document, the pending approval, the stream flag and
the hooks a stream collects through all lived on the `Agent`. One agent serves
many runs — a module-level agent in a warm Lambda, a Pool's items — so an
approver was shown one run's call while another's executed (approved 5,
executed 5000), and a tenant's stream carried another tenant's events. Each
run now works on its own copy of the agent's run state.
"""

import json
import os
import sys
import threading
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from yait_aichain import Agent, Model, PermissionPolicy  # noqa: E402
from yait_aichain.tools import tool  # noqa: E402


def _reply(*specs, text=None):
    msg = ({"content": None, "tool_calls": [
               {"id": i, "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}
               for i, n, a in specs]} if specs else {"content": text or "done"})
    return json.dumps({"choices": [{"message": msg}],
                       "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}).encode()


def _model_by_task():
    """A model that answers each run by the amount in its task: one call, then done."""
    m = Model("gpt-4o", api_key="k")

    def post(path, body, headers, *a, **k):
        msgs = body["messages"]
        task = next(x["content"] for x in msgs if x["role"] == "user")
        amount = int(task.split()[-1])
        if any(x["role"] == "tool" for x in msgs):
            return _reply(text=f"refunded {amount}")
        return _reply((f"c{amount}", "refund", {"amount": amount}))
    def sse(path, body, headers, *a, **k):
        message = json.loads(post(path, body, headers))["choices"][0]["message"]
        if message.get("content"):
            yield {"choices": [{"delta": {"content": message["content"]}}]}
        for index, tc in enumerate(message.get("tool_calls") or []):
            yield {"choices": [{"delta": {"tool_calls": [
                {"index": index, "id": tc["id"],
                 "function": {"name": tc["function"]["name"],
                              "arguments": tc["function"]["arguments"]}}]}}]}
        yield {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}

    m.client._post = MagicMock(side_effect=post)
    m.client._post_sse = MagicMock(side_effect=sse)
    m.client._auth_headers = MagicMock(return_value={})
    return m


executed: list = []
_lock = threading.Lock()


@tool(risk="financial")
def refund(amount: int) -> str:
    """Refund an amount."""
    with _lock:
        executed.append(amount)
    return f"refunded {amount}"


class TestTwoStreamsInOneThread(unittest.TestCase):

    def test_interleaved_streams_keep_their_own_events(self):
        agent = Agent(_model_by_task(), tools=[refund])
        a, b = agent.stream("refund 5"), agent.stream("refund 5000")
        seen = {"a": [], "b": []}
        pending = {"a": a, "b": b}
        while pending:
            for key in list(pending):
                try:
                    seen[key].append(next(pending[key]))
                except StopIteration:
                    del pending[key]
        for key, amount in (("a", 5), ("b", 5000)):
            ids = {e.run_id for e in seen[key] if e.run_id}
            self.assertEqual(len(ids), 1, f"stream {key} carried events of {len(ids)} runs")
            calls = [e.payload.get("arguments") for e in seen[key] if e.type == "tool_call.started"]
            self.assertEqual(calls, [{"amount": amount}])
        self.assertEqual(agent.hooks, [])                       # nothing left behind


class TestApprovalsDoNotCross(unittest.TestCase):

    def test_what_was_approved_is_what_runs(self):
        executed.clear()
        approved: list = []
        barrier = threading.Barrier(2)

        def hold(event):
            # Both runs stop between asking and being answered — the window in
            # which one run's request overwrote the other's on the instance.
            if event.type == "approval.requested":
                try:
                    barrier.wait(timeout=2)
                except threading.BrokenBarrierError:
                    pass

        def approve(request):
            approved.append(request.arguments["amount"])
            return request.arguments["amount"] < 100

        agent = Agent(_model_by_task(), tools=[refund], hooks=[hold],
                      permissions=PermissionPolicy({"financial": "approve"}), approve=approve)
        results = {}
        threads = [threading.Thread(target=lambda n=n: results.update({n: agent.run(f"refund {n}")}))
                   for n in (5, 5000)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        self.assertEqual(sorted(approved), [5, 5000])
        self.assertEqual(executed, [5])
        self.assertTrue(results[5].success and results[5000].success)


class TestAPoolLeavesTheAgentAsItWas(unittest.TestCase):

    def test_the_pools_budget_is_not_left_on_the_agent(self):
        from yait_aichain import Budget, Pool
        agent = Agent(_model_by_task(), tools=[refund])
        before = agent.max_cost
        Pool(agent, items=[{"task": "refund 1"}], max_cost=Budget(1.0)).run()
        self.assertIs(agent.max_cost, before)


if __name__ == "__main__":
    unittest.main()
