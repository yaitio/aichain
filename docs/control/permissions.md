# Permissions

A model decides which tool to call. Whether the call runs is decided outside
the model, before the tool starts, from two pieces of data: the tool's `risk`
class and the `policy` of whatever runs it — an `Agent`, a `Chain`, a `Pool`.

```python
from yait_aichain import Agent, Model
from yait_aichain.tools import ApprovalDecision, PermissionPolicy, tool

@tool(risk="financial")
def issue_refund(order: int, amount: float) -> str:
    """Refund an order.

    order: the order's id.
    amount: how much to refund.
    """
    return f"refunded {amount} on {order}"

def manager(request):                     # what the agent asks before the call runs
    if request.arguments["amount"] <= 50:
        return True
    return ApprovalDecision(False, "refunds over $50 need a ticket")

agent = Agent(Model("claude-haiku-5-5"), tools=[issue_refund],
              policy={"rules": {"financial": "approve"}, "approve": manager})
```

`policy` holds everything that decides what happens to a call, and its keys
are checked when the agent is built — a misspelt `"aprove"` is refused, not
ignored:

| key | what it is |
|---|---|
| `rules` | a risk class → `allow`, `approve` or `deny` |
| `default` | the decision for a class no rule names |
| `approve` | who answers an `approve`: a function, `"park"`, or `"deny"` |
| `interceptors` | objects that act on each call before and after it runs |

A `PermissionPolicy` or a plain dict of rules is the `rules` alone.

---

## Risk is data on the tool

| `risk` | For |
|---|---|
| `read` | looks, changes nothing |
| `draft` | prepares something a person will review |
| `write` | changes something — **the default** for every tool |
| `external` | reaches outside: sends a message, calls a third party |
| `financial` | moves money |
| `destructive` | deletes, or cannot be undone |
| `privileged` | acts with more rights than the caller has |

`@tool(risk=…)` sets it on a function, a class attribute on a `Tool` subclass.
The default is `write` because the library cannot tell what your code does;
saying `read` is your claim to make.

The library's own tools say what they risk, so a policy holds for them too:
search, fetch, embeddings and reranking are `read`, images are `draft`.
`Shell`'s `browse` and `read` are `read`, `write` is `write`, and `run` is
`privileged` — it runs any command with your process's rights. An MCP tool
takes its class from the server's annotations and is `privileged` when the
server says nothing; `MCP(server, risk={...})` overrides it. A REST endpoint
follows its method — `GET` is `read`, `DELETE` is `destructive`, a write is
`external` — and `endpoint(..., risk=…)` overrides it.

## The policy decides

| Decision | What happens |
|---|---|
| `allow` | the tool runs |
| `approve` | the policy's `approve` answers, and the tool runs only on a yes |
| `deny` | the tool never runs |

An agent with no `policy` runs every tool. With one, the rules you
do not give take the shipped default: `read`, `draft` and `write` run;
`external`, `financial` and `privileged` need approval; `destructive` is
denied.

**With an `approve` rule and no one to answer it, the call is refused** — a
rule whose whole content is "ask a person" cannot mean "go ahead" when there
is no one to ask.

**Where the policy holds.** Everywhere a tool runs: the agent's loop, its
`execute()` seam, `resume()`, the built-in `pool`, `delegate` and
`write_plan` (asked about as `draft`), a `Chain` step, a `Pool` item. And the
rules around a call govern what it starts: a pipeline run inside an agent's
tool, or a team member, decides under the stricter of its own rules and the
agent's, and asks the agent's approver when it has none.

## The approver

A function in `approve` gets an `ApprovalRequest` — the tool's name, its risk class, the
arguments and the call's id — and returns `True`, `False`, or
`ApprovalDecision(False, reason)`. A refusal reaches the model as the tool's
result, with the reason, so it can tell the user or try something else; the
run does not crash.

The request and the answer are also events, `approval.requested` and
`approval.decided`, on the same channel as everything else a run emits — which
is how a UI shows a person the question before the approver is asked.
→ [Observability](observability.md)

## When the answer comes later

A function decides within one call of `run()`. When the person answers hours
later, in another process, say `"approve": "park"` and give the agent a
`store=`: the run parks with the question, and
`agent.resume(run_id, signal={"approved": True})` runs the call — or `False`,
with a `"reason"`, refuses it. A `gate(tool)` does the same for one tool, in a
chain or in an agent's hands. → [Pause and resume](../pipelines/pause-and-resume.md)

## Interceptors

Where a hook watches, an interceptor acts. `before(call, run)` may return a
`Block(reason)` to stop the call, or the call with other arguments to run that
instead — before the rules, so they decide on what will really run.
`after(call, result, run)` may return another result. Every change goes into
the run's journal and an `interceptor.changed` event, with what it was and
what it became; one that raises blocks the call rather than being skipped.

```python
from yait_aichain import Block, Interceptor

class NoFridays(Interceptor):
    def before(self, call, run):
        if call.name == "deploy" and is_friday():
            return Block("no deploys on a Friday")

agent = Agent(model, tools=[deploy], policy={"interceptors": [NoFridays()]})
```
