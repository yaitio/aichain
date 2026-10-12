# Permissions

A model decides which tool to call. Whether the call runs is decided outside
the model, before the tool starts, from two pieces of data: the tool's `risk`
class and the agent's `PermissionPolicy`.

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
              permissions=PermissionPolicy({"financial": "approve"}),
              approve=manager)
```

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
| `approve` | the agent asks `approve=`, and the tool runs only on a yes |
| `deny` | the tool never runs |

An agent with no `permissions=` runs every tool. With a policy, the rules you
do not give take the shipped default: `read`, `draft` and `write` run;
`external`, `financial` and `privileged` need approval; `destructive` is
denied.

**With `approve` in the policy and no `approve=` attached, the call is
refused** — a rule whose whole content is "ask a person" cannot mean "go
ahead" when there is no one to ask.

**Where the policy holds, in 4.0.** It is consulted in the agent's own loop.
The built-in `pool` tool, `Agent.execute()`, `resume()`, and tools run as
steps of a `Chain` or items of a `Pool` do not consult it yet — so a tool that
must never run without a person should not be reachable through them. This is
a known issue of 4.0.0, to be closed in 4.0.x.

## The approver

`approve=` gets an `ApprovalRequest` — the tool's name, its risk class, the
arguments and the call's id — and returns `True`, `False`, or
`ApprovalDecision(False, reason)`. A refusal reaches the model as the tool's
result, with the reason, so it can tell the user or try something else; the
run does not crash.

The request and the answer are also events, `approval.requested` and
`approval.decided`, on the same channel as everything else a run emits — which
is how a UI shows a person the question before the approver is asked.
→ [Observability](observability.md)

## When the answer comes later

`approve=` decides within one call of `run()`. When the person answers hours
later, in another process, the tool belongs behind a `gate(tool)` in a chain
or in the agent's hands, with a `store=`: the run parks, and resumes when the
signal arrives. → [Pause and resume](../pipelines/pause-and-resume.md)
