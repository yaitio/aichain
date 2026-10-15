# From laptop to production

The code you tested is the code you ship. A pipeline or an agent is written
once, as a scenario — what it does: its steps, prompts, tools. Everything that
differs between a notebook and production is the environment — where it lives
— and is handed in from outside, never written into the scenario. Moving to
production is changing what is handed in, not editing what was tested.

```python
def build(store, hooks):                         # the scenario: one definition, used in both
    return Chain(steps=[(draft, "reply"), (gate(send), "sent")],
                 store=store, hooks=hooks, max_cost=0.20)

build(InMemoryStore(), [])                       # on a laptop
build(MyTableStore(table), [my_tracer])          # in production: a StateStore and a hook you write
```

The test of the design is that line: if moving the same chain to production
needs an edit anywhere but its arguments, the environment has leaked into the
scenario.

---

## The six things that change

| | On a laptop | In production | Handed in as |
|---|---|---|---|
| **What starts a run** | `chain.run()` from a script | a webhook, a cron tick, a queue message | your handler calls `run()`, or `resume(run_id, signal)` on a run `arm()` parked |
| **Where state lives** | memory, a file | a table, a bucket — survives restarts, shared by instances | `store=` — any `StateStore`: save, load, delete |
| **How it runs** | one process, one run | many at once, cold starts, no shared memory | nothing to hand in: synchronous, no global state, MCP connected per call |
| **Whose keys** | `.env` | a vault, per tenant, rotated | `Model(api_key=callable)` with a `RunContext` |
| **What you can see** | the console | logs, traces, metrics in a sink | `hooks=` — every step boundary is an `Event` |
| **What it may cost** | whatever the run costs | a limit per request, per tenant | `max_cost=` — one `Budget`, shared by every step |

## State is a document, so a run outlives its process

A Chain's run is a document: the variables so far, each step's status. An
Agent's state is its conversation, a message list. Both serialise as they are,
so a run can start in one invocation, park, and finish in another that shares
nothing with the first but the store and the run's id. The fifteen-minute
limit of a serverless function becomes the size of a checkpoint instead of a
ceiling on the run: a chain parks at a `wait()`; an agent can be driven one
turn per invocation. → [Pause and resume](pipelines/pause-and-resume.md) ·
[Driving the loop](agents/driving-the-loop.md)

A trigger that delivers twice is normal in production. A `run_id` that
already finished is no longer in the store and `resume()` raises `KeyError`, so
the second delivery can be read as "already handled" instead of running the
tail twice.

## Keys per request, and never in the run

One deployment serves many tenants, each with its own provider keys. The run
carries the tenant's **name**, in a `RunContext`; the key is resolved from it
when a request is made:

```python
model = Model("gpt-6-luna", api_key=lambda ctx: vault.key_for(ctx.tenant))

chain.run(variables={...}, context=RunContext(tenant="acme"))   # a Chain takes it directly
with using(RunContext(tenant="acme")):                          # anything else, for a block
    skill.run()
```

The callable is asked once per request, so one `Model` serves every tenant. A
key is never put in the context, because the context is written into the run
document and a file store writes that to disk. A resolver that returns nothing
raises — it does not fall back to the process-wide key, which would send one
tenant's request under another's credential and get a normal answer.

The context is ambient for the run, not a parameter of every primitive: a
`context=` on each Skill and tool would be the environment leaking into the
scenario. A `Pool` carries it into each worker thread itself.

## Seeing and limiting

Hooks receive an event at every boundary — a run, a step, a model call, a tool
call, an approval — with the run's id, the arguments and the raw result, so a
tracer can rebuild a turn and a UI can show it as it happens. What a sink does
with them is yours: the library emits, the environment decides where it goes.
→ [Observability](control/observability.md)

`max_cost=` is the limit a request may spend, in dollars, shared by everything
the run does; a tenant's budget is a `Budget` you build per request.
→ [Cost and budget](control/cost-and-budget.md)

## Testing without a provider

A test should not need a key, a network or a bill. `yait_aichain.testing` has
two stand-ins that go through the library's own request and response path, so
what they exercise is the code that runs in production.

`ScriptedModel` replays a script — answers, and tool calls written as
`call(name, **arguments)` — and works wherever a `Model` does, `stream()`
included. Give it `usage=` and `price=` and every call has a cost, so a budget
or a report has something to count. `.requests` keeps what the loop sent.

```python
from yait_aichain import Agent
from yait_aichain.testing import ScriptedModel, call

model = ScriptedModel([call("issue_refund", amount=42), "Refunded."])
result = Agent(model, tools=[issue_refund]).run("Refund order 7")
assert result.output == "Refunded."
assert model.requests[1].messages[-1]["role"] == "tool"
```

`recorded(model)` keeps a real `Model` — its name, options, adaptations and
price — and replaces only its transport, so you can check what a request
would have carried: the body, the headers (an auth header as a fingerprint,
never the key) and what was adapted on the way.

```python
from yait_aichain import Model, Skill
from yait_aichain.testing import recorded

model = Model("gpt-5.5", options={"top_p": 0.9})
with recorded(model) as rec:
    Skill(model, prompt="Hi").run()
assert rec.requests[0].body["model"] == "gpt-5.5"
```

→ [API: Testing](api/testing.md)

## What the library does not do

It does not deploy. There is no daemon, no scheduler, no queue in it: a run is
started by whatever calls it, and parks rather than waits. That is on purpose
— a library that brings its own runtime is one you have to run — and it is why
the same code runs in a script, a container and a Lambda function.

The deployment itself — the agent API, sessions and their event log, stores,
vaults, triggers, one command into your own AWS account — is a separate
project, `agents`, built on this library.
