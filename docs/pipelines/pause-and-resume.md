# Pause and resume

Some runs have to wait: for a manager to approve a refund, for a customer to
reply, for a webhook to say the payment cleared, for tomorrow morning. A
process cannot sit there for a day, and a serverless function cannot sit there
for more than a few minutes. So a run that waits does not wait — it **parks**:
it writes down where it is, returns, and is picked up later by whatever
receives the answer, in another process if need be.

```python
chain = Chain(steps=[
    (draft, "reply"),
    (wait(reason="A person approves the reply", resume_with={"approved": "bool"}), "decision"),
    (send, "sent"),
], store=FileStore("runs/"))

paused = chain.run(variables={"complaint": "My order arrived late."})
paused.run_id, paused.awaiting["reason"]

# later, anywhere that shares the store:
chain.resume(paused.run_id, signal={"approved": True})
```

→ Every parameter: [API: State](../api/state.md) · [API: wait and gate](../api/tools/wait-gate.md)

---

## Two ways to pause

**`wait()`** is a step that parks the run until a signal arrives. On resume
its output is the signal, so the steps after it read what the person or the
webhook sent.

**`gate(tool)`** wraps any tool behind a decision. The run parks before the
tool; on resume the tool runs only if the signal approves — `{"approved":
True}` by default — and is skipped otherwise. A gate takes the wrapped tool's
schema, so it drops into a chain or an agent's tools in its place.

Both are ordinary tools. In a chain they are steps you put where the run must
wait. In an agent's tools they are something the **model** may call, when it
decides the world has to answer before it can go on.

## The store is the run

A paused run is a document: the variables computed so far, each step's
status, and enough of the definition to continue. Parking writes it to the
store; resuming reads it, hands the signal to the waiting step, and continues
from the first step not yet done. Finished steps are never run again — their
outputs are already in the document — and nothing is replayed from the top.

| Store | When |
|---|---|
| `InMemoryStore()` | the default: the answer comes back within this process |
| `FileStore("runs/")` | the answer comes in another process on the same disk, or after a restart |
| your own `StateStore` | S3, DynamoDB, Redis — anything with save, load and delete |

The default is in memory, and that is the mistake to avoid: a chain with a
`wait()` and no store warns when it is built, and an agent given a pausing
tool with no store refuses to be built, because a pause nobody can resume is
otherwise discovered at the worst moment — after someone answered.

## Across processes

The point of all this is that the two halves share nothing but the store and
the run's id:

```python
# invocation 1 — starts the run; it parks
def start(order):
    chain = build_chain(store=FileStore("/srv/runs"))
    return chain.run(variables={"order": order}).run_id      # keep this id

# invocation 2 — a webhook, hours later
def on_approval(run_id, approved):
    chain = build_chain(store=FileStore("/srv/runs"))         # a new object, same store
    return chain.resume(run_id, signal={"approved": approved})
```

**Resuming is idempotent.** The store holds only parked runs; a run that
finished is deleted. A `run_id` that is no longer there raises `KeyError`, so a
trigger that delivers twice — and webhooks do — can read that as "already
handled" instead of running the tail twice.

## A run that has not started

`arm()` creates a run parked before its first step: an id, a document in the
store, and the shape of the signal that will start it. That is a
subscription — what a trigger waits on, and what a person is shown a form for.

```python
armed = chain.arm(reason="waiting for an order", resume_with={"topic": "str"},
                  hint={"on": "order.created"})
chain.resume(armed.run_id, signal={"topic": "solar power"})    # the order arrived
```

The signal seeds the variables, since nothing has run yet. For an agent, the
signal is the task. `hint` is recorded for whatever does the waking and never
acted on: the library parks runs and schedules nothing. What it buys is one
shape for three things usually built apart — a run waiting to start, waiting at
a gate, and waiting for an answer the model asked for all end in
`resume(run_id, signal)`.

## What it is not

A pause is not a snapshot of an object. An agent's state is its conversation —
a message list — so an agent parks the same way a chain does, through a
`wait()` or `gate()` in its tools and a `store=`, and there is no
`agent.save()`. Nothing stops a run from outside it either: a run parks only
where one of its own steps asked the world for something.

A person deciding within the same call needs none of this: an `approve` function in the
agent's `policy` asks them before the tool runs, and returns. Parking is for
answers that outlive the process — `policy={"approve": "park"}` parks any call
that needs one. → [Permissions](../control/permissions.md)

---

▶ [`examples/23_chain_human.py`](../../examples/23_chain_human.py) ·
[`examples/24_chain_webhook.py`](../../examples/24_chain_webhook.py)
