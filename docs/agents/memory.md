# AgentMemory — a key-value store you can hand to a tool

> **The Agent does not have one.** Rewritten 2026-09-11: this page described a
> memory subsystem the agent lost in `2.0.0` — `agent.memory`, seeding from
> `variables`, a `store_as` key per step, the whole state pasted into every
> prompt. The word `memory` does not appear in the agent's source at all.
>
> Its state **is** the conversation: a message list that already holds every
> attempt with what came back, is serialisable by construction, and grows by
> appending — which is also the shape a provider's prefix cache rewards. A
> second store beside it was a copy of that with an extra way to disagree.
> `AgentResult.memory` survives as a field and is never written to.

What remains is `AgentMemory` itself: a small, honest key-value store with a
pluggable backend. Nothing in the loop reads or writes it, so it is yours to
use — from inside your own tools, across runs, or as scratch space in a
`Chain`.

```python
from yait_aichain.agent import AgentMemory

memory = AgentMemory({"topic": "fusion energy"})
memory.set("summary", "…")
memory.get("topic")                 # "fusion energy"
memory.all()                        # {'topic': …, 'summary': …}
```

`set` / `get` / `update` / `delete` / `keys` / `values` / `items` / `all` /
`clear` / `reset` / `flush` — a dict with a persistence seam, deliberately not
more.

---

## Backends

`AgentMemory` separates the in-process store from how it persists.

### `InMemoryBackend` — default

Pure in-process. `clear()` wipes the dict; nothing is written anywhere. This
is what you get when you pass no backend.

### `FileBackend` — durable JSON

Writes the whole state to a JSON file atomically — temp file plus rename, so a
crash mid-write cannot corrupt it.

```python
from yait_aichain.agent import AgentMemory, FileBackend

memory = AgentMemory(backend=FileBackend("~/.my_agent.json"))
```

On construction `AgentMemory` calls `backend.load()` and uses the result as
its initial contents. That is how persisted state comes back on a new
instance.

### Your own

Subclass `MemoryBackend` and implement `load()`, `save(state)`, `clear()` —
three methods over S3, DynamoDB, Redis, anything.

---

## Using it from a tool

The loop passes nothing to it, so a tool that needs shared state closes over
one:

```python
notes = AgentMemory(backend=FileBackend("notes.json"))

class Remember(Tool):
    name = "remember"
    parameters = {"type": "object",
                  "properties": {"key": {"type": "string"},
                                 "value": {"type": "string"}},
                  "required": ["key", "value"]}
    def run(self, key, value, options=None):
        notes.set(key, value)
        return f"stored {key!r}"
```

Two things worth knowing before reaching for this. A model that *can* write
notes will spend tokens writing them, and in the `scaffold` measurements three
roles asked in their prompts to hand work over through files did not do it
once — an instruction is not a mechanism. And anything a tool returns is
already in the conversation, so storing it again buys persistence across runs
and nothing else.

---

## See also

- [Configuration](configuration.md) — the parameters that exist
- [Observability](observability.md) — the journal, which *is* the loop's record
- [State](../primitives/state.md) — `Chain`'s store, suspend and resume
