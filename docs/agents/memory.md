# Agent memory

Every Agent has a shared key-value store — `agent.memory` — that every step reads from and writes to. It's the agent's scratchpad: seed data goes in, step outputs accumulate, and the full state is included in every orchestrator prompt so the planner always has complete context.

```python
result = agent.run(
    task      = "Write a 200-word summary of the topic in the target language.",
    variables = {"topic": "fusion energy", "language": "Spanish"},
)

result.memory
# {'topic': 'fusion energy', 'language': 'Spanish',
#  'search_results': '...', 'summary_es': '...'}
```

Memory lives for the duration of one `agent.run()` call by default. For state that survives across runs, use a persistent backend.

---

## Lifecycle

Every `agent.run(task, variables)` call does this at the start:

1. `memory.clear()` — wipe any prior state.
2. `memory.update(variables)` — seed with the passed variables.

Then, during execution:

3. Every **action prompt** includes the full memory as context.
4. Every **reflection** picks a `store_as` key — the step's output is written under that key (after being sanitised to snake_case).
5. Every **reflection prompt** also includes the full memory, so the orchestrator can reason about what's already been collected.

And finally:

6. When `run()` returns, `result.memory` is a snapshot of the final state.

Note: the `clear()` at the start means **reusing the same `AgentMemory` instance across runs wipes your state**. To persist across runs, use a `FileBackend` and instantiate a fresh `AgentMemory` from it each time.

---

## Reading and writing

From outside the agent:

```python
from agent import AgentMemory

memory = AgentMemory({"language": "Ukrainian"})

memory.set("topic", "AI safety")
memory.get("topic")            # "AI safety"
memory.get("missing", "n/a")   # "n/a"
memory.update({"k1": 1, "k2": 2})
"topic" in memory              # True
len(memory)                    # 4
memory.all()                   # {'language': ..., 'topic': ..., ...}
memory.keys() / .values() / .items()
memory.delete("topic")
memory.clear()
```

From inside a tool or skill, the agent can't touch `memory` directly — but orchestrator-generated prompts can reference any key with `{placeholder}` syntax. Missing keys are left as the literal `{key}` (they don't raise).

---

## Seeding memory

Two ways to put initial state in memory:

### 1. Via `agent.run(variables=...)` — the usual way

```python
agent.run(
    task      = "Research the topic in the target language.",
    variables = {
        "topic":    "quantum computing",
        "language": "Japanese",
        "horizon":  "18 months",
    },
)
```

These are merged in **after** `clear()`, so they're always present at step 1.

### 2. Via a pre-populated `AgentMemory`

```python
from agent import AgentMemory

memory = AgentMemory({
    "company_profile": "…",
    "product_catalog": "…",
})
agent = Agent(..., memory=memory)

result = agent.run(task="…")
```

This is useful when the initial state is too large or too structured for a `variables` dict, or when you want to share one memory across several agents in sequence.

> Either way, the agent's `run()` will call `clear()` first if the backend is volatile. When using a `FileBackend`, state is loaded from disk on construction and preserved across agent instances — see below.

---

## Pluggable backends

`AgentMemory` separates the in-process store from its persistence mechanism. Two backends are built-in; you can write your own.

### `InMemoryBackend` — default

Pure in-process. `clear()` wipes the dict; nothing is written anywhere. This is what you get when you don't pass a `backend` argument.

### `FileBackend` — durable JSON

Writes the full memory state to a JSON file atomically (temp file + rename, so a crash mid-write never corrupts the file).

```python
from agent import AgentMemory, FileBackend

backend = FileBackend("~/.my_agent.json")
memory  = AgentMemory(backend=backend)
```

On construction, `AgentMemory.__init__()` calls `backend.load()` and uses the result as the initial store. That's how persisted state "comes back" on a new `AgentMemory` instance.

### Persistence pattern across runs

`agent.run()` starts with `memory.clear()` — which wipes **both** the in-process store **and** the backing file (if the backend supports it). So reusing one `AgentMemory` instance across runs does not preserve state.

The correct pattern:

```python
from agent import AgentMemory, FileBackend

PATH = "~/.my_agent.json"

# ── First run ──
memory1 = AgentMemory(backend=FileBackend(PATH))
agent1  = Agent(..., memory=memory1)
result1 = agent1.run(task="Phase 1: gather raw data.")
memory1.flush()        # checkpoint final state to disk

# ── Second run — fresh AgentMemory, same file ──
memory2 = AgentMemory(backend=FileBackend(PATH))   # loads Phase 1's state
agent2  = Agent(..., memory=memory2)
result2 = agent2.run(task="Phase 2: build on what we found.")
memory2.flush()
```

Two rules:

1. Call `memory.flush()` **after** `run()` returns — this writes the current in-process state to the file. Without this, nothing persists.
2. Create a **new** `AgentMemory` at the start of each run. Reusing the same instance would trigger the `clear()` wipe.

### Writing a custom backend

Subclass `MemoryBackend` and implement three methods:

```python
from agent import MemoryBackend

class RedisBackend(MemoryBackend):
    def __init__(self, redis_client, key: str):
        self._r   = redis_client
        self._key = key

    def load(self) -> dict:
        raw = self._r.get(self._key)
        return json.loads(raw) if raw else {}

    def save(self, data: dict) -> None:
        self._r.set(self._key, json.dumps(data))

    def clear(self) -> None:
        self._r.delete(self._key)
```

Then pass it in:

```python
memory = AgentMemory(backend=RedisBackend(redis_client, "agent:session:42"))
```

---

## Memory in prompts

The orchestrator's planning, action, and reflection prompts all include the full memory as context. Two implications:

1. **Keep values reasonably sized.** Storing a 50 000-character web page in memory means that blob lands in every subsequent orchestrator prompt. Summarise first, then store the summary.
2. **Store the data, not the plan.** The orchestrator owns the plan. Memory holds facts, intermediate outputs, and reference material.

Key names from reflection's `store_as` are sanitised to snake_case automatically (`"Market Share!"` → `"market_share"`).

---

## Inspecting memory after a run

`result.memory` is a shallow copy of the final state:

```python
result = agent.run(task="…", variables={"topic": "…"})

print(result.memory.keys())
# dict_keys(['topic', 'search_results', 'vendor_a_profile',
#            'vendor_b_profile', 'final_table'])

for key, value in result.memory.items():
    print(key, "→", str(value)[:80])
```

This is the complete audit trail of what the agent retained. Pair it with `result.history` (what the agent *did*) and you have full observability of the run.

---

## Common patterns

### Pass a pre-built knowledge base in

```python
memory = AgentMemory({
    "company_profile": open("docs/company.md").read(),
    "product_catalog": open("docs/products.md").read(),
})
agent = Agent(
    orchestrator = Model("claude-opus-4-6"),
    memory       = memory,
    persona      = "You have full access to the company profile and product catalog in memory.",
)
result = agent.run(task="Draft a launch-announcement outline that reflects our positioning.")
```

### Accumulate across a pipeline of runs

```python
PATH = "./.agent_workspace.json"

def run_phase(task: str):
    memory = AgentMemory(backend=FileBackend(PATH))
    agent  = Agent(..., memory=memory)
    result = agent.run(task=task)
    memory.flush()
    return result

run_phase("Gather market data.")
run_phase("Interview the data you gathered and identify whitespace.")
run_phase("Draft a go-to-market brief using the whitespace analysis.")
```

Each phase sees everything the previous phases left behind.

---

## See also

- **Three-phase execution flow** → [Overview](overview.md)
- **Full configuration reference** → [Configuration](configuration.md)
- **Embedding an Agent in a Chain** → [Agent as Chain step](agent-as-chain-step.md)
