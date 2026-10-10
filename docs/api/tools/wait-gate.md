# API: wait-gate

Generated from the code's signatures and docstrings. What it is for and how to use it: [the guide](../../tools/wait-gate.md).

<!-- g:tool -->
## `wait()`

```python
wait(
    reason: str = '',
    resume_with: dict | None = None,
    *,
    name: str = 'wait',
    description: str | None = None,
    hint: dict | None = None,
) -> yait_aichain.tools.general._wait.Wait
```

A pause: the run parks here until `resume(run_id, signal=…)`.

```python
Chain(steps=[(draft, "post"), (wait(reason="Approve the post?"), "approval"), (publish, "done")])
```

## `gate()`

```python
gate(
    tool,
    *,
    reason: str = '',
    resume_with: dict | None = None,
    decision_key: str = 'approved',
    name: str | None = None,
    description: str | None = None,
    hint: dict | None = None,
) -> yait_aichain.tools.general._wait.Gate
```

*tool*, held until its call is approved: the run parks with the call, and
the action runs only on a signal whose *decision_key* is true.

```python
Agent(model, tools=[gate(publish, reason="Publish this?")], store=FileStore("runs/"))
```

---

## The classes behind it

What the factory returns, with the schema a model is shown.

### `wait` — `Wait`

Pause the run until an external signal arrives.

| | |
|---|---|
| Import | `from yait_aichain.tools import Wait` |
| Risk class | `write` |

```python
Wait(
    reason: str = '',
    resume_with: dict | None = None,
    *,
    name: str = 'wait',
    hint: dict | None = None,
)
```

No parameters.

### `Gate`

Gate any Tool behind an external signal.

| | |
|---|---|
| Import | `from yait_aichain.tools import Gate` |
| Risk class | `write` |

```python
Gate(
    tool: Tool,
    *,
    reason: str = '',
    resume_with: dict | None = None,
    decision_key: str = 'approved',
    name: str | None = None,
    hint: dict | None = None,
)
```

The call schema is built from the constructor arguments, so it is whatever the caller declares.
<!-- /g:tool -->
