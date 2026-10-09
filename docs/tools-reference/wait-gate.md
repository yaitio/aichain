<!-- g:tool -->
# Pause for a signal

## `wait` — `Wait`

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

## `Gate`

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

## Made by name

```python
from yait_aichain.tools import gate, wait
from yait_aichain.state import FileStore

Chain(steps=[(draft, "post"), (wait(reason="Approve the post?"), "approval"), (publish, "done")],
      store=FileStore("runs/"))
Agent(model, tools=[gate(refund, reason="Approve this refund?")], store=FileStore("runs/"))
```

`wait(...)` and `gate(tool, ...)` make `Wait` and `Gate`, with `description=`
beside the class's own settings. Until 3.5.0 they lived in `tools._wait`, which
still imports.
