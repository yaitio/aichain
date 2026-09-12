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
