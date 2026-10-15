# API: Skill

Generated from the code's signatures and docstrings. What it is for and how to use it: [the guide](../pipelines/skills.md).

<!-- g:api -->
[`Skill`](#skill)

---

## `Skill`

```python
Skill(
    model: Model | list[Model],
    input: dict | None = None,
    *,
    prompt: str | None = None,
    output: dict | None = None,
    variables: dict | None = None,
    name: str | None = None,
    description: str | None = None,
    max_retries: int = 0,
    retry_delay: float = 2.0,
    hooks: list | None = None,
    max_cost: float | Budget | None = None,
    _tools: list | None = None,
    _image: bool = False,
    **refused,
)
```

A reusable, model-bound task unit.

**Parameters**

- `model` (*Model | list[Model]*) — The model that runs the skill; a list is a fallback chain.
- `input` (*dict*, default `None`) — Universal message template, `{"messages": [{"role", "parts"}]}`; text
  parts may hold `{placeholder}` tokens and may be plain strings. An
  `assistant` turn with no `parts` marks one model call, for
  several turns you direct.
- `output` (*dict | None, optional*, default `None`) — Expected format; default `{"modalities": ["text"], "format": {"type":
  "text"}}`, else e.g. `{"format": {"type": "json"}}` or
  `"json_schema"` with a `schema`. Image output raises: use `image()`.
- `variables` (*dict | None, optional*, default `None`) — Default variable values; `run(variables=...)` overrides them.
- `**refused` — `options=` is refused; a model's options are `Model(name, options={…})`.
- `name` (*str | None, optional*, default `None`) — Human-readable label for the skill.
- `description` (*str | None, optional*, default `None`) — Short description of what the skill does.
- `prompt` (*str | None, optional*, default `None`) — Shortcut for one user message; give `prompt` or `input`, not both.
- `max_retries` (*int, optional*, default `0`) — Retries after a transient failure (429, 5xx, network); `0` is one attempt.
- `retry_delay` (*float, optional*, default `2.0`) — Seconds before the first retry; each later one waits twice as long.
- `hooks` (*list | None, optional*, default `None`) — Callables given an `Event` for each model call.
- `max_cost` (*float | Budget | None, optional*, default `None`) — A ceiling in dollars; a call that would begin past it raises
  `BudgetExceeded`.

**Examples**

```python
skill = Skill(Model("gpt-6-luna"), prompt="What is {topic}?")
skill.run(variables={"topic": "gravity"})
```

### `Skill.run()`

```python
Skill.run(
    variables: dict | None = None,
    max_retries: int | None = None,
    retry_delay: float | None = None,
) -> str | dict
```

Execute the skill and return the model's response.

**Parameters**

- `variables` (*dict | None, optional*, default `None`) — Variables for this call, merged over `self.variables`.
- `max_retries` (*int | None, optional*, default `None`) — Retries for this call only, over the constructor's; `0` is one attempt.
- `retry_delay` (*float | None, optional*, default `None`) — First retry delay for this call only; it doubles on each retry.

**Returns**

- `str | dict` — Text, or a dict when the format is `"json"` / `"json_schema"`.

**Raises**

- `APIError` — Non-transient, or transient (429, 5xx) after every retry and model.
- `ValueError` — The response cannot be parsed; never retried.

### `Skill.stream()`

```python
Skill.stream(variables: dict | None = None)
```

Run the skill and yield the answer as it arrives.

Unlike `run()`: retries stop at the first piece and there is no
fallback chain; `last_result` holds the whole (parsed) answer at the
end; `last_usage` is None when the provider reported nothing; a
provider that cannot stream yields one piece and records a `declined`.
See `docs/pipelines/skills.md`.

**Parameters**

- `variables` (*dict | None, optional*, default `None`) — Variables for this call, merged over the Skill's own.

### `Skill.save()`

```python
Skill.save(path: str)
```

Serialise this skill to a YAML file at *path*.

Stores the model name, input, output, variables, name and description;
API keys are **never** written (`load` reads them from the
environment).

**Parameters**

- `path` (*str*) — Destination file path; parent directories are created.

### `Skill.to_dict()`

```python
Skill.to_dict() -> dict
```

What `save` writes, and a Chain's saved step holds.

The model's own options, the fallback chain, the retries and a ceiling
given as a number all come back: a saved Skill at temperature 0 came
back at 1.0, with no fallback and no budget. A shared `Budget` cannot
be written down — it is an object others hold — and saving one says so.

### `Skill.from_dict()`

```python
Skill.from_dict(data: dict, api_key: str | None = None) -> Skill
```

A Skill from what `to_dict` wrote.

**Parameters**

- `data` (*dict*) — What `to_dict` returned.
- `api_key` (*str | None, optional*, default `None`) — The key for the model; its environment variable when omitted.

### `Skill.load()`

```python
Skill.load(path: str, api_key: str | None = None) -> Skill
```

Load a skill from a YAML file previously created by `save`.

**Parameters**

- `path` (*str*) — Path to the YAML file.
- `api_key` (*str | None, optional*, default `None`) — Key for the model; read from the provider's environment variable
  when omitted.

**Returns**

- `Skill` — A skill with a ready-to-use model.

**Raises**

- `FileNotFoundError` — If *path* does not exist.
- `ImportError` — If PyYAML is not installed.
- `ValueError` — If no API key is available for the model's provider.
<!-- /g:api -->
