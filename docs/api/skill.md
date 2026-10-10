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
    options: dict | None = None,
    name: str | None = None,
    description: str | None = None,
    max_retries: int = 0,
    retry_delay: float = 2.0,
    hooks: list | None = None,
    max_cost: float | object | None = None,
    _tools: list | None = None,
    _image: bool = False,
)
```

A reusable, model-bound task unit.

**Parameters**

- `model` (*Model subclass instance*) — The language model that will execute this skill.
- `input` (*dict*, default `None`) — Universal message template.  Text parts may contain
  `{placeholder}` tokens that are filled in at run time.

  Shorthand — `"type": "text"` is the default; plain strings work too:

  ```python
  {
    "messages": [
      {"role": "system", "parts": ["Be concise."]},
      {"role": "user",   "parts": [{"text": "Explain {topic}."}]},
    ]
  }
  ```

  Full explicit form (required only for non-text parts such as images):

  ```python
  {
    "messages": [
      {"role": "system", "parts": [{"type": "text", "text": "Be concise."}]},
      {"role": "user",   "parts": [{"type": "text", "text": "Explain {topic}."}]},
    ]
  }
  ```

  **Multi-turn (directed reasoning).** Several `user` turns
  separated by a "generate here" marker — an `assistant` turn with no
  `parts` — run as a sequence: each marker is one model call whose reply
  is appended to the running context (an `assistant` turn *with* parts
  is a fixed/seed turn, not a call). `run()` returns the last reply;
  `history` holds each turn. No markers → one call.
- `output` (*dict | None, optional*, default `None`) — Declares the expected output format.  Omit entirely (or pass `None`
  / `{}`) for the default plain-text output.

  Defaults to `{"modalities": ["text"], "format": {"type": "text"}}`
  when omitted.  Only specify this when deviating from the default:

  ```python
  # plain text — these are all equivalent:
  output=None
  output={}
  output={"modalities": ["text"], "format": {"type": "text"}}

  # JSON object
  output={"format": {"type": "json"}}

  # validated JSON schema
  output={"format": {"type": "json_schema", "schema": { ... }}}
  ```

  A Skill makes text: `output={"modalities": ["image"]}` raises
  `ValueError`. Images are a tool — `image(service, model=…)`.
- `variables` (*dict | None, optional*, default `None`) — Default variable values.  Can be overridden or extended at call
  time via `run(variables={...})`.  Merged at run time; call-time
  values take precedence.
- `options` (*dict | None, optional*, default `None`) — Reserved for future use (currently stored but not applied).
- `name` (*str | None, optional*, default `None`) — Human-readable label for the skill.
- `description` (*str | None, optional*, default `None`) — Short description of what the skill does.
- `prompt` (*str | None, optional*, default `None`) — A shortcut for one user message: `Skill(model, prompt="Summarise:
  {text}")`. Give `prompt` or `input`, not both.
- `max_retries` (*int, optional*, default `0`) — Retries after a transient failure (429, 5xx, a network error), on top
  of the transport's own. `0`, the default, is one attempt.
- `retry_delay` (*float, optional*, default `2.0`) — Seconds before the first retry; each later one waits twice as long.
- `hooks` (*list | None, optional*, default `None`) — Callables given an `Event` for each model call.
- `max_cost` (*float | Budget | None, optional*, default `None`) — A ceiling in dollars; a call that would begin past it raises
  `BudgetExceeded`.

**Examples**

Text output (minimal — `output` omitted, shorthand parts):

```python
from yait_aichain.models import Model
from yait_aichain.skills import Skill

skill = Skill(
    model=Model("gpt-6-luna"),
    input={
        "messages": [
            {"role": "system", "parts": ["Be concise."]},
            {"role": "user",   "parts": [{"text": "What is {topic}?"}]},
        ]
    },
    variables={"topic": "gravity"},
    name="explainer",
)
result = skill.run()                                  # uses default variable
result = skill.run(variables={"topic": "relativity"}) # override
```

JSON output:

```python
skill = Skill(
    model=Model("claude-haiku-5-5"),
    input={
        "messages": [
            {"role": "user", "parts": [{"text": "List 3 facts about {topic} as JSON."}]},
        ]
    },
    output={"format": {"type": "json"}},
    variables={"topic": "Mars"},
)
data = skill.run()   # returns dict
```

Save and load:

```python
skill.save("skills/explainer.yaml")
loaded = Skill.load("skills/explainer.yaml")   # model re-created from YAML
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

- `variables` (*dict | None, optional*, default `None`) — Variables to substitute in this invocation.  Merged with the
  instance-level `self.variables`; call-time values win.
- `max_retries` (*int | None, optional*, default `None`) — Application-level retry count for this call only.  Overrides the
  constructor value when provided.  `0` means a single attempt
  with no retries (the default).
- `retry_delay` (*float | None, optional*, default `None`) — Base sleep in seconds between retries for this call only.
  Overrides the constructor value when provided.  Each subsequent
  attempt doubles the delay (exponential back-off).

**Returns**

- `str` — When `output["format"]["type"] == "text"`.
- `dict` — When `output["format"]["type"]` is `"json"` or
  `"json_schema"`.

**Raises**

- `clients._base.APIError` — On a non-transient HTTP error, or after all retry attempts are
  exhausted on a transient error (status in
  `{429, 500, 502, 503, 504}`).  With a fallback chain, the last
  model's transient error is raised only after every model failed.
- `ValueError` — If the provider response cannot be parsed (never retried —
  this indicates a prompt or schema error, not a transient fault).

### `Skill.stream()`

```python
Skill.stream(variables: dict | None = None)
```

Run the skill and yield the answer as it arrives.

`run()` is untouched — this is a second way to spend the same
request, not a replacement — and the differences from it are real
rather than incidental, so they are stated instead of discovered:

* **Retries stop at the first piece; there is no fallback chain.**
  Both work by throwing the attempt away and starting again, which is
  impossible once the caller has seen something. *Before the first
  piece* is a state the caller can see — nothing has been yielded —
  so a transient failure (429, 5xx, network) retries
  while the attempt is still discardable, exactly as `run()` does,
  and is raised once it is not. The fallback chain does stay out: a
  second model is a different answer, not a retry of this one, and
  swapping models mid-sentence is not something to do quietly.
* **The result is assembled as well as yielded.** `last_result`
  holds the whole text at the end, parsed when the output format
  asks for JSON, so a caller does not have to choose between showing
  progress and having the value.
* **Usage is whatever the provider reported**, on `last_usage`, and
  None when it reported nothing.

A provider that cannot stream yields the whole answer as one piece
and says so through the adaptation channel — see `Model.stream`.

**Parameters**

- `variables` (*dict | None, optional*, default `None`) — Variables for this call, merged over the Skill's own.

### `Skill.save()`

```python
Skill.save(path: str)
```

Serialise this skill to a YAML file at *path*.

The file stores everything needed to recreate the skill — the model
name, input template, output spec, variables, options, name, and
description.  API keys are **never** written; `load` resolves
them from environment variables at load time.

**Parameters**

- `path` (*str*) — Destination file path.  Parent directories are created
  automatically if they do not exist.

**Examples**

```python
skill.save("skills/translation.yaml")
```

### `Skill.load()`

```python
Skill.load(path: str, api_key: str | None = None) -> Skill
```

Load a skill from a YAML file previously created by `save`.

The model is constructed automatically from the `model_name` stored
in the file.  The API key is resolved from the matching environment
variable (e.g. `OPENAI_API_KEY`) unless *api_key* is supplied
explicitly.

**Parameters**

- `path` (*str*) — Path to the YAML file.
- `api_key` (*str | None, optional*, default `None`) — Override the API key used to initialise the model.  When omitted,
  the key is read from the provider's environment variable.

**Returns**

- `Skill` — A fully initialised `Skill` instance with a ready-to-use
  model attached.

**Raises**

- `FileNotFoundError` — If *path* does not exist.
- `ImportError` — If PyYAML is not installed.
- `ValueError` — If no API key is available for the model's provider.

**Examples**

```python
from yait_aichain.skills import Skill

skill  = Skill.load("skills/translation.yaml")
result = skill.run(variables={"text": "Hello, world!"})
```
<!-- /g:api -->
