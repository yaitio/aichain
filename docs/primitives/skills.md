# Skill

A **Skill** is one reusable task: one `Model` + one prompt template + one output format.

```python
skill = Skill(model=..., input={...}, output={...})
result = skill.run(variables={...})
```

It's the smallest unit that actually does work. Everything else in the library (Chain, Agent) composes Skills.

---

## Anatomy

```python
from models import Model
from skills import Skill

skill = Skill(
    model = Model("gpt-4o"),
    input = {
        "messages": [
            {"role": "system", "parts": [{"type": "text", "text": "Be concise."}]},
            {"role": "user",   "parts": [{"type": "text", "text": "What is {topic}?"}]},
        ]
    },
    output    = {"modalities": ["text"], "format": {"type": "text"}},
    variables = {"topic": "gravity"},
    name      = "explainer",
)
```

Three required pieces:

| Parameter | Purpose |
|---|---|
| `model` | The `Model` instance that will execute the call. |
| `input` | Universal message template with `{placeholder}` tokens. |
| `output` | Declares the expected response format. |

Optional:

| Parameter | Purpose |
|---|---|
| `variables` | Default variable values. Call-time values override them. |
| `name` | Human-readable label. Used by `Chain` in step identifiers and in `repr`. |
| `description` | Free-text description. |
| `max_retries` | Automatic retry count on transient HTTP errors (default `0`). |
| `retry_delay` | Base back-off seconds between retries (default `2.0`, doubled each attempt). |

---

## The input template

The template is a list of messages; each message has a `role` and a list of `parts`.

```python
{
    "messages": [
        {"role": "system", "parts": [{"type": "text", "text": "You are a translator."}]},
        {"role": "user",   "parts": [{"type": "text", "text": "Translate {text} into {language}."}]},
    ]
}
```

At `run()` time, every `{placeholder}` in every text part is replaced with the matching variable. Missing variables are replaced with an empty string — not an error.

Each provider converts this universal shape into its native format (OpenAI `messages`, Anthropic `messages`, Google `contents`, etc.). You never touch those differences.

### Multimodal input

Image parts use the same structure:

```python
{"role": "user", "parts": [
    {"type": "text",  "text": "Describe this image."},
    {"type": "image", "image": {"url": "https://example.com/photo.jpg"}},
]}
```

See [Models](models.md) for per-provider support tables.

---

## Output formats

Three formats are supported; `skill.run()` returns the right Python type automatically.

### Text

```python
output = {"modalities": ["text"], "format": {"type": "text"}}
```

`skill.run()` returns a `str`.

### JSON (free-form)

```python
output = {"modalities": ["text"], "format": {"type": "json"}}
```

The Skill tells the provider to emit JSON and parses the response. `skill.run()` returns a `dict` (or `list`).

```python
skill = Skill(
    model  = Model("claude-sonnet-4-6"),
    input  = {"messages": [
        {"role": "user", "parts": [{"type": "text",
            "text": "List 3 facts about Mars as a JSON array."}]},
    ]},
    output = {"modalities": ["text"], "format": {"type": "json"}},
)
data = skill.run()   # → list[dict] or dict
```

### JSON Schema (validated)

```python
output = {"modalities": ["text"], "format": {
    "type":   "json_schema",
    "name":   "cities",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "cities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name":       {"type": "string"},
                        "population": {"type": "integer"},
                    },
                    "required": ["name", "population"],
                },
            },
        },
        "required": ["cities"],
    },
}}
```

The provider is instructed to conform to the schema; the response is validated on the way back. `skill.run()` returns a `dict` matching the schema shape.

Useful when you need predictable keys and types without prompt-engineering JSON correctness.

---

## Variables

Variables are merged in this order (later wins):

1. `Skill(variables=...)` — defaults set at construction.
2. `skill.run(variables=...)` — per-call overrides.

```python
skill = Skill(model=model, input=..., output=..., variables={"language": "English"})
skill.run()                                 # language = English
skill.run(variables={"language": "French"}) # language = French
skill.run(variables={"text": "Hello"})      # text set, language stays English
```

Missing variables are silently replaced with an empty string at substitution time.

---

## Retries

Transient server errors (HTTP 429, 500, 502, 503, 504) are retried when `max_retries > 0`:

```python
skill = Skill(model=model, input=..., output=...,
              max_retries=3, retry_delay=2.0)

skill.run()
```

Back-off is exponential: `2s`, then `4s`, then `8s`. Non-transient errors (401, 400, 404, …) are raised immediately — retries cannot fix a bad request.

Per-call overrides:

```python
skill.run(max_retries=5, retry_delay=1.0)
```

Parsing errors (`ValueError` from malformed JSON) are **never** retried — they indicate a prompt or schema issue, not a server fault.

---

## Save and load

A Skill is the smallest portable unit in the library. Save to YAML, reload anywhere:

```python
skill.save("skills/explainer.yaml")

from skills import Skill
skill = Skill.load("skills/explainer.yaml")
```

What gets saved:

- `model_name` (not the model instance)
- `input`, `output`, `variables`, `options`
- `name`, `description`

What does **not** get saved: API keys. They're resolved from environment variables when the file loads.

```yaml
# skills/explainer.yaml
model_name: gpt-4o
name: explainer
input:
  messages:
    - role: system
      parts:
        - type: text
          text: Be concise.
    - role: user
      parts:
        - type: text
          text: What is {topic}?
output:
  modalities: [text]
  format: {type: text}
variables:
  topic: gravity
```

You can override the API key at load time:

```python
Skill.load("skills/explainer.yaml", api_key="sk-...")
```

---

## Using a Skill in a Chain

A Skill is already a valid `Chain` step. It receives the full accumulated variable dict and its output is stored under a named key:

```python
from chain import Chain

chain = Chain(steps=[
    (summariser, "summary"),
    (translator, "final"),
])
```

See [Chain](chain.md) for full step syntax.

---

## Bare-minimum skill

```python
from models import Model
from skills import Skill

skill = Skill(
    model  = Model("claude-haiku-4-5-20251001"),
    input  = {"messages": [
        {"role": "user", "parts": [{"type": "text", "text": "Say hi."}]}
    ]},
    output = {"modalities": ["text"], "format": {"type": "text"}},
)

print(skill.run())
```

That's the entire runtime surface. Everything else is optional.

---

## See also

- **Configure the model** → [Models](models.md)
- **Compose Skills into a pipeline** → [Chain](chain.md)
- **Add real-world actions** → [Tools](tools.md)
