# Skill

A **Skill** is the smallest unit that does real work: one model + one prompt
template + one declared output. Everything else — Chain, Pool, Agent — composes
Skills.

---

## Quick start

Bind a model to a prompt with `{placeholders}`, then run it with values:

```python
import os
from yait_aichain.models import Model
from yait_aichain.skills import Skill

skill = Skill(
    model = Model("claude-sonnet-4-6", api_key=os.getenv("ANTHROPIC_API_KEY")),
    input = {"messages": [{"role": "user", "parts": ["What is {topic} in one sentence?"]}]},
)

print(skill.run(variables={"topic": "gravity"}))   # -> a string
```

`run()` returns a **string** for text output (a **dict** for JSON or image —
see [Output formats](#output-formats)).

▶ Runnable: [`examples/01_skill.py`](../../examples/01_skill.py) ·
Swap providers: [`examples/02_skill_models.py`](../../examples/02_skill_models.py) ·
Deep dive ↓

---

## Common gotchas

- **A skill is a template, not a result.** Building it makes no network call;
  only `run()` does. The same skill can be run many times with different
  variables.
- **An unknown `{placeholder}` is left in the text verbatim** — it is *not* an
  error and *not* blanked out. If you see a literal `{topic}` in the output, the
  variable name didn't match. (This is deliberate — see
  [Variables & substitution](#variables--substitution).)
- **`parts` can be a bare string** (`["Hello {name}"]`) as shorthand, or the
  full `{"type": "text", "text": "…"}` form. Both work; mix freely.
- **The result type follows `output`**: text → `str`, json/json_schema → `dict`,
  image → `dict` with a base64 payload. Don't `str()` a JSON result.
- **Keys never touch YAML.** `save()` stores the model *name*; the key is
  re-resolved from the environment on `load()`.

---

## Reference

### How a Skill runs

When you call `skill.run(variables=...)`, four things happen, in order:

1. **Substitute.** Every `{placeholder}` in the input template's text parts is
   replaced from the merged variables ([details](#variables--substitution)).
2. **Translate.** The model converts the now-filled universal template into the
   provider's native request shape — OpenAI `messages`, Anthropic `messages`,
   Google `contents`, an image-generation body, etc. This is why the *same*
   `input` works on every provider: you write it once, the model layer adapts
   it. You never write provider-specific payloads.
3. **Send.** The request goes out through the model's client. Transient failures
   are retried and, if you gave a model list, fall back to the next model
   ([Fallback chains](#fallback-chains)).
4. **Parse.** The raw response is turned into the declared `output` type — a
   string, a parsed/validated dict, or an image dict — and token usage is
   recorded on `skill.last_usage` ([Token usage & cost](#token-usage--cost)).

Holding this four-step model in mind explains every parameter below.

### Constructor

```python
Skill(model, input, output=None, variables=None, options=None,
      name=None, description=None, max_retries=0, retry_delay=2.0)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `Model` \| `list[Model]` | — | The model that runs the skill. A **list** is a [fallback chain](#fallback-chains). |
| `input` | `dict` | — | Provider-agnostic message template. See [The input template](#the-input-template). |
| `output` | `dict` \| `None` | `None` | Declares the response format. `None`/`{}` = plain text. See [Output formats](#output-formats). |
| `variables` | `dict` \| `None` | `None` | Default placeholder values, merged with (and overridden by) `run()`'s `variables`. |
| `options` | `dict` \| `None` | `None` | Per-call model setting overrides (e.g. `{"temperature": 0.2}`) layered on top of the model's own configuration for this skill. |
| `name` | `str` \| `None` | `None` | Identifier used in `Chain.history` and YAML persistence. |
| `description` | `str` \| `None` | `None` | Free-text note; surfaced to an Agent when the skill is used as a tool. |
| `max_retries` | `int` | `0` | Retries on the **same** model for transient errors, with exponential backoff. |
| `retry_delay` | `float` | `2.0` | Base backoff seconds (doubled each attempt). |

### The input template

The input is a list of messages; each message has a `role`
(`system` / `user` / `assistant`) and a list of `parts`. A part is either a
plain string (shorthand for a text part) or an explicit
`{"type": "text", "text": "…"}` object.

```python
input = {
    "messages": [
        {"role": "system", "parts": ["You are a translator."]},
        {"role": "user",   "parts": ["Translate {text} into {language}."]},
    ]
}
```

This is the library's **one universal format**. You never write OpenAI's
`messages`, Anthropic's `system`+`messages` split, or Google's `contents` by
hand — the model layer translates this shape into each provider's native
request (step 2 above). Swapping `Model("gpt-4o")` for
`Model("claude-sonnet-4-6")` changes nothing about your `input`.

**Multimodal input.** Image and audio parts carry a `source`, either a URL or
inline base64:

```python
{"role": "user", "parts": [
    "Describe this image.",
    {"type": "image", "source": {"kind": "url", "url": "https://example.com/photo.jpg"}},
    # inline alternative:
    # {"type": "image", "source": {"kind": "base64", "mime": "image/png", "data": "<b64>"}},
]}
```

Whether a given model accepts image input depends on its capabilities — see
[Models](models.md) for the per-provider vision support.

### Variables & substitution

Variables come from two places and are merged, **call-time wins**:

1. `Skill(variables=...)` — defaults baked into the skill.
2. `skill.run(variables=...)` — overrides for this one call.

```python
skill = Skill(model=model, input=..., variables={"language": "English"})
skill.run()                                  # language = English
skill.run(variables={"language": "French"})  # language = French
skill.run(variables={"text": "Hello"})       # text set; language still English
```

Substitution is intentionally **safe and minimal**, not Python's `str.format`:

- Only a bare `{identifier}` whose key exists in the variables is replaced.
- An **unknown placeholder is left untouched** — `{topic}` with no `topic`
  variable stays the literal text `{topic}`. A typo surfaces visibly in the
  output instead of vanishing or raising.
- **Literal braces pass through unchanged**, so your prompt can contain JSON
  examples (`{"role": "user"}`), LaTeX, or code without escaping anything, and
  model-generated braces never get mangled.

That trade — visible-on-typo plus brace-safe — is why missing variables are not
turned into empty strings.

### Output formats

`output` declares what you expect back, and the library both *steers the
provider* and *shapes the return value* accordingly.

| `output` | Returns | What the library does |
|---|---|---|
| `None` / `{}` / `{"format": {"type": "text"}}` | `str` | Plain completion, returned verbatim. |
| `{"format": {"type": "json"}}` | `dict` | Tells the provider to emit JSON, then parses it. Raises `ValueError` if the reply isn't valid JSON. |
| `{"format": {"type": "json_schema", "schema": {…}}}` | `dict` | As above, plus the response is constrained to (and validated against) your JSON Schema. |
| `{"modalities": ["image"], "format": {"type": "image", "size": "1536x1024"}}` | `dict` | Routes to the provider's image endpoint; returns `{"base64", "mime_type", …}`. |

**JSON vs JSON Schema.** Use `json` when you just want "valid JSON back" and
will read it loosely. Use `json_schema` when you need *predictable keys and
types* — the schema is sent to the provider as a constraint and re-checked on
return, so you don't prompt-engineer correctness by hand:

```python
output = {"format": {
    "type": "json_schema", "name": "cities", "strict": True,
    "schema": {
        "type": "object",
        "properties": {"cities": {"type": "array", "items": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "population": {"type": "integer"}},
            "required": ["name", "population"],
        }}},
        "required": ["cities"],
    },
}}
```

**Transparent images.** For `gpt-image-*` / `chatgpt-image-*`, add
`"background": "transparent"` and `"output_format": "png"` to get a real PNG
alpha channel (other image providers bake an opaque background regardless of the
prompt):

```python
output = {"modalities": ["image"], "format": {
    "type": "image", "size": "1536x1024",
    "background": "transparent", "output_format": "png",
}}
```

### `run()`

```python
skill.run(variables=None, max_retries=None, retry_delay=None) -> str | dict
```

`variables` are merged over the constructor's (later wins). `max_retries` /
`retry_delay` override the constructor values for this call only. The return
type is whatever the `output` format implies.

### Token usage & cost

After every `run()`, `skill.last_usage` holds a `Usage` for *that* call. Token
counts are normalised across providers (each provider reports usage
differently), and the cost is computed from the prices in the model registry:

```python
skill.run(variables={"topic": "gravity"})
u = skill.last_usage
print(u.input_tokens, u.output_tokens, u.total_tokens)
print(u.cost)          # USD, derived from the model's registry price
```

`last_usage` reflects the most recent run only; read it right after the call (or
accumulate it yourself across calls).

### Fallback chains

Pass a **list of models** instead of one. On a *transient* failure — HTTP 429
(rate limit), 5xx (server), or a network error — the skill advances to the next
model in the list and tries again. The successful result is identical to a
single-model run; only resilience changes.

```python
skill = Skill(
    model = [Model("gpt-5.4"), Model("claude-sonnet-4-6"), Model("gemini-2.5-flash")],
    input = {"messages": [{"role": "user", "parts": ["Summarise: {text}"]}]},
)
```

`max_retries` and the fallback chain compose along different axes:
`max_retries` retries *within* one model (the same model might just be briefly
overloaded); the chain advances *between* models (this provider is down, try
another). A request first exhausts retries on a model, then falls back.

A **non-transient** error (400 bad request, 401 auth, 404 model-not-found) is
*not* retried or fallen back from — a different model won't fix a malformed
request — so it raises immediately.

### Retries

When `max_retries > 0`, transient errors are retried with exponential backoff:
`retry_delay`, then `retry_delay × 2`, then `× 4`, … (so `2s, 4s, 8s` by
default). Parsing errors (malformed JSON from the model) are **never** retried —
they signal a prompt or schema problem, not a server fault, so retrying would
just burn tokens.

```python
skill = Skill(model=model, input=..., output=..., max_retries=3, retry_delay=2.0)
skill.run(max_retries=5)   # per-call override
```

### Save and load

A skill is the smallest portable unit in the library. `save()` writes the
template and the model *name* to YAML; `load()` re-creates the model and
re-resolves the API key from the environment — keys are **never** written to
disk.

```python
skill.save("skills/explainer.yaml")
loaded = Skill.load("skills/explainer.yaml")
# loaded = Skill.load("skills/explainer.yaml", api_key="sk-...")  # explicit override
```

Saved: `model_name`, `input`, `output`, `variables`, `options`, `name`,
`description`. Not saved: API keys.

### Using a Skill in a Chain

A Skill is already a valid `Chain` step. In a chain it receives the accumulated
variable dict, and its output is stored back under a named key for later steps:

```python
from yait_aichain.chain import Chain

chain = Chain(steps=[(summariser, "summary"), (translator, "final")])
```

See [Chain](chain.md) for the full step syntax and variable flow.

---

## See also

- [Model](models.md) — the providers and models a Skill can bind to.
- [Chain](chain.md) — wire Skills in sequence with automatic variable flow.
- [Tools](tools.md) — add real-world actions (search, convert, embeddings).
