# Skill

A Skill is one call to a model, made reusable: a model, a prompt template and
the shape of the answer you expect. It is the smallest thing in the library
that does work, and everything larger — a Chain, a Pool, an Agent — is built
from Skills or calls models the way a Skill does.

```python
from yait_aichain import Model, Skill

summarise = Skill(Model("claude-haiku-5-5"), prompt="Summarise in one line: {text}")
summarise.run(variables={"text": "…"})          # a str
```

Building a Skill sends nothing. It is a template; only `run()` calls the model,
and the same Skill runs as many times as you like with different variables.

→ Every parameter: [API: Skill](../api/skill.md)

---

## One message format for every provider

A prompt is written once, as messages with a `role` and a list of `parts`:

```python
input = {"messages": [
    {"role": "system", "parts": ["You are a translator."]},
    {"role": "user",   "parts": ["Translate {text} into {language}."]},
]}
```

That is the only format you write. OpenAI's `messages`, Anthropic's split
between `system` and `messages`, Google's `contents` are the model layer's job,
which is why changing `Model("claude-haiku-5-5")` to `Model("gpt-6-luna")`
changes nothing about the prompt. An image goes in the same list, as a part
with a `source` — a URL or base64.

`prompt="…"` is a shortcut for the commonest case: one user message, one text
part. Anything more — a system message, several turns, an image — is `input=`.
The shortcut stays that small on purpose. A shortcut that grows options becomes
a second input format, and two ways to say one thing is how code and
documentation drift apart. Giving both raises, because which one the model
should see would be a guess.

## Placeholders are replaced safely, not formatted

`{name}` is filled from the variables — the Skill's own defaults, overridden by
the ones passed to `run()`. Only a bare `{identifier}` that has a value is
replaced. Everything else passes through untouched:

- an unknown `{topic}` stays in the text as `{topic}`, so a typo shows up in
  the output instead of vanishing into an empty string;
- literal braces — a JSON example in the prompt, LaTeX, code — need no escaping,
  and braces a model writes are never mangled.

This is not Python's `str.format`, deliberately: a prompt is full of braces
that are not placeholders.

## The answer has the shape you declared

`output` says what comes back, and the library both steers the provider and
shapes the return value:

| `output` | `run()` returns |
|---|---|
| nothing, or `{"format": {"type": "text"}}` | a `str` |
| `{"format": {"type": "json"}}` | a `dict` — the provider is asked for JSON, and the reply is parsed |
| `{"format": {"type": "json_schema", "schema": {…}}}` | a `dict` matching your schema — sent to the provider as a constraint, checked again on return |

Use `json` when you want valid JSON and will read it loosely. Use
`json_schema` when code depends on the keys and types: correctness then comes
from the schema rather than from prompt wording. A reply cut off at the token
ceiling raises `TruncatedResponseError`, and one that arrived whole but does
not match raises `InvalidStructuredOutputError` naming where — two different
problems, told apart.

Not every provider enforces a schema. Where one cannot, the schema goes in the
prompt instead and the notice says so; the check on return is the same. Some
models still answer in prose now and then — Qwen 3.6 Flash and 3.7 Flash with
tools beside the schema, about half the time when measured — and that is the
`InvalidStructuredOutputError`, never a wrong value.

Images are not a Skill's job any more: an image model is a tool,
`image(service, model=…)`. → [Images](../tools/media.md)

## Several turns you direct

An `assistant` turn with no `parts` means "generate here, and keep it":

```python
Skill(Model("claude-haiku-5-5"), input={"messages": [
    {"role": "user", "parts": ["Write ten quotes about {topic}."]},
    {"role": "assistant"},
    {"role": "user", "parts": ["Drop the five most clichéd and write five fresh ones."]},
    {"role": "assistant"},
    {"role": "user", "parts": ["Translate the result into {language}."]},
]})
```

Each marker is one model call whose reply joins the context, and the last
turn's reply, in the declared `output` format, is what `run()` returns. This is
reasoning in steps you wrote, not steps the model chose — when the steps are
known, it is cheaper and more predictable than an
[Agent](../agents/overview.md). An `assistant` turn *with* parts is a fixed
turn, a few-shot example, and costs nothing.

## When a call fails

Two mechanisms, along two different axes:

- **`max_retries`** retries the same model after a transient failure — 429,
  5xx, a network error — waiting `retry_delay`, then twice that, then four
  times. The provider was briefly busy.
- **A list of models** — `Skill([Model("gpt-6-luna"), Model("claude-haiku-5-5")], …)`
  — moves to the next one when a model keeps failing. The provider is down.

A request first spends its retries on one model, then falls back. Nothing else
is retried: a 400, a 401 or a 404 will not be fixed by another attempt or
another model, and a reply that does not parse means the prompt or the schema
needs work, so it raises at once instead of burning tokens.

## Streaming

`stream()` yields the answer as it arrives — the same request at the same
price, a second way to spend it. Four things differ from `run()`, each on
purpose:

- **Retries stop at the first piece, and there is no fallback.** Both throw the
  attempt away and start again, which cannot be done once you have shown
  someone half a sentence. Before the first piece nothing has been seen, so a
  transient failure there is still retried.
- **Usage may be `None`.** Most providers report it; where one does not,
  nothing is invented from the text that happened to arrive.
- **JSON is parsed at the end.** Pieces are text; `last_result` holds the
  parsed value once the stream is done, and an answer that does not parse keeps
  its text.
- **A provider that cannot stream still answers**, in one piece, and records a
  `declined` adaptation — one chunk at the end looks like a working stream, so
  it is said rather than left to look like one.

## What a Skill reports

After each call, `skill.last_usage` has the tokens and the cost, and
`skill.last_adaptations` has everything the library changed to fit the
provider — an option dropped, a value translated, a model swapped. A ceiling in
dollars is `max_cost=`. → [Cost and budget](../control/cost-and-budget.md)

## A portable unit

`skill.save("explain.yaml")` writes the template and the model's *name*;
`Skill.load()` rebuilds it and reads the key from the environment. A key is
never written. A saved Skill is how a prompt moves between projects, or from a
notebook into a pipeline, without being copied by hand.

---

▶ [`examples/01_skill.py`](../../examples/01_skill.py) ·
[`examples/02_switch_provider.py`](../../examples/02_switch_provider.py) ·
[`examples/05_multi_turn.py`](../../examples/05_multi_turn.py)
