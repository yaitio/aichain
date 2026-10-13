# API: Model

Generated from the code's signatures and docstrings. What it is for and how to use it: [the guide](../models/models.md).

<!-- g:api -->
[`Model`](#model) · [`Usage`](#usage) · [`Adaptation`](#adaptation) · [`UnsupportedOption`](#unsupportedoption) · [`portable_schema`](#portable_schema) · [`check_structure`](#check_structure) · [`SchemaNotExpressible`](#schemanotexpressible) · [`ToolCall`](#toolcall) · [`ToolCallRequest`](#toolcallrequest) · [`tool_result_turn`](#tool_result_turn) · [`dangling_calls`](#dangling_calls)

---

## `Model`

```python
Model(
    name: str,
    options: dict | None = None,
    client_options: dict | None = None,
    api_key: str | None = None,
    on_unsupported: str = 'warn',
)
```

A configured model: provider resolved from the name, settings from data,
format + transport delegated to the matching family client.

`Model(name, options, client_options, api_key)` resolves *name* to a
provider (via the `providers/` data), merges that provider's default
generation parameters, and attaches a ready-to-use family client.  There
is a single `Model` class — no per-provider subclasses; the provider is
available as `model._provider`.

**Parameters**

- `name` (*str*) — Model identifier, e.g. `"gpt-4o"`, `"claude-sonnet-4-5"`,
  `"gemini-2.0-flash"`, `"grok-3"`, `"sonar-pro"`.
- `options` (*dict | None, optional*, default `None`) — Override any subset of the model's default generation parameters.
  Supported keys (all optional):

  ================  ========  =======================================
  Key               Type      Description
  ================  ========  =======================================
  `temperature`   float     Sampling temperature.
  `max_tokens`    int       Maximum output tokens.
  `top_p`         float     Nucleus-sampling probability mass.
  `top_k`         int       Top-K sampling (provider-dependent).
  `cache_control` bool|int  Prompt caching. True marks the
                    |str      second-to-last message; an int names
                              the message the stable prefix ends at;
                              `"last"` marks the newest block, for
                              agent loops whose history ends in a
                              tool turn.
  `cache_ttl`     str       How long the mark lives: `"5m"`
                              (default) or `"1h"`.
  `reasoning`     str|None  Universal reasoning depth (see below).
  ================  ========  =======================================

  **reasoning** takes `None`, `"none"`, `"low"`, `"medium"` or
  `"high"` — the levels a provider's `reasoning_map` declares, and a
  level it lacks is refused here, naming the ones it has. Each family
  translates the level to its own wire: a token budget, an effort word,
  a thinking switch, or a different model. What each provider does with
  it, and with every other option, is the generated table in
  `docs/reference/parameters.md`; what this request did is in
  `last_adaptations`.
- `client_options` (*dict | None, optional*, default `None`) — Override settings for the underlying HTTP client.
  Supported keys (all optional):

  ===========  ========================  ==========================
  Key          Type                      Description
  ===========  ========================  ==========================
  `url`      str                       Base URL override.
  `timeout`  `urllib3.Timeout`       Custom connect/read timeout.
  `retries`  `urllib3.Retry`         Custom retry policy.
  `proxy`    dict                      Proxy config (see BaseClient).
  ===========  ========================  ==========================
- `api_key` (*str | None, optional*, default `None`) — Provider API key.  When omitted it is read from the provider's
  environment variable (e.g. `OPENAI_API_KEY`), named in the provider
  data.  Raises `ValueError` if neither is found.
- `on_unsupported` (*str, optional*, default `'warn'`) — What happens when a provider cannot do an option it was asked:
  `"warn"` (the default) adapts or drops it and says so in
  `last_adaptations`; `"requirements"` raises for the losses that
  leave a result looking right and silently wrong — `seed`, `size`,
  `aspect_ratio`, `background`, `fidelity`, `output_format`,
  `compression`; `"raise"` raises for any.

**Attributes**

- `name` (*str*)
- `temperature` (*float*)
- `max_tokens` (*int*)
- `top_p` (*float | None*)
- `top_k` (*int | None*)
- `cache_control` (*bool | int | "last"*)
- `reasoning` (*str | None  (None | "low" | "medium" | "high")*)
- `client` (*family client (ready to use)*)

**Examples**

>>> m = Model("gpt-4o")
>>> m._provider
'openai'

>>> m = Model("claude-sonnet-4-5", options={"temperature": 0.5, "reasoning": "high"})
>>> m.temperature
0.5

>>> m = Model("gemini-2.0-flash", client_options={"proxy": {"url": "http://corp-proxy:3128"}})
>>> m.client._base_url
'https://generativelanguage.googleapis.com/v1beta'

### `Model.effective_options`

What sampling this model actually runs with, asked for or not.

Provider defaults are not a shared baseline and never were:
`temperature` starts at 0.0 on DeepSeek, 0.2 on Perplexity, 0.7 on
Qwen and 1.0 on OpenAI, Google, Kimi and xAI; `max_tokens` spans
2048 to 32768; `top_k` has a value on Google alone. Two models
compared with no options set are therefore not one variable apart,
which is exactly the mistake this library tells its users not to make.

Converging them was considered and rejected: a library-wide default
overrides a number each provider chose for its own model, and it
would change every existing caller's output to fix a comparison only
some of them are making. Publishing the fact is the honest half —
`defaults` in the provider data, this property at runtime, and a
table in `docs/reference/parameters.md`. A measurement run records
this beside its results and its arms are comparable or visibly not.

### `Model.to_request()`

```python
Model.to_request(
    messages: list,
    output: dict,
    tools: list | None = None,
) -> tuple[str, dict]
```

Translate substituted universal *messages* + *output* spec into the
provider's native `(path, body)` pair, by delegating to the family
client that owns this provider's wire format.

*tools* is a list of function-calling schemas (what `Tool.schema()`
returns) declared to the provider natively. Families that have not
implemented the tool wire format refuse loudly here — silently sending
a request without the declared tools would produce a model that simply
never calls anything, with no error anywhere.

**Parameters**

- `messages` (*list*) — Universal messages, placeholders already filled: `{"role", "parts"}`
  with typed parts.
- `output` (*dict*) — The output spec: `{"modalities": [...], "format": {"type": …}}`.
- `tools` (*list | None, optional*, default `None`) — Function-calling schemas, as `Tool.schema()` returns them.

### `Model.build()`

```python
Model.build(messages: list, output: dict, tools: list | None = None)
```

`to_request`, and what building changed: `(path, body, adaptations)`.

The adaptations are this request's, returned with it. One Model is
shared — by a Pool's threads, by tenants — so a fact about one call
left on the object (`last_adaptations`, kept as a mirror) is read by
whichever call looks next.

**Parameters**

- `messages` (*list*) — Universal messages, placeholders already filled.
- `output` (*dict*) — The output spec.
- `tools` (*list | None, optional*, default `None`) — Function-calling schemas, as `Tool.schema()` returns them.

### `Model.stream()`

```python
Model.stream(messages: list, output: dict, tools: list | None = None)
```

Yield the answer in pieces as the provider produces it.

Not every provider can. An image endpoint has nothing to deliver
progressively, and two of OpenAI's own paths speak a different event
vocabulary that is not wired yet. The question that decides the
design is what to do then, and it is the same question 2.3.0 answered
for options: **the request still goes, and the caller is told.** So a
provider that cannot stream returns the whole answer as one piece and
records a `declined` — because a caller who asked to stream and got
one chunk at the end has something that looks like it worked.
`on_unsupported="raise"` raises on it; `"requirements"` does not,
since the answer itself is whole.

When the stream ends, the generator returns a `StreamOutcome`
— the whole result, the usage the provider reported, and what was
adapted — for `yield from` to receive. `last_stream_usage`,
`last_stream_result` and `last_adaptations` mirror it on the
model, for a single caller; the library reads the outcome, because a
shared model's attributes belong to whichever stream wrote last.

**Parameters**

- `messages` (*list*) — Universal messages, placeholders already filled: `{"role", "parts"}`
  with typed parts.
- `output` (*dict*) — The output spec: `{"modalities": [...], "format": {"type": …}}`.
- `tools` (*list | None, optional*, default `None`) — Function-calling schemas, as `Tool.schema()` returns them.

### `Model.from_response()`

```python
Model.from_response(
    response: dict,
    output: dict,
    *,
    tools_sent: bool | None = None,
) -> str | dict
```

Extract the clean result (str for text, dict for json / image) from a
raw provider response, by delegating to the family client.

**Parameters**

- `response` (*dict*) — The provider's raw response body.
- `output` (*dict*) — The output spec the request was built with.
- `tools_sent` (*bool | None, optional*, default `None`) — Whether that request offered tools — only then can a reply be a
  request to act. Passed by the caller that built the request; the
  model's own record of its last request is the fallback.

---

## `Usage`

```python
Usage(
    input_tokens: int = 0,
    output_tokens: int = 0,
    total_tokens: int = 0,
    cost: float | None = None,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
    image_input_tokens: int = 0,
    unpriced_calls: int = 0,
)
```

Token counts for one or more LLM calls.

**Attributes**

- `input_tokens` (*prompt / input tokens billed.*, default `0`)
- `output_tokens` (*completion / output tokens billed.*, default `0`)
- `total_tokens` (*provider-reported total, or input+output when absent.*, default `0`)
- `cost` (*estimated cost in USD (filled in 1.2-C; `None` until then).*, default `None`)
- `cache_write_tokens` (*int*, default `0`) — Prompt prefix stored for reuse this call, billed above fresh input.
- `cache_read_tokens` (*int*, default `0`) — Prompt prefix reused from an earlier call, billed well below it.
- `image_input_tokens` (*int*, default `0`) — Images sent in, kept apart because they are billed at their own rate.
- `unpriced_calls` (*int*, default `0`) — Calls the registry had no price for; `cost` leaves them out.

---

## `Adaptation`

```python
Adaptation(
    kind: str,
    option: str,
    asked: object = None,
    sent: object = None,
    why: str = '',
    model: str = '',
)
```

One change the library made to a request, or refused to make.

**Attributes**

- `kind` (*one of the five above.*)
- `option` (*the universal option's name, as the caller wrote it.*)
- `asked` (*the value the caller set.*, default `None`)
- `sent` (*what actually went out — a field name, a shape, a model name,*, default `None`) — or `None` when nothing did.
- `why` (*one sentence a reader can act on.*, default `''`)
- `model` (*the model the request was built for.*, default `''`)

---

## `UnsupportedOption`

```python
UnsupportedOption(message: str, adaptations: list)
```

An option the caller set could not be honoured, and they asked to be
stopped rather than warned. Carries the adaptations that caused it.

**Parameters**

- `message` (*str*) — What could not be honoured.
- `adaptations` (*list*) — The `Adaptation` records behind it, also on `last_adaptations`.

---

## `portable_schema()`

```python
portable_schema(schema: object, target: str) -> object
```

Return *schema* in the form *target* accepts.

`target` is a provider family: `"openai"` (strict mode) or
`"google"`. The original is never modified.

Neither direction is lossless, and one shape has no strict equivalent at
all — a map with arbitrary keys. For `"openai"` that raises rather than
being mangled, because the provider's own message points at the wire
format instead of at the schema the caller wrote.

**Parameters**

- `schema` (*dict*) — A JSON Schema.
- `target` (*str*) — The provider family it must be sent to, e.g. `"google"`.

---

## `check_structure()`

```python
check_structure(value: object, schema: object, path: str = '$') -> list[str]
```

Compare *value* against *schema*, returning one message per violation.

An empty list means nothing was found — not that the value is valid under
the full specification. Checked: declared types, required properties,
unexpected properties when `additionalProperties` is false, enum
membership, and array element types. Recurses through `properties` and
`items`.

**Parameters**

- `value` (*any*) — What a model returned.
- `schema` (*dict*) — The JSON Schema it was asked for.
- `path` (*str, optional*, default `'$'`) — Where *value* sits, for the messages; `"$"` at the top.

---

## `SchemaNotExpressible`

A schema that strict mode cannot represent, named at the point of use.

Raised instead of letting the provider answer, because the provider's
message points at the wire format rather than at the schema the caller
wrote.

---

## `ToolCall`

```python
ToolCall(id: str, name: str, arguments: dict = <factory>)
```

One requested call: which tool, with which arguments.

**Attributes**

- `id` (*str*) — The provider's id for this call; a tool result answers it by this id.
- `name` (*str*) — The tool's name.
- `arguments` (*dict*, default `<factory>`) — The arguments, parsed from the provider's JSON.

---

## `ToolCallRequest`

```python
ToolCallRequest(calls: tuple[ToolCall, ...], text: str = '', opaque: dict = <factory>)
```

A model reply that asks for tool calls.

Returned by `from_response` (and, from phase 2 on, by `Skill.run`)
*instead of* a string when the model chose to act. `text` carries any
prose the model sent alongside — some providers emit both.

Providers may request **several** calls in one turn. Honouring all of them
is deliberate: it is free parallelism, and executing only the first would
silently narrow the channel this feature exists to widen.

**Attributes**

- `calls` (*tuple[ToolCall, ...]*) — Every call the model asked for in this turn, in order.
- `text` (*str*, default `''`) — Prose the model sent beside them, if any.
- `opaque` (*dict*, default `<factory>`) — What a provider needs back verbatim on the next turn, keyed by the
  provider family — Anthropic's thinking blocks, which must open the
  assistant turn before a tool result when extended thinking is on.
  Not for reading; carried so the history stays one the provider takes.

### `ToolCallRequest.from_turn()`

```python
ToolCallRequest.from_turn(turn: dict) -> ToolCallRequest
```

The reply an assistant turn holds — the inverse of `as_turn`, for a
driver that kept the conversation as JSON between invocations and has
to act on its last turn.

**Parameters**

- `turn` (*dict*) — An assistant turn with `tool_calls`, as `as_turn` wrote it.

### `ToolCallRequest.as_turn()`

```python
ToolCallRequest.as_turn() -> dict
```

This reply as a universal assistant turn, for appending.

---

## `tool_result_turn()`

```python
tool_result_turn(call_id: str, result) -> dict
```

One call's result as a universal `tool` turn.

A result is normally text, and anything that is not a string is
serialised. The exception is media: a tool that returns a media part — or
a list mixing media parts and strings — hands those through as parts
instead of describing them as JSON. Without this an agent that renders
something cannot look at what it rendered, and a vision loop has to be
lifted out of the agent into a separate Skill.

Providers disagree about whether a tool result may carry an image, so the
parts are carried here and each family decides on the wire: Anthropic puts
them inside the tool result, the others send them as a following user
message.

**Parameters**

- `call_id` (*str*) — The `ToolCall.id` this result answers.
- `result` (*any*) — What the tool returned; a non-string is rendered as text.

---

## `dangling_calls()`

```python
dangling_calls(messages: list) -> list
```

Ids of tool calls in *messages* that no later turn answers.

A conversation in which the model asked for a tool and never heard back
is malformed, and providers disagree about it in the worst possible way:
OpenAI's chat completions, the Responses API and Anthropic all reject the
request; Google keys results by name and is looser; a self-hosted
OpenAI-compatible server — vLLM, llama.cpp, mlx — validates nothing at
all and templates whatever it was handed, so the model meets its own
unanswered call and improvises. It repeats it, apologises, or invents the
result, differently each time.

That is run-to-run variance manufactured by us, and it is worst where it
is least visible: the condition that produces a dangling call — a tool
that raised, a result lost across an invocation boundary — is itself
intermittent, so the same input behaves differently on different trials.

`run()` cannot produce one: it appends a result turn for every call in
a reply, failed ones included. The externally driven seam can, because
`step()` hands the obligation to the caller and nothing checked that
they met it — and that seam is precisely what a benchmark harness and a
serverless driver use, which is where a result crosses a process boundary
and can be lost.

**Parameters**

- `messages` (*list*) — A conversation, as `Agent.step()` takes it.
<!-- /g:api -->
