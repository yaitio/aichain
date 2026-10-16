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
format and transport delegated to the matching family client.

One `Model` class serves every provider; the provider is
`model._provider`.

**Parameters**

- `name` (*str*) — Model identifier, e.g. `"gpt-4o"`, `"claude-sonnet-4-5"`.
- `options` (*dict | None, optional*, default `None`) — Overrides of the defaults: `temperature`, `max_tokens`, `top_p`,
  `top_k`, `reasoning` (`"none"`/`"low"`/`"medium"`/`"high"`),
  `cache_control` (True, a message index, or `"last"` for an agent
  loop), `cache_ttl` (`"5m"`/`"1h"`), `extra` (fields sent
  verbatim). Per provider: `docs/reference/parameters.md`.
- `client_options` (*dict | None, optional*, default `None`) — HTTP client settings: `url` (base URL), `timeout`
  (`urllib3.Timeout`), `retries` (`urllib3.Retry`), `proxy` (dict).
- `api_key` (*str | None, optional*, default `None`) — Provider key; read from the provider's environment variable when
  omitted. `ValueError` if neither is found.
- `on_unsupported` (*str, optional*, default `'warn'`) — `"warn"` (default) adapts and records in `last_adaptations`;
  `"requirements"` raises only for losses that leave a result looking
  right (`seed`, `size`, `aspect_ratio`, ...); `"raise"` for any.

**Attributes**

- `name, temperature, max_tokens, top_p, top_k, cache_control, reasoning` — The resolved settings.
- `client` (*family client, ready to use.*)

**Examples**

>>> Model("claude-sonnet-4-5", options={"temperature": 0.5}).temperature
0.5

### `Model.effective_options`

What sampling this model actually runs with, asked for or not.

Provider defaults differ (`temperature` from 0.0 to 1.0,
`max_tokens` from 2048 to 32768), so two models compared with no
options set are not one variable apart: record this beside a
measurement. The defaults per provider: `docs/reference/parameters.md`.

### `Model.to_request()`

```python
Model.to_request(
    messages: list,
    output: dict,
    tools: list | None = None,
) -> tuple[str, dict]
```

Translate universal *messages* + *output* into the provider's native
`(path, body)` pair, through the family client.

**Parameters**

- `messages` (*list*) — Universal messages, placeholders filled: `{"role", "parts"}`.
- `output` (*dict*) — The output spec: `{"modalities": [...], "format": {"type": …}}`.
- `tools` (*list | None, optional*, default `None`) — Function-calling schemas (`Tool.schema()`); a family without a
  tool wire format refuses them rather than send a request without.

### `Model.build()`

```python
Model.build(messages: list, output: dict, tools: list | None = None)
```

`to_request`, and what building changed: `(path, body, adaptations)`.

The adaptations belong to this request; `last_adaptations` is only a
mirror, unsafe on a model shared between threads.

**Parameters**

- `messages` (*list*) — Universal messages, placeholders already filled.
- `output` (*dict*) — The output spec.
- `tools` (*list | None, optional*, default `None`) — Function-calling schemas, as `Tool.schema()` returns them.

### `Model.stream()`

```python
Model.stream(messages: list, output: dict, tools: list | None = None)
```

Yield the answer in pieces as the provider produces it.

A provider that cannot stream returns the whole answer as one piece
and records a `declined` adaptation (`on_unsupported="raise"`
raises on it).

**Parameters**

- `messages` (*list*) — Universal messages, placeholders filled: `{"role", "parts"}`.
- `output` (*dict*) — The output spec: `{"modalities": [...], "format": {"type": …}}`.
- `tools` (*list | None, optional*, default `None`) — Function-calling schemas, as `Tool.schema()` returns them.

**Returns**

- `StreamOutcome` — On exhaustion, for `yield from`: result, usage, adaptations.
  `last_stream_usage`/`last_stream_result` mirror it.

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

Return *schema* in the form *target* accepts; the original is untouched.

Not lossless. A map with arbitrary keys has no strict equivalent, and
for `"openai"` it raises instead of being mangled.

**Parameters**

- `schema` (*dict*) — A JSON Schema.
- `target` (*str*) — The provider family: `"openai"` (strict mode) or `"google"`.

---

## `check_structure()`

```python
check_structure(value: object, schema: object, path: str = '$') -> list[str]
```

Compare *value* against *schema*, returning one message per violation.

Empty means nothing found, not full validity: types, required and
unexpected properties, enums, items and limits, recursively.

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
tool_result_turn(call_id: str, result, is_error: bool = False) -> dict
```

One call's result as a universal `tool` turn.

A non-string result is serialised to text, except media parts (alone or
in a list), carried as parts for each family to place on its wire.

**Parameters**

- `call_id` (*str*) — The `ToolCall.id` this result answers.
- `result` (*any*) — What the tool returned; a non-string is rendered as text.
- `is_error` (*bool, optional*, default `False`) — The call failed and *result* is its error, marked as such on the wire.

---

## `dangling_calls()`

```python
dangling_calls(messages: list) -> list
```

Ids of tool calls in *messages* that no later turn answers.

Providers disagree about an unanswered call (some reject, a self-hosted
server lets the model improvise), so `Agent.step()` refuses one; a
driver can ask first. See `docs/agents/driving-the-loop.md`.

**Parameters**

- `messages` (*list*) — A conversation, as `Agent.step()` takes it.
<!-- /g:api -->
