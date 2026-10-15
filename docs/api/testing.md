# API: Testing

Generated from the code's signatures and docstrings. What it is for and how to use it: [the guide](../production.md#testing-without-a-provider).

<!-- g:api -->
[`ScriptedModel`](#scriptedmodel) · [`call`](#call) · [`recorded`](#recorded) · [`Request`](#request)

---

## `ScriptedModel`

```python
ScriptedModel(
    replies: list,
    *,
    usage: dict | None = None,
    price: dict | None = None,
    name: str = 'scripted',
)
```

A model that replays a script, for tests and offline examples.

Each reply is a string (an answer), a dict (a JSON answer),
`call(name, **arguments)` (a tool call) or a list of calls (several in
one turn). The last reply repeats if the loop asks for more. It goes
through the library's own request and response path, so `Skill`,
`Chain`, `Pool` and `Agent` treat it as any other model — `run()` and
`stream()` both, a streamed answer arriving word by word.

**Parameters**

- `replies` (*list*) — The script, one item per model call.
- `usage` (*dict, optional*, default `None`) — Tokens each call reports: `{"input": 1000, "output": 200}`. Zero when
  not given.
- `price` (*dict, optional*, default `None`) — USD per million tokens: `{"input": 1.0, "output": 3.0}`. With it,
  each call reports its cost and a budget has something to count;
  without it the cost is `None`, as for any unpriced model.
- `name` (*str, optional*, default `'scripted'`) — The model's name in events and records. Default `"scripted"`.

**Attributes**

- `requests` (*list[Request]*) — What the loop sent each turn — messages and tool schemas.

**Examples**

>>> from yait_aichain import Agent
>>> model = ScriptedModel([call("refund", amount=42), "Refunded."])
>>> Agent(model, tools=[refund]).run("refund 42").output
'Refunded.'

### `ScriptedModel.build()`

```python
ScriptedModel.build(messages, output, tools=None)
```

`to_request`, and what building changed: `(path, body, adaptations)`.

The adaptations belong to this request; `last_adaptations` is only a
mirror, unsafe on a model shared between threads.

**Parameters**

- `messages` (*list*) — Universal messages, placeholders already filled.
- `output` (*dict*) — The output spec.
- `tools` (*list | None, optional*, default `None`) — Function-calling schemas, as `Tool.schema()` returns them.

### `ScriptedModel.stream()`

```python
ScriptedModel.stream(messages, output, tools=None)
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

---

## `call()`

```python
call(name: str, **arguments) -> _Call
```

A tool call in a `ScriptedModel`'s script: `call("refund", amount=42)`.

**Parameters**

- `name` (*str*) — The tool the model calls.
- `**arguments` — The arguments it calls it with.

---

## `recorded()`

```python
recorded(model: Model, reply: str = 'ok')
```

Put a real `Model` on a canned transport and record what it sends.

Every call in the block answers *reply* (10 input, 5 output tokens) in the
provider's own shape, offline; options, adaptations and cost stay the
library's real answers. Buffered calls only; a stream raises.

**Parameters**

- `model` (*Model*) — The model whose transport is replaced for the duration of the block.
- `reply` (*str, optional*, default `'ok'`) — The text every call answers. Default `"ok"`.

**Yields**

- `An object whose ``requests`` lists one `Request` per call: ``path``,`
- ```body`` and ``headers``, an auth header as a fingerprint, never the key.`

**Examples**

>>> model = Model("gpt-5.5")
>>> with recorded(model) as rec:
...     Skill(model, prompt="Hi").run()
>>> rec.requests[0].body["model"]
'gpt-5.5'

---

## `Request`

```python
Request(
    messages: list = <factory>,
    tools: list = <factory>,
    path: str = '',
    body: dict = <factory>,
    headers: dict = <factory>,
)
```

One request a model sent.

For a `ScriptedModel`: `messages` and `tools` as the loop handed them.
Under `recorded()`: `path`, `body` and `headers` as they would have
gone on the wire, an auth header as a fingerprint (`sk-…a1b2`).

**Parameters**

- `messages` (*list*, default `<factory>`) — The universal messages of the turn.
- `tools` (*list*, default `<factory>`) — The tool schemas offered.
- `path` (*str*, default `''`) — The request path.
- `body` (*dict*, default `<factory>`) — The request body.
- `headers` (*dict*, default `<factory>`) — The request headers.
<!-- /g:api -->
