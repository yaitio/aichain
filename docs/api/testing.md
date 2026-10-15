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

The adaptations are this request's, returned with it. One Model is
shared — by a Pool's threads, by tenants — so a fact about one call
left on the object (`last_adaptations`, kept as a mirror) is read by
whichever call looks next.

**Parameters**

- `messages` (*list*) — Universal messages, placeholders already filled.
- `output` (*dict*) — The output spec.
- `tools` (*list | None, optional*, default `None`) — Function-calling schemas, as `Tool.schema()` returns them.

### `ScriptedModel.stream()`

```python
ScriptedModel.stream(messages, output, tools=None)
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

Inside the block every call answers *reply* in the provider's own response
shape, with 10 input and 5 output tokens, and nothing reaches the network.
The model keeps its name, options, adaptations and price, so
`effective_options`, `last_adaptations` and the cost are the library's
real answers. Buffered calls only; a stream inside the block raises.

**Parameters**

- `model` (*Model*) — The model whose transport is replaced for the duration of the block.
- `reply` (*str, optional*, default `'ok'`) — The text every call answers. Default `"ok"`.

**Yields**

- `An object whose ``requests`` lists one `Request` per call: ``path``,`
- ```body`` and ``headers``, an auth header as a fingerprint, never the key.`

**Examples**

>>> model = Model("gpt-5.5", options={"temperature": 0.2})
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
