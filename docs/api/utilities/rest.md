# API: rest

Generated from the code's signatures and docstrings. What it is for and how to use it: [the guide](../../utilities/index.md).

<!-- g:tool -->
## `REST`

```python
REST(
    base_url: str,
    *,
    description: str,
    auth: 'dict | None' = None,
    headers: 'dict | None' = None,
    timeout: 'float | None' = None,
)
```

An HTTP API: `endpoint(...)` makes each of its tools; `tools` lists them.

`description` says what the API is, and is required: an endpoint's own
description — "One booking by its id." — does not tell a model which
system it is in, so every endpoint carries the API's description before its
own: "Restful Booker, a hotel booking API — One booking by its id."

### `REST.endpoint()`

```python
REST.endpoint(
    name: str,
    method: str,
    path: str,
    *,
    description: str,
    path_params=(),
    query_params=(),
    body_params=(),
    required_params=(),
    content_type: str = 'application/json',
    response_field: 'str | None' = None,
) -> RestApiTool
```

A tool for one endpoint: *method* on the base URL plus *path* —
`"/booking/{id}"` with `path_params=["id"]`. Its fields are its
arguments; the API's auth and headers are the utility's.

### `REST.TOOLS`

Built-in immutable sequence.

If no argument is given, the constructor returns an empty tuple.
If iterable is specified the tuple is initialized from iterable's items.

If the argument is a tuple, the return value is the same object.

### `REST.tools`

The utility's tools, to hand an agent at once: `Agent(tools=kb.tools)`.

---

## The classes behind it

What the factory returns, with the schema a model is shown.

### `RestApiTool`

A single REST endpoint exposed as a library Tool.

| | |
|---|---|
| Import | `from yait_aichain.utilities.rest import RestApiTool` |
| Risk class | `write` |

```python
RestApiTool(
    name: str,
    description: str,
    method: str,
    url: str,
    *,
    path_params: list[str] = (),
    query_params: list[str] = (),
    body_params: list[str] = (),
    required_params: list[str] = (),
    headers: dict | None = None,
    auth: dict | None = None,
    content_type: str = 'application/json',
    response_field: str | None = None,
    timeout: float | None = None,
)
```

The call schema is built from the constructor arguments, so it is whatever the caller declares.
<!-- /g:tool -->
