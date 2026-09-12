<!-- g:tool -->
# `RestApiTool`

A single REST endpoint exposed as a library Tool.

| | |
|---|---|
| Import | `from yait_aichain.tools import RestApiTool` |
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
)
```

The call schema is built from the constructor arguments, so it is whatever the caller declares.
<!-- /g:tool -->
