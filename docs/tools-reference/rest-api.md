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

## `REST` — an API as a utility

```python
from yait_aichain.utilities import REST

api = REST("https://restful-booker.herokuapp.com", headers={"Accept": "application/json"})
get_booking = api.endpoint("get_booking", "GET", "/booking/{id}", path_params=["id"],
                           description="One booking by its id.")
get_booking.run(id=1)
Agent(model, tools=api.tools)                        # every endpoint made so far
```

The base URL, `auth` and `headers` are the API's, set once and shared by every
endpoint. An endpoint is a `RestApiTool`, named and described for the model, and
reachable as `api.get_booking` once made.
