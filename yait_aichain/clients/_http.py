"""
One HTTP call, for the model clients and for every tool.

Each client and tool used to carry its own try / status check / error: four
copies in the base client, five in the vector backends, one in nearly every
tool — and they had drifted. Some raised a typed `APIError`, others a bare
`RuntimeError` carrying the whole body, so `except RateLimitError` around a
tool caught nothing and an HTML error page reached a model's context.

`request()` is the one shape: the body serialised before the call (a value
that does not serialise is a bug to see, not a network failure to retry), a
transport failure as `NetworkError`, a status as the typed error
`error_from_status` gives it — with the body clipped, since an error message
travels into logs, run records and model context.
"""

from __future__ import annotations

import json as _json

from ._errors import NetworkError, error_from_status

#: How much of an error body an exception carries.
MAX_ERROR_BODY = 2000


def request(http, method: str, url: str, *, json=None, what: str = "", **kwargs):
    """Send one request through *http* (a urllib3 pool) and return the response.

    *json* is serialised as the body with its content type; any other keyword
    goes to urllib3 as is (``headers``, ``fields``, ``retries``, ...). *what*
    names the API in an error — "Brave Search API: ..." rather than a bare body.
    Raises `NetworkError` when no response came back, and the typed error for
    the status otherwise.
    """
    if json is not None:
        kwargs["body"] = _json.dumps(json).encode("utf-8")
        kwargs["headers"] = {"Content-Type": "application/json", **(kwargs.get("headers") or {})}
    try:
        response = http.request(method, url, **kwargs)
    except Exception as exc:
        raise NetworkError(0, f"{what}: {exc}" if what else str(exc)) from exc
    if 200 <= response.status < 300:
        return response
    body = response.data.decode("utf-8", errors="replace")[:MAX_ERROR_BODY]
    raise error_from_status(response.status, f"{what}: {body}" if what else body, response.headers)


def request_json(http, method: str, url: str, **kwargs) -> dict:
    """`request()`, with the response read as JSON."""
    return _json.loads(request(http, method, url, **kwargs).data)
