"""One HTTP call, for the model clients and for every tool: ``request()``
serialises the body before the call (a value that does not serialise is a
bug, not a network failure to retry), raises ``NetworkError`` for transport
failures and the typed error for a status, with the body clipped since it
travels into logs and model context. See docs/models/models.md.
"""

from __future__ import annotations

import json as _json

import urllib3

from ._errors import NetworkError, error_from_status

#: What the network raises: urllib3's own errors and the OS's socket errors.
TRANSPORT_ERRORS = (urllib3.exceptions.HTTPError, OSError)

#: How much of an error body an exception carries.
MAX_ERROR_BODY = 2000


def _sendable(headers: dict) -> dict:
    """Header values stripped (a key read from a file keeps its newline); a
    control character left is refused naming the header, never its value,
    which is a credential the HTTP library would quote in its error."""
    out = {}
    for name, value in headers.items():
        if isinstance(value, str):
            value = value.strip()
            if any(ord(c) < 32 or ord(c) == 127 for c in value):
                raise ValueError(f"the {name} header contains a control character; check how "
                                 "its value was read — the value is not shown")
        out[name] = value
    return out


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
    if kwargs.get("headers"):
        kwargs["headers"] = _sendable(kwargs["headers"])
    try:
        response = http.request(method, url, **kwargs)
    except TRANSPORT_ERRORS as exc:
        # Only what the network does. A TypeError in our own code, or a
        # header that cannot be sent, became a NetworkError — retried, and
        # switching the fallback model — and carried the header in its text.
        raise NetworkError(0, f"{what}: {exc}" if what else str(exc)) from exc
    if 200 <= response.status < 300:
        return response
    body = response.data.decode("utf-8", errors="replace")[:MAX_ERROR_BODY]
    raise error_from_status(response.status, f"{what}: {body}" if what else body, response.headers)


def request_json(http, method: str, url: str, **kwargs) -> dict:
    """`request()`, with the response read as JSON."""
    return _json.loads(request(http, method, url, **kwargs).data)
