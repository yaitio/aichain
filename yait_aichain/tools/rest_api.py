"""
tools.rest_api — RestApiTool
=============================

A universal REST endpoint tool.  One instance = one configured endpoint.
No boilerplate, no framework — drop any REST call directly into a Chain,
Agent, or standalone script.

Design
------
``RestApiTool`` is constructed once with the static parts of an API call
(method, URL, which params go where, auth strategy) and called many times
with the dynamic parts (the actual values).

``run()`` accepts data in two equivalent forms so the tool works naturally
both directly and inside a Chain:

  * ``run(input={"bookingid": 1, "token": "abc"})``  — dict, direct use
  * ``run(bookingid=1, token="abc")``                 — individual kwargs,
                                                         Chain auto-passes
                                                         from accumulated vars

The two forms are merged, so partial overlap is fine.

Parameter routing
-----------------
The constructor declares which field names go where:

  * ``path_params``  — substituted into ``{field}`` placeholders in the URL
  * ``query_params`` — appended to the URL as ``?field=value``
  * ``body_params``  — serialised as JSON in the request body

Values can be any JSON-serialisable type (str, int, bool, list, dict).
Fields absent from the call are silently omitted.

Auth strategies (``auth`` kwarg)
---------------------------------
+---------------+-----------------------------------------------------------+
| ``type``      | Behaviour                                                 |
+===============+===========================================================+
| ``"bearer"``  | ``Authorization: Bearer <value>``                         |
|               | Source: ``token_field`` from data, else ``env`` var.      |
+---------------+-----------------------------------------------------------+
| ``"cookie"``  | ``Cookie: <cookie_name>=<value>``                         |
|               | Source: ``token_field`` from data, else ``env`` var.      |
|               | Default cookie_name: ``"token"``                          |
+---------------+-----------------------------------------------------------+
| ``"basic"``   | ``Authorization: Basic base64(user:pass)``                |
|               | Sources: ``username_env`` / ``password_env`` env vars.    |
+---------------+-----------------------------------------------------------+
| ``"api_key"`` | Custom header set to ``<value>``                          |
|               | Header name: ``header`` (default ``"X-Api-Key"``).        |
|               | Source: ``token_field`` from data, else ``env`` var.      |
+---------------+-----------------------------------------------------------+

Response handling
-----------------
* Valid JSON → returned as-is (``dict`` or ``list``).
  When the Chain receives a **dict**, it merges the keys directly into the
  accumulated variable dict — field names become variables automatically.
* Non-JSON (plain text, HTML) → returned as ``str``.
* ``response_field`` — when set, extracts ``response[response_field]``
  from a dict response before returning.
* HTTP errors → raise ``RuntimeError`` with the status code and body.

Quick example
-------------
::

    from tools import RestApiTool

    # Declare the endpoint once
    create_user = RestApiTool(
        name        = "create_user",
        description = "Create a new user account.",
        method      = "POST",
        url         = "https://api.example.com/users",
        body_params = ["username", "email", "role"],
        required_params = ["username", "email"],
        auth        = {"type": "bearer", "env": "EXAMPLE_API_KEY"},
    )

    # Call it directly
    result = create_user.run(input={"username": "alice", "email": "a@b.com"})

    # Or in a Chain — the Chain passes matching accumulated variables
    # automatically as individual kwargs:
    (create_user, "user_response")

See ``examples/booking_api.py`` for a complete Chain example with
the Restful Booker test API.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.parse
from typing import Any

import urllib3

from ._base import Tool

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT = urllib3.Timeout(connect=10.0, read=30.0)
_DEFAULT_RETRIES = urllib3.Retry(total=2, backoff_factor=0.5,
                                  status_forcelist={429, 502, 503, 504})


# ---------------------------------------------------------------------------
# RestApiTool
# ---------------------------------------------------------------------------

class RestApiTool(Tool):
    """
    A single REST endpoint exposed as a library Tool.

    Parameters
    ----------
    name : str
        Unique machine-readable identifier.  Used as the tool's function
        name when registered with an Agent.
    description : str
        One-sentence description.  This is what an LLM reads to decide
        whether to use the tool.
    method : str
        HTTP method: ``"GET"``, ``"POST"``, ``"PUT"``, ``"PATCH"``,
        ``"DELETE"``.  Case-insensitive.
    url : str
        Endpoint URL.  May contain ``{field_name}`` placeholders for path
        parameters, e.g. ``"https://api.example.com/items/{id}"``.
    path_params : list[str], optional
        Field names whose values are substituted into ``{…}`` placeholders
        in *url*.
    query_params : list[str], optional
        Field names appended to the URL as query-string parameters.
    body_params : list[str], optional
        Field names serialised as a JSON object in the request body.
        Values may be any JSON-serialisable type (str, int, dict, list …).
    required_params : list[str], optional
        Subset of the above that are required.  Agents and validators use
        this list; ``run()`` does not enforce it — missing required fields
        simply produce an incomplete request.
    headers : dict | None, optional
        Static request headers merged into every call.
    auth : dict | None, optional
        Authentication strategy.  See module docstring for supported types
        and fields.
    content_type : str, optional
        ``Content-Type`` header for requests with a body.
        Default: ``"application/json"``.
    response_field : str | None, optional
        When set, ``run()`` returns ``response[response_field]`` instead of
        the full response dict.  Useful for extracting a single key from a
        wrapper object (e.g. ``response_field="token"`` for ``{"token": …}``).

    Examples
    --------
    Parameterless GET::

        ping = RestApiTool(
            name        = "ping",
            description = "Check API health.",
            method      = "GET",
            url         = "https://restful-booker.herokuapp.com/ping",
        )
        result = ping.run()

    GET with path parameter::

        get_booking = RestApiTool(
            name        = "get_booking",
            description = "Retrieve a booking by ID.",
            method      = "GET",
            url         = "https://restful-booker.herokuapp.com/booking/{bookingid}",
            path_params = ["bookingid"],
            required_params = ["bookingid"],
        )
        booking = get_booking.run(input={"bookingid": 1})

    POST with JSON body and Bearer auth from env::

        create_item = RestApiTool(
            name        = "create_item",
            description = "Create a new item.",
            method      = "POST",
            url         = "https://api.example.com/items",
            body_params = ["name", "price", "category"],
            required_params = ["name", "price"],
            auth        = {"type": "bearer", "env": "EXAMPLE_API_KEY"},
        )

    Cookie-based auth with token from a previous Chain step::

        delete_booking = RestApiTool(
            name        = "delete_booking",
            description = "Delete a booking (auth required).",
            method      = "DELETE",
            url         = "https://restful-booker.herokuapp.com/booking/{bookingid}",
            path_params = ["bookingid"],
            required_params = ["bookingid", "token"],
            auth        = {"type": "cookie", "cookie_name": "token",
                           "token_field": "token"},
        )
        # In a Chain: accumulated["bookingid"] and accumulated["token"] are
        # passed automatically as individual kwargs.
    """

    def __init__(
        self,
        name:            str,
        description:     str,
        method:          str,
        url:             str,
        *,
        path_params:     list[str]   = (),
        query_params:    list[str]   = (),
        body_params:     list[str]   = (),
        required_params: list[str]   = (),
        headers:         dict | None = None,
        auth:            dict | None = None,
        content_type:    str         = "application/json",
        response_field:  str | None  = None,
    ) -> None:
        # ── Public identity (Tool class attributes) ────────────────────
        self.name        = name
        self.description = description

        # ── Endpoint configuration ─────────────────────────────────────
        self._method         = method.upper()
        self._url_template   = url
        self._path_params    = list(path_params)
        self._query_params   = list(query_params)
        self._body_params    = list(body_params)
        self._static_headers = dict(headers or {})
        self._auth           = dict(auth or {})
        self._content_type   = content_type
        self._response_field = response_field

        # ── Auto-build parameters schema ───────────────────────────────
        # All declared field names become schema properties so Agents and
        # Chains know exactly what to pass.  The auth token field (when
        # sourced from data) is included too.
        all_fields: set[str] = (
            set(path_params) | set(query_params) | set(body_params)
        )
        token_field = self._auth.get("token_field")
        if token_field:
            all_fields.add(token_field)

        self.parameters = {
            "type": "object",
            "properties": {
                field: {
                    "type":        "string",
                    "description": field,
                }
                for field in sorted(all_fields)
            },
            "required": list(required_params),
        }

        # ── HTTP client ────────────────────────────────────────────────
        self._http = urllib3.PoolManager(
            timeout = _DEFAULT_TIMEOUT,
            retries = _DEFAULT_RETRIES,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        input:   "dict | str | None" = None,
        options: "dict | None"        = None,
        **kwargs: Any,
    ) -> "dict | list | str":
        """
        Execute the configured REST endpoint and return the response.

        Parameters
        ----------
        input : dict | str | None, optional
            Field values as a dict.  A string value is treated as a
            single ``"input"`` field (compatibility shim for plain-string
            Chain steps).
        options : dict | None, optional
            Runtime overrides:
              ``headers`` (dict) — merged on top of static + auth headers.
              ``timeout``  (float) — per-call read timeout in seconds.
        **kwargs
            Individual field values.  Merged with *input*; kwargs win on
            conflict.  This is the form used by Chains — each accumulated
            variable whose name matches a declared parameter is passed here
            automatically.

        Returns
        -------
        dict | list | str
            Parsed JSON response (dict or list), or raw text string when
            the response is not valid JSON.  When *response_field* is set,
            returns ``response[response_field]`` from a dict response.

        Raises
        ------
        RuntimeError
            On any non-2xx HTTP response.
        ValueError
            When a required path parameter is absent from the data.
        """
        # ── 1. Merge data sources (input dict + individual kwargs) ─────
        data: dict[str, Any] = {}
        if isinstance(input, dict):
            data.update(input)
        elif isinstance(input, str) and input:
            data["input"] = input
        data.update(kwargs)          # kwargs win on conflict

        opts = options or {}

        # ── 2. Build URL (substitute path params) ─────────────────────
        url = self._url_template
        for p in self._path_params:
            val = data.get(p)
            if val is None:
                raise ValueError(
                    f"RestApiTool '{self.name}': "
                    f"path parameter '{p}' is required but was not provided."
                )
            url = url.replace("{" + p + "}", urllib.parse.quote(str(val), safe=""))

        # ── 3. Build query string ──────────────────────────────────────
        qs_parts: list[tuple[str, str]] = []
        for p in self._query_params:
            val = data.get(p)
            if val is not None:
                qs_parts.append((p, str(val)))

        if qs_parts:
            connector = "&" if "?" in url else "?"
            url = f"{url}{connector}{urllib.parse.urlencode(qs_parts)}"

        # ── 4. Build request body ──────────────────────────────────────
        body: bytes | None = None
        if self._body_params:
            body_dict = {
                p: data[p]
                for p in self._body_params
                if p in data
            }
            if body_dict:
                body = json.dumps(body_dict).encode("utf-8")

        # ── 5. Build headers ───────────────────────────────────────────
        headers: dict[str, str] = {
            "Accept": "application/json",
        }
        if body is not None:
            headers["Content-Type"] = self._content_type
        headers.update(self._static_headers)
        headers = self._apply_auth(headers, data)
        headers.update(opts.get("headers") or {})

        # ── 6. Execute ─────────────────────────────────────────────────
        req_kwargs: dict[str, Any] = {"headers": headers}
        if body is not None:
            req_kwargs["body"] = body

        if opts.get("timeout") is not None:
            req_kwargs["timeout"] = urllib3.Timeout(read=float(opts["timeout"]))

        response = self._http.request(self._method, url, **req_kwargs)
        raw = response.data.decode("utf-8", errors="replace")

        if not (200 <= response.status < 300):
            raise RuntimeError(
                f"RestApiTool '{self.name}' — "
                f"HTTP {response.status} {self._method} {url}: {raw[:500]}"
            )

        # ── 7. Parse response ──────────────────────────────────────────
        result: Any
        try:
            result = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            return raw         # plain-text response (e.g. "Created")

        # Optionally extract a single field
        if self._response_field and isinstance(result, dict):
            return result.get(self._response_field, result)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_auth(self, headers: dict[str, str], data: dict[str, Any]) -> dict[str, str]:
        """
        Inject authentication headers based on the configured strategy.

        Mutates and returns a copy of *headers*.
        """
        auth = self._auth
        if not auth:
            return headers

        headers = dict(headers)
        auth_type = auth.get("type", "").lower()

        if auth_type == "bearer":
            token = self._resolve_secret(auth, data)
            if token:
                headers["Authorization"] = f"Bearer {token}"

        elif auth_type == "cookie":
            token = self._resolve_secret(auth, data)
            if token:
                cookie_name = auth.get("cookie_name", "token")
                # Merge with any existing Cookie header
                existing = headers.get("Cookie", "")
                new_cookie = f"{cookie_name}={token}"
                headers["Cookie"] = (
                    f"{existing}; {new_cookie}" if existing else new_cookie
                )

        elif auth_type == "basic":
            username = os.environ.get(auth.get("username_env", ""), "")
            password = os.environ.get(auth.get("password_env", ""), "")
            if username or password:
                encoded = base64.b64encode(
                    f"{username}:{password}".encode()
                ).decode()
                headers["Authorization"] = f"Basic {encoded}"

        elif auth_type == "api_key":
            token = self._resolve_secret(auth, data)
            if token:
                header_name = auth.get("header", "X-Api-Key")
                headers[header_name] = token

        return headers

    @staticmethod
    def _resolve_secret(
        auth: dict[str, str],
        data: dict[str, Any],
    ) -> str | None:
        """
        Resolve a token/secret from (in priority order):
        1. ``data[auth["token_field"]]``  — runtime value (e.g. from a prior Chain step)
        2. ``os.environ[auth["env"]]``    — environment variable
        """
        token_field = auth.get("token_field")
        if token_field and token_field in data:
            return str(data[token_field])
        env_key = auth.get("env")
        if env_key:
            return os.environ.get(env_key)
        return None

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"RestApiTool(name={self.name!r}, "
            f"method={self._method!r}, "
            f"url={self._url_template!r})"
        )
