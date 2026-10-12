"""
utilities.rest._tool — RestApiTool
===================================

A REST endpoint as a tool.  One instance = one configured endpoint; the
usual way to make one is ``REST(base_url, …).endpoint(…)``.
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

The two forms are merged; a field given in both is an error.

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
|               | Source: one of ``token`` (a value or a callable),         |
|               | ``token_field`` (a field code or a chain fills — never    |
|               | the model) or ``env``; two of them is an error.           |
+---------------+-----------------------------------------------------------+
| ``"cookie"``  | ``Cookie: <cookie_name>=<value>``                         |
|               | Source: as for ``"bearer"``.                              |
|               | Default cookie_name: ``"token"``                          |
+---------------+-----------------------------------------------------------+
| ``"basic"``   | ``Authorization: Basic base64(user:pass)``                |
|               | Sources: ``username_env`` / ``password_env`` env vars.    |
+---------------+-----------------------------------------------------------+
| ``"api_key"`` | Custom header set to ``<value>``                          |
|               | Header name: ``header`` (default ``"X-Api-Key"``).        |
|               | Source: as for ``"bearer"``.                              |
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

    from yait_aichain.utilities import REST

    api = REST("https://api.example.com", description="Example user directory",
               auth={"type": "bearer", "env": "EXAMPLE_API_KEY"})
    create_user = api.endpoint("create_user", "POST", "/users",
                               description="Create a new user account.",
                               body_params=["username", "email", "role"],
                               required_params=["username", "email"])

    # Call it directly
    result = create_user.run({"username": "alice", "email": "a@b.com"})

    # Or in a Chain — the Chain passes matching accumulated variables
    # automatically as individual kwargs:
    (create_user, "user_response")
"""

from __future__ import annotations

import base64
import json
import os
import urllib.parse
from typing import Any

import urllib3
from yait_aichain.clients._base import make_http
from yait_aichain.clients._constants import Retry

from ...tools._base import Tool
from ...tools._permissions import RISK_CLASSES
from ...tools._security import safe_request

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT = urllib3.Timeout(connect=10.0, read=30.0)
_DEFAULT_RETRIES = Retry(total=2, backoff_factor=0.5,
                         status_forcelist={429, 502, 503, 504})


# What a method does, as a risk class. HTTP calls GET, HEAD and OPTIONS safe —
# they change nothing — and DELETE removes; the rest change state on a service
# the library does not own, which the risk table calls `external`.
_METHOD_RISK = {"GET": "read", "HEAD": "read", "OPTIONS": "read", "DELETE": "destructive"}


def risk_for_method(method: str) -> str:
    """The risk class an endpoint has when nobody says: from its HTTP method."""
    return _METHOD_RISK.get(method.upper(), "external")


# ---------------------------------------------------------------------------
# RestApiTool
# ---------------------------------------------------------------------------

class RestApiTool(Tool):
    """
    A single REST endpoint exposed as a library Tool.

    ``REST(...).endpoint(...)`` builds one with the API's base URL, auth and
    headers filled in; construct it directly for a lone endpoint.

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
    risk : str | None, optional
        The risk class a permission policy reads. Omitted, it follows the
        method: ``GET``, ``HEAD`` and ``OPTIONS`` change nothing and are
        ``read``; ``DELETE`` is ``destructive``; ``POST``, ``PUT`` and
        ``PATCH`` change something on another service and are ``external``.
        Say otherwise when you know better — a ``POST`` that only searches is
        ``read``.

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
        timeout:         float | None = None,
        risk:            str | None  = None,
    ) -> None:
        # ── Public identity (Tool class attributes) ────────────────────
        self.name        = name
        self.description = description
        self.risk        = risk_for_method(method) if risk is None else risk
        if self.risk not in RISK_CLASSES:
            raise ValueError(f"RestApiTool {name!r}: risk {risk!r} is not one of {sorted(RISK_CLASSES)}")

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
        self._timeout        = timeout
        self._required_params = list(required_params)
        # The token has one source: a callable, a field the code or a chain
        # fills, or an environment variable. Two of them is an error, so none
        # can win over another in silence.
        sources = [k for k in ("token", "token_field", "env") if self._auth.get(k)]
        if len(sources) > 1:
            raise ValueError(f"RestApiTool {name!r}: auth takes one source of its token; got "
                             f"{', '.join(sources)}")
        #: Fields a model never sees nor fills: the token. A chain still fills it.
        self._hidden = {self._auth["token_field"]} if self._auth.get("token_field") else set()

        # Every declared field is a schema property, so agents and chains know
        # what to pass — the token's field too, which only a chain fills.
        all_fields = set(path_params) | set(query_params) | set(body_params) | self._hidden

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
        self._http = make_http(
            timeout = _DEFAULT_TIMEOUT,
            retries = _DEFAULT_RETRIES,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        input:   dict | str | None = None,
        options: dict | None        = None,
        **kwargs: Any,
    ) -> dict | list | str:
        """
        Execute the configured REST endpoint and return the response.

        Parameters
        ----------
        input : dict | str | None, optional
            Field values as a dict.  A string value is treated as a
            single ``"input"`` field, for Chain steps that pass a plain
            string.
        options : dict | None, optional
            None. Headers and the timeout are settings of the tool; options
            given here are an error that says so.
        **kwargs
            Individual field values, merged with *input*; a field given in
            both is an error.  This is the form used by Chains — each accumulated
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
        if clash := sorted(set(data) & set(kwargs)):
            raise ValueError(f"RestApiTool {self.name!r}: {clash} given both in the input dict and "
                             "as keyword arguments — pass each field once")
        data.update(kwargs)

        # Headers and the timeout are the tool's settings, not per-call options.
        if options:
            moved = [k for k in options if k in ("headers", "timeout")]
            raise ValueError(f"RestApiTool {self.name!r}: "
                             + (f"{', '.join(moved)} is a setting of the tool — RestApiTool(…, "
                                f"{moved[0]}=…) or REST(…, headers=…)" if moved else
                                f"a call takes the endpoint's fields, not options {sorted(options)}"))

        # ── 2. Build URL (substitute path params) ─────────────────────
        url = self._url_template
        for p in self._path_params:
            val = data.get(p)
            if val is None:
                raise ValueError(
                    f"RestApiTool '{self.name}': "
                    f"path parameter '{p}' is required but was not provided."
                )
            text = str(val)
            # `quote` leaves `.` and `..` alone and the HTTP layer collapses
            # them, so `{mid}` = ".." turned DELETE /projects/1/members/{mid}
            # into DELETE /projects/1. An empty value made `//`.
            if text in ("", ".", ".."):
                raise ValueError(f"RestApiTool '{self.name}': path parameter '{p}' "
                                 f"cannot be {text!r}")
            url = url.replace("{" + p + "}", urllib.parse.quote(text, safe=""))

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

        # ── 6. Execute ─────────────────────────────────────────────────
        req_kwargs: dict[str, Any] = {"headers": headers}
        if body is not None:
            req_kwargs["body"] = body

        if self._timeout is not None:
            req_kwargs["timeout"] = urllib3.Timeout(read=float(self._timeout))

        # Only ever speak http(s), and block SSRF targets (private / loopback /
        # metadata IPs) unless AICHAIN_ALLOW_PRIVATE_URLS is set — needed when a
        # REST endpoint is on an internal network.
        response = safe_request(self._http, self._method, url, **req_kwargs)
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
        token = auth.get("token")
        if token is not None:
            return str(token() if callable(token) else token)
        token_field = auth.get("token_field")
        if token_field and token_field in data:
            return str(data[token_field])
        env_key = auth.get("env")
        if env_key:
            return os.environ.get(env_key)
        return None

    # ------------------------------------------------------------------
    # Saving in a chain
    # ------------------------------------------------------------------

    #: Header names a secret travels in. A saved chain is a file, and a
    #: value in one of these would be written into it.
    _SECRET_HEADER = ("authorization", "cookie", "token", "key", "secret", "password")

    def _serialise_init_args(self) -> dict:
        """What `Chain.save` writes: the endpoint, never a secret.

        A token given as a value or a callable is not written — a file is
        not a place for it — and saving refuses, naming the two sources that
        can be saved: ``env`` (the variable's name) and ``token_field`` (a
        field the chain fills). A static header whose name says it carries a
        secret is refused for the same reason.
        """
        if self._auth.get("token") is not None:
            raise ValueError(
                f"RestApiTool {self.name!r} cannot be saved: its token is a value or a "
                f"callable, and a saved chain is a file. Give it by name — "
                f"auth={{..., 'env': 'VARIABLE'}} — or as a field the chain fills — "
                f"auth={{..., 'token_field': 'token'}}.")
        secret = [h for h in self._static_headers
                  if any(word in h.lower() for word in self._SECRET_HEADER)]
        if secret:
            raise ValueError(
                f"RestApiTool {self.name!r} cannot be saved: header(s) {secret} would "
                f"write a secret into the file. Put the credential in auth=, by env name.")
        args = {
            "name": self.name, "description": self.description,
            "method": self._method, "url": self._url_template,
            "path_params": self._path_params, "query_params": self._query_params,
            "body_params": self._body_params, "required_params": self._required_params,
            "headers": self._static_headers or None, "auth": self._auth or None,
            "content_type": self._content_type, "response_field": self._response_field,
            "timeout": self._timeout, "risk": self.risk,
        }
        return {k: v for k, v in args.items() if v not in (None, [], ())}

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def schema(self) -> dict:
        """The schema a model sees: the endpoint's fields, without the token."""
        out = super().schema()
        params = dict(out["function"]["parameters"])        # a copy: self.parameters stays whole for a chain
        params["properties"] = {k: v for k, v in params["properties"].items() if k not in self._hidden}
        params["required"] = [k for k in params.get("required", []) if k not in self._hidden]
        out["function"] = {**out["function"], "parameters": params}
        return out

    def check_args(self, kwargs: dict) -> str | None:
        if given := sorted(set(kwargs) & self._hidden):
            return (f"Invalid call to tool '{self.name}': {given} is the API's credential and is "
                    "not an argument of the call.")
        return super().check_args(kwargs)

    def __repr__(self) -> str:
        return (
            f"RestApiTool(name={self.name!r}, "
            f"method={self._method!r}, "
            f"url={self._url_template!r})"
        )
