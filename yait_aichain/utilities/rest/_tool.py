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
|               | the model; wrap it in ``state.Secret`` so a paused chain  |
|               | does not store it) or ``env``; two of them is an error.   |
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
import re
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
#: The most of a response REST reads.
MAX_RESPONSE_BYTES = 10 << 20
# raise_on_status=False: retries spent, the last answer comes back and is
# raised as the API's own typed error, where it was a raw MaxRetryError.
_DEFAULT_RETRIES = Retry(total=2, backoff_factor=0.5, raise_on_status=False,
                         status_forcelist={429, 502, 503, 504})

#: What each auth type reads. A key outside its type's set was dropped and
#: the request went anonymous — GitHub answers that with less data, not 401.
_AUTH_KEYS = {
    "bearer":  {"type", "token", "token_field", "env"},
    "cookie":  {"type", "token", "token_field", "env", "cookie_name"},
    "api_key": {"type", "token", "token_field", "env", "header"},
    "query":   {"type", "token", "token_field", "env", "param"},
    "basic":   {"type", "username_env", "password_env"},
}

#: Content types a body can be written in.
_FORM = "application/x-www-form-urlencoded"

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _field_schema(name: str, spec) -> dict:
    """A field's schema from ``(type, description)``, a type alone, or nothing."""
    from ...tools._base import _json_type
    kind, desc = (spec if isinstance(spec, tuple) else (spec, None)) if spec is not None else (None, None)
    if kind is None:
        out = {"type": "string"}
    elif isinstance(kind, str):
        out = {"type": kind}
    else:
        out = _json_type(kind, f"REST field {name!r}")
    out["description"] = desc or name
    return out


def _text(value) -> str:
    """A value as a URL carries it: `true`, not `True`."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


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
        fields:          dict | None = None,
        allow_private:   bool | None = None,
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
        self._allow_private  = allow_private
        self._fields         = dict(fields or {})
        self._check(name)
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

        # Typed where `fields` says how: every field was a string with its
        # own name as its description, and an array or an object could not be
        # asked for at all.
        self.parameters = {
            "type": "object",
            "properties": {field: _field_schema(field, self._fields.get(field))
                           for field in sorted(all_fields)},
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

    def _check(self, name: str) -> None:
        """Settings that would be dropped in silence, refused by name."""
        placed = [*self._path_params, *self._query_params, *self._body_params]
        if bad := [f for f in placed if not _IDENT.match(str(f))]:
            raise ValueError(f"RestApiTool {name!r}: {bad} cannot be field names — letters, digits and _")
        if twice := sorted({f for f in placed if placed.count(f) > 1}):
            raise ValueError(f"RestApiTool {name!r}: {twice} are placed twice (path, query, body); "
                             "a field goes one place")
        holes = set(re.findall(r"\{([^{}]*)\}", self._url_template))
        if missing := sorted(holes - set(self._path_params)):
            # Sent as written: `/items/{id}` reached the server with the braces.
            raise ValueError(f"RestApiTool {name!r}: the URL has {missing} and path_params does not "
                             "name them")
        if unused := sorted(set(self._path_params) - holes):
            raise ValueError(f"RestApiTool {name!r}: path_params {unused} are not in the URL")
        if stray := sorted(set(self._fields) - set(placed) - {self._auth.get("token_field")}):
            raise ValueError(f"RestApiTool {name!r}: fields= describes {stray}, which no list places")
        if self._body_params and self._content_type != _FORM and "json" not in self._content_type:
            raise ValueError(f"RestApiTool {name!r}: a body is written as JSON or as {_FORM}; "
                             f"{self._content_type!r} is neither")
        if self._auth:
            kind = str(self._auth.get("type", "")).lower()
            if kind not in _AUTH_KEYS:
                raise ValueError(f"RestApiTool {name!r}: auth type {self._auth.get('type')!r} is not one "
                                 f"of {sorted(_AUTH_KEYS)}")
            if unknown := sorted(set(self._auth) - _AUTH_KEYS[kind]):
                raise ValueError(f"RestApiTool {name!r}: auth {kind!r} reads {sorted(_AUTH_KEYS[kind])}; "
                                 f"{unknown} would be ignored")
        query = urllib.parse.urlsplit(self._url_template).query
        if query and any(w in query.lower() for w in ("key", "token", "secret", "password")):
            raise ValueError(f"RestApiTool {name!r}: the URL carries a credential in its query; it "
                             "would be in every error and every saved chain — use auth={'type': "
                             "'query', 'param': …, 'env': …}")

    def run(
        self,
        input:   dict | str | None = None,
        options: dict | None        = None,
        /,
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
            both is an error.  This is the form used by Chains and agents.
            *input* and *options* are positional only, so a field named
            ``input`` or ``options`` is passed here like any other.

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
            text = _text(val)
            # `quote` leaves `.` and `..` alone and the HTTP layer collapses
            # them, so `{mid}` = ".." turned DELETE /projects/1/members/{mid}
            # into DELETE /projects/1. An empty value made `//`.
            if text in ("", ".", ".."):
                raise ValueError(f"RestApiTool '{self.name}': path parameter '{p}' "
                                 f"cannot be {text!r}")
            url = url.replace("{" + p + "}", urllib.parse.quote(text, safe=""))

        # ── 3. Build query string ──────────────────────────────────────
        # A list repeats its key and a bool is `true`: `['a', 'b']` went as
        # its repr and `True` as "True".
        qs_parts: list[tuple[str, Any]] = []
        for p in self._query_params:
            val = data.get(p)
            if val is not None:
                qs_parts.append((p, [_text(v) for v in val] if isinstance(val, (list, tuple))
                                 else _text(val)))
        shown_url = url                          # what an error may print
        qs_parts += self._query_auth(data)
        if qs_parts:
            connector = "&" if "?" in url else "?"
            url = f"{url}{connector}{urllib.parse.urlencode(qs_parts, doseq=True)}"

        # ── 4. Build request body ──────────────────────────────────────
        body: bytes | None = None
        if self._body_params:
            body_dict = {
                p: data[p]
                for p in self._body_params
                if p in data
            }
            if body_dict:
                # Written in the type the header names: a form endpoint was
                # sent JSON under a form content type.
                body = (urllib.parse.urlencode({k: ([_text(x) for x in v] if isinstance(v, (list, tuple))
                                                    else _text(v)) for k, v in body_dict.items()},
                                               doseq=True)
                        if self._content_type == _FORM else json.dumps(body_dict)).encode("utf-8")

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
            # The connect timeout kept: `timeout=5` alone left connecting unbounded.
            req_kwargs["timeout"] = urllib3.Timeout(connect=10.0, read=float(self._timeout))

        # Only ever speak http(s), and block SSRF targets (private / loopback /
        # metadata IPs) unless AICHAIN_ALLOW_PRIVATE_URLS is set — needed when a
        # REST endpoint is on an internal network.
        response = safe_request(self._http, self._method, url, preload_content=False,
                                allow_private=self._allow_private, **req_kwargs)
        try:
            data = response.read(MAX_RESPONSE_BYTES + 1)
        finally:
            response.release_conn()
        if len(data) > MAX_RESPONSE_BYTES:
            raise ValueError(f"RestApiTool '{self.name}': the response is over "
                             f"{MAX_RESPONSE_BYTES:,} bytes, too large to hand a model; "
                             "ask the API for less (a page, a filter)")
        raw = data.decode("utf-8", errors="replace")

        if not (200 <= response.status < 300):
            # Typed, as every other API error is, and without the query: an
            # auth key in it was in every error message.
            from ...clients._errors import error_from_status
            raise error_from_status(response.status,
                                    f"RestApiTool '{self.name}' {self._method} {shown_url}: {raw[:500]}",
                                    response.headers)

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
            token = self._need_secret(auth, data)
            headers["Authorization"] = f"Bearer {token}"

        elif auth_type == "cookie":
            token = self._need_secret(auth, data)
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
            if not (username or password):
                raise ValueError(f"RestApiTool '{self.name}': basic auth reads "
                                 f"{auth.get('username_env')!r} and {auth.get('password_env')!r}, "
                                 "and neither is set — the request would have gone anonymous")
            if username or password:
                encoded = base64.b64encode(
                    f"{username}:{password}".encode()
                ).decode()
                headers["Authorization"] = f"Basic {encoded}"

        elif auth_type == "api_key":
            token = self._need_secret(auth, data)
            header_name = auth.get("header", "X-Api-Key")
            headers[header_name] = token

        return headers

    def _query_auth(self, data: dict) -> list:
        if str(self._auth.get("type", "")).lower() != "query":
            return []
        return [(self._auth.get("param", "key"), self._need_secret(self._auth, data))]

    def _need_secret(self, auth: dict, data: dict) -> str:
        """The token, or an error saying where it was looked for. An unset
        variable sent the request anonymous, and some APIs answer that with
        less data rather than with a 401."""
        token = self._resolve_secret(auth, data)
        if not token:
            where = (f"the environment variable {auth['env']!r}" if auth.get("env") else
                     f"the field {auth['token_field']!r}" if auth.get("token_field") else
                     "auth['token']")
            raise ValueError(f"RestApiTool '{self.name}': no token in {where}; the request would "
                             "have gone without its credential")
        return token

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
            from ...state._secret import reveal
            return str(reveal(data[token_field]))
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
            "fields": ({k: (v if isinstance(v, (str, tuple)) or v is None else str(v))
                        for k, v in self._fields.items()} or None),
            "allow_private": self._allow_private,
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
