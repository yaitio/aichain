import codecs
import inspect
import os
import urllib3
import json
import base64
from urllib.parse import unquote, urlsplit, urlunsplit
from urllib.request import getproxies_environment, proxy_bypass_environment

from .._multipart import Multipart
from ._http import TRANSPORT_ERRORS
from ._constants import (
    DEFAULT_TIMEOUT,
    DEFAULT_RETRIES,
    DEFAULT_IDEMPOTENT_RETRIES,
)


def _proxy_from_env() -> "dict | None":
    """Read a proxy URL from the standard HTTPS_PROXY / HTTP_PROXY env vars."""
    url = (os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
           or os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy"))
    return {"url": url} if url else None


def _raise_if_error(event, kind: str = "") -> None:
    """Raise the typed error a stream reports mid-way — Anthropic's ``event:
    error`` (``overloaded_error`` is its 529), an OpenAI-compatible ``{"error"}``
    — so a reply cut short there is not taken for the final answer."""
    if not isinstance(event, dict):
        return
    error = event.get("error")
    if kind != "error" and event.get("type") != "error" and not error:
        return
    error = error if isinstance(error, dict) else {"message": str(error or event)}
    code = error.get("code") or error.get("status")
    status = code if isinstance(code, int) else 529 if error.get("type") == "overloaded_error" else 500
    raise error_from_status(status, error.get("message") or json.dumps(error))


def _clean_key(key):
    """The key stripped (``open().read()`` keeps a newline) and refused, never
    quoted, if a control character remains: the header library would put the
    whole ``Bearer sk-…`` in its error, and errors reach logs and run records."""
    if not isinstance(key, str):
        return key
    key = key.strip()
    if any(ord(c) < 32 or ord(c) == 127 for c in key):
        raise ValueError("the API key contains a control character, so no header can carry "
                         "it; check how it was read — the value is not shown")
    return key


def make_http(proxy: "dict | None" = None, *, timeout=DEFAULT_TIMEOUT, retries=DEFAULT_RETRIES):
    """
    Build the urllib3 manager used for all HTTP in the library.

    Shared by ``BaseClient`` (model APIs) and every tool client, so a single
    proxy configuration applies everywhere. An explicit *proxy* dict wins;
    otherwise the standard ``HTTPS_PROXY`` / ``HTTP_PROXY`` environment
    variables are honoured (urllib3 does not read them on its own).

    *proxy* shape::

        {"url": "http://host:3128", "username": "u", "password": "p"}

    Basic ``Proxy-Authorization`` is added when username + password are given,
    or when the URL carries them — ``http://user:pass@host:3128``, which is how
    ``HTTPS_PROXY`` usually holds credentials. urllib3 does not read the userinfo
    of a proxy URL, so without this every such proxy answers 407.
    """
    if proxy is None:
        if not _proxy_from_env():
            return urllib3.PoolManager(timeout=timeout, retries=retries)
        return _EnvironmentRouted(timeout=timeout, retries=retries)
    return _proxy_manager(proxy, timeout=timeout, retries=retries)


class _EnvironmentRouted:
    """A proxy chosen per request from the environment variables: ``NO_PROXY``
    exempts local and internal hosts, and each scheme gets its own proxy (an
    https proxy on an ``http://`` server could read the key)."""

    def __init__(self, *, timeout, retries) -> None:
        self._kw = {"timeout": timeout, "retries": retries}
        self._direct = urllib3.PoolManager(**self._kw)
        self._through: dict = {}

    def _for(self, url: str):
        parts = urlsplit(url)
        if proxy_bypass_environment(parts.hostname or ""):
            return self._direct
        proxy_url = getproxies_environment().get(parts.scheme)
        if not proxy_url:
            return self._direct
        if proxy_url not in self._through:
            self._through[proxy_url] = _proxy_manager({"url": proxy_url}, **self._kw)
        return self._through[proxy_url]

    @property
    def connection_pool_kw(self) -> dict:
        """The settings every route shares — timeout and retries."""
        return self._direct.connection_pool_kw

    def request(self, method, url, **kwargs):
        return self._for(url).request(method, url, **kwargs)

    def urlopen(self, method, url, **kwargs):
        return self._for(url).urlopen(method, url, **kwargs)

    def clear(self) -> None:
        self._direct.clear()
        for manager in self._through.values():
            manager.clear()


def _proxy_manager(proxy: dict, *, timeout, retries):
    """A ProxyManager for *proxy*, with its credentials as a header."""
    proxy_headers: dict = {}
    url      = proxy["url"]
    username = proxy.get("username")
    password = proxy.get("password")
    parts = urlsplit(url)
    if parts.username is not None:
        if not username:
            username, password = unquote(parts.username), unquote(parts.password or "")
        host = parts.hostname or ""
        if ":" in host:                                   # IPv6 literal
            host = f"[{host}]"
        if parts.port:
            host = f"{host}:{parts.port}"
        url = urlunsplit((parts.scheme, host, parts.path, parts.query, parts.fragment))
    if username and password:
        encoded = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("utf-8")
        proxy_headers["Proxy-Authorization"] = f"Basic {encoded}"
    return urllib3.ProxyManager(
        proxy_url=url,
        proxy_headers=proxy_headers,
        timeout=timeout,
        retries=retries,
    )

# APIError and its subclasses live in ._errors; re-exported here so the
# long-standing ``from ._base import APIError`` imports keep working.
from ._http import request
from ._errors import (  # noqa: F401
    APIError,
    NetworkError,
    RateLimitError,
    AuthenticationError,
    InsufficientCreditsError,
    InvalidRequestError,
    NotFoundError,
    ServerError,
    TaskFailedError,
    error_from_status,
)


class BaseClient:
    """
    Provider-agnostic HTTP transport for AI provider APIs.

    Wraps urllib3 and provides:
      - Optional proxy support with Basic auth
      - JSON POST / multipart POST / binary GET helpers
      - Abstract ``_auth_headers()`` and ``list_models()`` that every
        subclass must implement
      - Concrete ``check_auth()`` that delegates to ``list_models()``

    Each subclass declares a ``BASE_URL`` class constant.  The caller may
    override it per-instance via the ``url`` parameter (useful for
    enterprise gateways, Azure OpenAI, etc.).

    Parameters
    ----------
    api_key : str
        Secret key used to authenticate against the provider.
    url : str | None, optional
        Override the provider base URL.  Defaults to ``cls.BASE_URL``.
        Trailing slash is stripped automatically.
    timeout : urllib3.Timeout, optional
        Connect + read timeout.  Defaults to ``DEFAULT_TIMEOUT``.
    retries : urllib3.Retry, optional
        Retry policy.  Defaults to ``DEFAULT_RETRIES``.
    proxy : dict | None, optional
        Route all traffic through this proxy.

        Expected shape::

            {
                "url":      "http://proxy.host:3128",  # required
                "username": "user",                    # optional
                "password": "secret",                  # optional
            }

        When both ``username`` and ``password`` are present the
        ``Proxy-Authorization: Basic …`` header is added automatically.
    """

    # Subclasses must override this.
    BASE_URL: str = ""
    #: Whether this family sends function-calling schemas natively. A model
    #: given tools on a family that says False is refused before the request.
    supports_tools: bool = False

    def __init_subclass__(cls, **kwargs) -> None:
        # A family that overrides `build_request` takes `tools`, whether it
        # uses them or not: an override that drops the parameter while
        # inheriting `supports_tools = True` loses tools without a word.
        # Caught when the class is defined, not on a call.
        super().__init_subclass__(**kwargs)
        own = cls.__dict__.get("build_request")
        if own is not None and "tools" not in inspect.signature(own).parameters:
            raise TypeError(f"{cls.__name__}.build_request must accept tools=None, "
                            "as BaseClient.build_request declares")

    def __init__(
        self,
        api_key: str,
        url: str | None = None,
        timeout: urllib3.Timeout = DEFAULT_TIMEOUT,
        retries: urllib3.Retry = DEFAULT_RETRIES,
        proxy: dict | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = (url or self.BASE_URL).rstrip("/")

        # An explicit proxy wins; otherwise HTTPS_PROXY / HTTP_PROXY env vars
        # are honoured (shared with every tool client via make_http).
        self._http = make_http(proxy, timeout=timeout, retries=retries)

    # ------------------------------------------------------------------
    # Authentication — must be overridden by every provider subclass
    # ------------------------------------------------------------------

    #: Set by `Model` when the caller passed a callable `api_key`. Asked
    #: once per request rather than once per Model, which is what makes one
    #: Model serve many tenants.
    _resolve_key = None

    @property
    def api_key(self) -> str:
        """The key for the request being built.

        A constant unless a resolver was given, in which case the current
        `RunContext` decides. Failing loudly on an empty answer is the point:
        falling back to the process-wide key would send one tenant's request
        under another's credential, and the provider would answer normally.
        """
        if self._resolve_key is None:
            return _clean_key(self._api_key)
        from ..state import current
        context = current()
        key = _clean_key(self._resolve_key(context))
        if not key:
            tenant = getattr(context, "tenant", None)
            raise ValueError(
                f"no API key resolved for tenant {tenant!r}. The resolver "
                "passed to Model(api_key=...) returned nothing; there is no "
                "process-wide key to fall back on, because falling back would "
                "bill one tenant to another and the provider would answer "
                "normally.")
        return key

    def _auth_headers(self) -> dict:
        """Headers from the provider's ``auth``: ``bearer``, a header name, or
        ``none`` (a bearer only when there is a key — some servers 401 on an
        empty one). The key is read once: it may be a vault lookup."""
        key = self.api_key
        scheme = ((getattr(self, "_data", None) or {}).get("provider") or {}).get("auth", "bearer")
        headers = {"Content-Type": "application/json"}
        if key:
            if scheme in ("bearer", "none"):
                headers["Authorization"] = f"Bearer {key}"
            else:
                headers[scheme] = key
        return headers

    # ------------------------------------------------------------------
    # Format — must be overridden by each API-family client
    # ------------------------------------------------------------------

    def build_request(
        self, messages: list, output: dict, params: dict, tools: list | None = None
    ) -> "tuple[str, dict | Multipart]":
        """
        Translate our universal *messages* + *output* spec + model *params*
        into the provider's native ``(path, body)`` pair — a JSON object, or
        a :class:`~yait_aichain._multipart.Multipart` for a file upload.
        *tools* are function-calling schemas, given only when
        ``supports_tools`` is True.

        ``params`` carries the model-level settings the family needs:
        ``{name, temperature, max_tokens, top_p, top_k, reasoning}``.  The
        client is stateless about the model — everything comes in here.

        Abstract: each family client (openai / anthropic / google / …)
        implements its own wire format.  Not a passthrough — an unimplemented
        family raises here rather than sending our format and getting a 400.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement build_request()"
        )

    def parse_response(self, response: dict, output: dict) -> "str | dict":
        """
        Translate the provider's raw response into our clean result
        (str for text, dict for json / image).  Abstract — see build_request.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement parse_response()"
        )

    # ------------------------------------------------------------------
    # Request lifecycle seam
    # ------------------------------------------------------------------

    def send(self, path: str, body: dict, headers: dict) -> bytes:
        """
        Execute one logical completion and return the raw response bytes that
        ``parse_response`` will consume.

        The default is a single JSON POST — the right behaviour for every
        synchronous API.  A provider whose endpoint is *asynchronous* (submit a
        job, poll for completion, then download the artefact) overrides this to
        run that multi-step flow and synthesise a response in the same shape the
        synchronous path would have returned, so ``parse_response`` stays
        unchanged.  See ``QwenClient`` for the DashScope image-synthesis case.
        """
        return self._post(path, body, headers)

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------
    #
    # Three seams, and only the middle one is per-family. The transport below
    # is the same server-sent-events reader for everybody; what differs is the
    # flag that turns streaming on in the body and the shape of one delta.
    #
    # A family that does not override these is not broken — it is a provider
    # that cannot stream, and the caller is told so rather than left waiting
    # for chunks that never come. See `Model.stream`.

    #: Whether this family can deliver an answer progressively. False here so
    #: a new family is honest by default: a provider is declared able to
    #: stream by someone who implemented and tested it, not by inheritance.
    supports_streaming: bool = False

    def build_stream_request(self, messages: list, output: dict, params: dict,
                             tools: "list | None" = None) -> "tuple[str, dict]":
        """``(path, body)`` for a streaming call — the ordinary request plus
        whatever this provider's word for "stream" is."""
        raise NotImplementedError(
            f"{type(self).__name__} cannot stream")

    def parse_stream_event(self, event: dict, output: dict) -> "str | None":
        """The text carried by one event, or ``None`` for an event that
        carries none — a role announcement, a heartbeat, a usage report."""
        raise NotImplementedError(
            f"{type(self).__name__} cannot stream")

    def stream_usage(self, event: dict) -> "dict | None":
        """
        The usage carried by one event, **shaped like a response**.

        The envelope rather than the inner block, so that `extract_usage`
        reads a streamed report through exactly the same branch as a
        buffered one: `{"usage": {...}}` for the OpenAI and Anthropic
        shapes, `{"usageMetadata": {...}}` for Google. Returning the bare
        block instead would work for two families and silently produce a
        zero for the third, which is the kind of asymmetry a "universal"
        layer exists to not have.

        None throughout when the provider reports nothing. That is honest:
        counting the text we happened to see would be a number with no
        provider behind it.
        """
        return None

    def stream_tool_fragments(self, event: dict) -> "list | None":
        """
        Fragments of tool calls carried by one event.

        Each fragment is ``{"slot", "id", "name", "arguments"}``, and any of
        the last three may be absent — that is the whole difficulty. A call
        does not arrive as a call: an id and a name land once, then the
        arguments trickle in as pieces of a JSON **string** across many
        events, and two calls in one turn interleave.

        ``slot`` is what keeps them apart. It is the provider's own index for
        the call, never the order the fragments arrived in: OpenAI numbers
        them and Anthropic uses its content-block index, precisely because
        arrival order does not identify anything once there is more than one
        call. Keying on arrival order splices two calls' arguments into one
        unparseable string, and the recovery from unparseable arguments is an
        empty dict — so the tool runs, with nothing, and the run continues.

        Returning None means "this family does not stream tool calls", which
        is different from "this event carried none" (an empty list).
        """
        return None

    def _post_sse(self, path: str, data: dict, headers: dict):
        """POST *data* and yield each server-sent event decoded. Unbuffered
        (``preload_content=False``), an event ends at a blank line and may span
        several ``data:`` lines, and ``[DONE]`` is a sentinel, not JSON."""
        body = json.dumps(data).encode("utf-8")       # a bug here is a bug, not the network
        try:
            response = self._http.request(
                "POST",
                self._base_url + path,
                body=body,
                headers={**headers, "Accept": "text/event-stream"},
                preload_content=False,
            )
        except TRANSPORT_ERRORS as exc:
            raise NetworkError(0, str(exc)) from exc

        if not (200 <= response.status < 300):
            # `.data`, not `.read()`: the retry policy may already have read
            # the body to look for billing, and `.data` is what keeps it.
            body = response.data.decode("utf-8", errors="replace")
            response.release_conn()
            raise error_from_status(response.status, body, response.headers)

        # A 200 that is not a stream — a proxy, a server ignoring `stream` —
        # is one answer, read as one event, where it gave an empty result.
        if "application/json" in (response.headers.get("content-type") or "").lower():
            whole = response.data
            response.release_conn()
            try:
                event = json.loads(whole) if whole.strip() else None
            except json.JSONDecodeError:
                event = None
            if event is not None:
                _raise_if_error(event)
                yield event
            return

        finished = False
        try:
            # Decoded incrementally: a non-ASCII character split across two
            # chunks would decode as two replacement marks. Lines split on "\n"
            # alone: U+2028, which `splitlines` also splits on, is legal inside
            # a JSON string.
            decoder = codecs.getincrementaldecoder("utf-8")("replace")
            buffer = ""
            for raw in response.stream(8192, decode_content=True):
                # The whole buffer: a "\r" and its "\n" can arrive apart.
                buffer = (buffer + decoder.decode(raw)).replace("\r\n", "\n")
                while "\n\n" in buffer:
                    chunk, buffer = buffer.split("\n\n", 1)
                    lines = chunk.split("\n")
                    kind = next((ln[6:].strip() for ln in lines if ln.startswith("event:")), "")
                    payload = "\n".join(ln[5:].lstrip() for ln in lines if ln.startswith("data:"))
                    if not payload or payload == "[DONE]":
                        continue
                    try:
                        event = json.loads(payload)
                    except json.JSONDecodeError:
                        # A provider that sends a non-JSON comment or keeps
                        # the connection warm with junk must not end the run.
                        continue
                    _raise_if_error(event, kind)
                    yield event
            finished = True
        except TRANSPORT_ERRORS as exc:
            # A read that timed out or a connection cut mid-stream is the
            # network, retried as such — not a raw urllib3 error past APIError.
            raise NetworkError(0, f"the stream broke off: {exc}") from exc
        finally:
            if finished:
                response.release_conn()
            else:
                # Abandoned or broken: closed, so the provider stops
                # generating — and billing — for a reader that has gone.
                getattr(response, "close", response.release_conn)()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def list_models(self) -> list[str]:
        """
        Return the list of model IDs available for the configured API key.

        Must be implemented by every subclass.

        Returns
        -------
        list[str]
            Sorted or provider-ordered list of model identifier strings.

        Raises
        ------
        APIError
            On HTTP error or network failure (including auth errors).
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement list_models()"
        )

    @staticmethod
    def _poll_timeout(deadline: float):
        """One poll's own timeout: at most 30 seconds, and never past the
        deadline. The deadline bounded the loop, not a request inside it — a
        status request that hung waited the client's whole read timeout."""
        import time
        remaining = max(1.0, deadline - time.monotonic())
        return urllib3.Timeout(connect=min(10.0, remaining), read=min(30.0, remaining))

    def check_auth(self) -> bool:
        """
        Verify that the configured API key is accepted by the provider.

        Calls ``list_models()`` and returns ``True`` on success.
        Returns ``False`` when the key is refused (401, 403, or 400 where a
        provider answers a bad key so). A rate limit, an empty balance, a
        missing endpoint, a server error and a network error are raised: none
        of them says the key is bad, and reading them as False sent people to
        rotate a key that worked.

        Returns
        -------
        bool
            ``True`` if the key is valid, ``False`` if it is rejected.
        """
        try:
            self.list_models()
            return True
        except (RateLimitError, InsufficientCreditsError, NotFoundError):
            raise
        except APIError as exc:
            if exc.status in (400, 401, 403):
                return False
            raise

    # ------------------------------------------------------------------
    # Protected HTTP primitives
    # ------------------------------------------------------------------

    def _get(self, path: str, headers: dict | None = None, timeout=None) -> bytes:
        """GET ``{base_url}{path}``: the body on a 2xx, a typed `APIError` otherwise."""
        extra = {"timeout": timeout} if timeout is not None else {}
        return request(self._http, "GET", self._base_url + path, headers=headers,
                       retries=DEFAULT_IDEMPOTENT_RETRIES, **extra).data

    def _post(self, path: str, data: dict, headers: dict) -> bytes:
        """POST *data* as JSON to ``{base_url}{path}``: the body, or a typed `APIError`."""
        return request(self._http, "POST", self._base_url + path, json=data, headers=headers).data

    def _post_form(self, path: str, fields: dict, headers: dict) -> bytes:
        """POST multipart *fields* to ``{base_url}{path}``. urllib3 sets the
        content type with its boundary, so *headers* must not carry one."""
        return request(self._http, "POST", self._base_url + path, fields=fields, headers=headers).data

    def _download(self, url: str, headers: dict | None = None, timeout=None) -> dict:
        """GET an absolute URL: ``{"data": bytes, "media_type": str}``."""
        extra = {"timeout": timeout} if timeout is not None else {}
        response = request(self._http, "GET", url, headers=headers, **extra)
        return {"data": response.data,
                "media_type": response.headers.get("Content-Type", "application/octet-stream")}
