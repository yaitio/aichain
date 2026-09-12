import os
import urllib3
import json
import base64

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


def make_http(proxy: "dict | None" = None, *, timeout=DEFAULT_TIMEOUT, retries=DEFAULT_RETRIES):
    """
    Build the urllib3 manager used for all HTTP in the library.

    Shared by ``BaseClient`` (model APIs) and every tool client, so a single
    proxy configuration applies everywhere. An explicit *proxy* dict wins;
    otherwise the standard ``HTTPS_PROXY`` / ``HTTP_PROXY`` environment
    variables are honoured (urllib3 does not read them on its own).

    *proxy* shape::

        {"url": "http://host:3128", "username": "u", "password": "p"}

    Basic ``Proxy-Authorization`` is added when username + password are given.
    """
    if proxy is None:
        proxy = _proxy_from_env()
    if not proxy:
        return urllib3.PoolManager(timeout=timeout, retries=retries)

    proxy_headers: dict = {}
    username = proxy.get("username")
    password = proxy.get("password")
    if username and password:
        encoded = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("utf-8")
        proxy_headers["Proxy-Authorization"] = f"Basic {encoded}"
    return urllib3.ProxyManager(
        proxy_url=proxy["url"],
        proxy_headers=proxy_headers,
        timeout=timeout,
        retries=retries,
    )

# APIError and its subclasses live in ._errors; re-exported here so the
# long-standing ``from ._base import APIError`` imports keep working.
from ._errors import (  # noqa: F401
    APIError,
    NetworkError,
    RateLimitError,
    AuthenticationError,
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
            return self._api_key
        from ..state import current
        context = current()
        key = self._resolve_key(context)
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
        """
        Return the provider-specific HTTP headers needed for authentication
        (e.g. ``Authorization``, ``x-api-key``, versioning headers, etc.).

        Must be implemented by every subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _auth_headers()"
        )

    # ------------------------------------------------------------------
    # Format — must be overridden by each API-family client
    # ------------------------------------------------------------------

    def build_request(
        self, messages: list, output: dict, params: dict
    ) -> "tuple[str, dict]":
        """
        Translate our universal *messages* + *output* spec + model *params*
        into the provider's native ``(path, body)`` pair.

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
        """
        POST *data* and yield each server-sent event as a decoded object.

        Deliberately not a generic line reader. Three details are what make
        the difference between this working and appearing to work:

        * ``preload_content=False`` — without it urllib3 buffers the whole
          response and every chunk arrives at once, at the end. The code
          reads as streaming and the user sees none of it.
        * an event is terminated by a blank line and its data may span
          several ``data:`` lines, which have to be joined before decoding.
        * ``[DONE]`` is a sentinel, not JSON. Feeding it to a decoder is the
          commonest way an SSE reader ends its stream with an exception
          instead of a return.
        """
        try:
            response = self._http.request(
                "POST",
                self._base_url + path,
                body=json.dumps(data).encode("utf-8"),
                headers={**headers, "Accept": "text/event-stream"},
                preload_content=False,
            )
        except Exception as exc:
            raise NetworkError(0, str(exc)) from exc

        if not (200 <= response.status < 300):
            body = response.read().decode("utf-8", errors="replace")
            response.release_conn()
            raise error_from_status(response.status, body, response.headers)

        try:
            buffer = ""
            for raw in response.stream(amt=None, decode_content=True):
                buffer += raw.decode("utf-8", errors="replace")
                while "\n\n" in buffer or "\r\n\r\n" in buffer:
                    sep = "\r\n\r\n" if "\r\n\r\n" in buffer and (
                        "\n\n" not in buffer
                        or buffer.index("\r\n\r\n") < buffer.index("\n\n")
                    ) else "\n\n"
                    chunk, buffer = buffer.split(sep, 1)
                    payload = "".join(
                        line[5:].lstrip() if line.startswith("data:") else ""
                        for line in chunk.splitlines()
                        if line.startswith("data:"))
                    if not payload or payload == "[DONE]":
                        continue
                    try:
                        yield json.loads(payload)
                    except json.JSONDecodeError:
                        # A provider that sends a non-JSON comment or keeps
                        # the connection warm with junk must not end the run.
                        continue
        finally:
            response.release_conn()

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

    def check_auth(self) -> bool:
        """
        Verify that the configured API key is accepted by the provider.

        Calls ``list_models()`` and returns ``True`` on success.
        Returns ``False`` for any 4xx response (invalid or revoked key).
        Re-raises ``APIError`` for 5xx / network errors so callers can
        distinguish a bad key from a provider outage.

        Returns
        -------
        bool
            ``True`` if the key is valid, ``False`` if it is rejected.
        """
        try:
            self.list_models()
            return True
        except APIError as exc:
            if 400 <= exc.status < 500:
                return False
            raise

    # ------------------------------------------------------------------
    # Protected HTTP primitives
    # ------------------------------------------------------------------

    def _get(self, path: str, headers: dict | None = None) -> bytes:
        """
        Send a GET request to ``{base_url}{path}``.

        Parameters
        ----------
        path    : URL path appended to base_url.
        headers : Optional HTTP headers.

        Returns
        -------
        bytes
            Raw response body on a 2xx status code.

        Raises
        ------
        APIError
            On any non-2xx status code or network failure.
        """
        try:
            response = self._http.request(
                "GET",
                self._base_url + path,
                headers=headers,
                retries=DEFAULT_IDEMPOTENT_RETRIES,
            )
        except Exception as exc:
            raise NetworkError(0, str(exc)) from exc

        if 200 <= response.status < 300:
            return response.data

        raise error_from_status(
            response.status,
            response.data.decode("utf-8", errors="replace"),
            response.headers,
        )

    def _post(self, path: str, data: dict, headers: dict) -> bytes:
        """
        Send a JSON POST request to ``{base_url}{path}``.

        Parameters
        ----------
        path    : URL path appended to base_url.
        data    : Payload to be JSON-encoded.
        headers : HTTP headers (must include ``Content-Type: application/json``).

        Returns
        -------
        bytes
            Raw response body on a 2xx status code.

        Raises
        ------
        APIError
            On any non-2xx status code or network failure.
        """
        try:
            response = self._http.request(
                "POST",
                self._base_url + path,
                body=json.dumps(data).encode("utf-8"),
                headers=headers,
            )
        except Exception as exc:
            raise NetworkError(0, str(exc)) from exc

        if 200 <= response.status < 300:
            return response.data

        raise error_from_status(
            response.status,
            response.data.decode("utf-8", errors="replace"),
            response.headers,
        )

    def _post_form(self, path: str, fields: dict, headers: dict) -> bytes:
        """
        Send a multipart/form-data POST request to ``{base_url}{path}``.

        urllib3 sets ``Content-Type: multipart/form-data`` automatically
        when ``fields`` is provided — do **not** include it in ``headers``.

        Parameters
        ----------
        path    : URL path appended to base_url.
        fields  : Form fields (strings or file tuples).
        headers : HTTP headers (excluding Content-Type).

        Returns
        -------
        bytes
            Raw response body on a 2xx status code.

        Raises
        ------
        APIError
            On any non-2xx status code or network failure.
        """
        try:
            response = self._http.request(
                "POST",
                self._base_url + path,
                fields=fields,
                headers=headers,
            )
        except Exception as exc:
            raise NetworkError(0, str(exc)) from exc

        if 200 <= response.status < 300:
            return response.data

        raise error_from_status(
            response.status,
            response.data.decode("utf-8", errors="replace"),
            response.headers,
        )

    def _download(self, url: str, headers: dict | None = None) -> dict:
        """
        Download binary content from an absolute URL (not base_url-relative).

        Parameters
        ----------
        url     : Fully-qualified URL to download.
        headers : Optional HTTP headers.

        Returns
        -------
        dict
            ``{"data": bytes, "media_type": str}``

        Raises
        ------
        APIError
            On any non-2xx status code or network failure.
        """
        try:
            response = self._http.request("GET", url, headers=headers)
        except Exception as exc:
            raise NetworkError(0, str(exc)) from exc

        if 200 <= response.status < 300:
            return {
                "data": response.data,
                "media_type": response.headers.get(
                    "Content-Type", "application/octet-stream"
                ),
            }

        raise error_from_status(
            response.status,
            response.data.decode("utf-8", errors="replace"),
            response.headers,
        )
