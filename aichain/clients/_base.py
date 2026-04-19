import urllib3
import json
import base64

from ._constants import DEFAULT_TIMEOUT, DEFAULT_RETRIES


class APIError(Exception):
    """
    Raised when an AI provider API returns a non-2xx HTTP status,
    or when a network-level error occurs.

    Attributes
    ----------
    status  : HTTP status code (0 for network/connection errors).
    message : Error detail from the response body or exception message.
    """

    def __init__(self, status: int, message: str) -> None:
        self.status = status
        self.message = message
        super().__init__(f"[HTTP {status}] {message}")


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

        if proxy is None:
            self._http = urllib3.PoolManager(timeout=timeout, retries=retries)
        else:
            proxy_headers: dict = {}
            username = proxy.get("username")
            password = proxy.get("password")
            if username and password:
                encoded = base64.b64encode(
                    f"{username}:{password}".encode("utf-8")
                ).decode("utf-8")
                proxy_headers["Proxy-Authorization"] = f"Basic {encoded}"

            self._http = urllib3.ProxyManager(
                proxy_url=proxy["url"],
                proxy_headers=proxy_headers,
                timeout=timeout,
                retries=retries,
            )

    # ------------------------------------------------------------------
    # Authentication — must be overridden by every provider subclass
    # ------------------------------------------------------------------

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
                "GET", self._base_url + path, headers=headers
            )
        except Exception as exc:
            raise APIError(0, str(exc)) from exc

        if 200 <= response.status < 300:
            return response.data

        raise APIError(
            response.status,
            response.data.decode("utf-8", errors="replace"),
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
            raise APIError(0, str(exc)) from exc

        if 200 <= response.status < 300:
            return response.data

        raise APIError(
            response.status,
            response.data.decode("utf-8", errors="replace"),
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
            raise APIError(0, str(exc)) from exc

        if 200 <= response.status < 300:
            return response.data

        raise APIError(
            response.status,
            response.data.decode("utf-8", errors="replace"),
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
            raise APIError(0, str(exc)) from exc

        if 200 <= response.status < 300:
            return {
                "data": response.data,
                "media_type": response.headers.get(
                    "Content-Type", "application/octet-stream"
                ),
            }

        raise APIError(
            response.status,
            response.data.decode("utf-8", errors="replace"),
        )
