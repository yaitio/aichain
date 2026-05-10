import json
import urllib3

from ._base import BaseClient
from ._constants import DEFAULT_TIMEOUT, DEFAULT_RETRIES

# Required on every Anthropic request.
_API_VERSION = "2023-06-01"


class AnthropicClient(BaseClient):
    """
    HTTP transport client for the Anthropic Claude API.

    Handles authentication and exposes provider-level utility methods.
    Model selection, sampling parameters, and task-specific logic
    (messages, vision, tool use, etc.) belong to the model/task layer
    above this class.

    Base URL: ``https://api.anthropic.com``
    Auth:     ``x-api-key: <api_key>`` + ``anthropic-version`` header

    Parameters
    ----------
    api_key : str
        Anthropic secret key (``sk-ant-...``).
    url : str | None, optional
        Override the default base URL.
    timeout : urllib3.Timeout, optional
        Network timeout.  Defaults to DEFAULT_TIMEOUT.
    retries : urllib3.Retry, optional
        Retry policy.  Defaults to DEFAULT_RETRIES.
    proxy : dict | None, optional
        Optional proxy configuration (see BaseClient).
    """

    BASE_URL = "https://api.anthropic.com"

    def __init__(
        self,
        api_key: str,
        url: str | None = None,
        timeout: urllib3.Timeout = DEFAULT_TIMEOUT,
        retries: urllib3.Retry = DEFAULT_RETRIES,
        proxy: dict | None = None,
    ) -> None:
        super().__init__(api_key, url, timeout, retries, proxy)

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def _auth_headers(self) -> dict:
        return {
            "x-api-key": self._api_key,
            "Content-Type": "application/json",
            "anthropic-version": _API_VERSION,
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def list_models(self) -> list[str]:
        """
        Return all model IDs available for the configured API key.

        Calls ``GET /v1/models``.

        Returns
        -------
        list[str]
            Model identifier strings, e.g. ``["claude-opus-4-5", …]``.

        Raises
        ------
        APIError
            On HTTP error or network failure (401 for invalid key).
        """
        data = self._get("/v1/models", self._auth_headers())
        return [m["id"] for m in json.loads(data)["data"]]
