import json
import urllib3

from ._base import BaseClient
from ._constants import DEFAULT_TIMEOUT, DEFAULT_RETRIES


class GoogleAIClient(BaseClient):
    """
    HTTP transport client for the Google AI (Gemini) API.

    Handles authentication and exposes provider-level utility methods.
    Model selection, sampling parameters, and task-specific logic
    (generateContent, embedContent, etc.) belong to the model/task
    layer above this class.

    Base URL: ``https://generativelanguage.googleapis.com/v1beta``
    Auth:     API key passed in the ``x-goog-api-key`` header (keeps the
              key out of URLs, which leak into proxy/server logs).

    Parameters
    ----------
    api_key : str
        Google AI API key (``AIza...``).
    url : str | None, optional
        Override the default base URL.
    timeout : urllib3.Timeout, optional
        Network timeout.  Defaults to DEFAULT_TIMEOUT.
    retries : urllib3.Retry, optional
        Retry policy.  Defaults to DEFAULT_RETRIES.
    proxy : dict | None, optional
        Optional proxy configuration (see BaseClient).
    """

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

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
        # The x-goog-api-key header keeps the key out of URLs, which end up
        # in proxy/server logs and tracebacks (query-string auth does not).
        return {
            "Content-Type":   "application/json",
            "x-goog-api-key": self._api_key,
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def list_models(self) -> list[str]:
        """
        Return all model IDs available for the configured API key.

        Calls ``GET /models`` (key sent via the ``x-goog-api-key`` header).

        The raw Google response returns names in the form
        ``"models/gemini-2.0-flash"``; this method strips the
        ``"models/"`` prefix and returns just the identifier.

        Returns
        -------
        list[str]
            Model identifier strings, e.g. ``["gemini-2.0-flash", …]``.

        Raises
        ------
        APIError
            On HTTP error or network failure (400/403 for invalid key).
        """
        path = "/models"
        data = self._get(path, self._auth_headers())
        return [
            m["name"].removeprefix("models/")
            for m in json.loads(data)["models"]
        ]
