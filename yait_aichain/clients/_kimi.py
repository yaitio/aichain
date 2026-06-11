import json
import urllib3

from ._base import BaseClient, APIError
from ._constants import DEFAULT_TIMEOUT, DEFAULT_RETRIES


class KimiClient(BaseClient):
    """
    HTTP transport client for the Kimi (Moonshot AI) API.

    Handles authentication and exposes provider-level utility methods.
    Model selection, sampling parameters, and task-specific logic
    belong to the model/task layer above this class.

    The Kimi API is OpenAI-compatible — request/response shapes mirror
    OpenAI's Chat Completions endpoint; only the base URL, model roster,
    and the ``thinking`` parameter differ.

    Base URL: ``https://api.moonshot.ai``
    Auth:     ``Authorization: Bearer <api_key>``

    Parameters
    ----------
    api_key : str
        Moonshot API key (``MOONSHOT_API_KEY``).
    url : str | None, optional
        Override the default base URL.
    timeout : urllib3.Timeout, optional
        Network timeout.  Defaults to DEFAULT_TIMEOUT.
    retries : urllib3.Retry, optional
        Retry policy.  Defaults to DEFAULT_RETRIES.
    proxy : dict | None, optional
        Optional proxy configuration (see BaseClient).
    """

    BASE_URL = "https://api.moonshot.ai"

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
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def list_models(self) -> list[str]:
        """
        Return all model IDs available for the configured API key.

        Calls ``GET /v1/models`` (OpenAI-compatible endpoint).

        Returns
        -------
        list[str]
            Model identifier strings, e.g. ``["kimi-k2.5", "kimi-k2-thinking", …]``.

        Raises
        ------
        APIError
            On HTTP error or network failure (401 for invalid key).
        """
        data = self._get("/v1/models", self._auth_headers())
        return [m["id"] for m in json.loads(data)["data"]]
