import json
import urllib3

from ._base import BaseClient, APIError
from ._constants import DEFAULT_TIMEOUT, DEFAULT_RETRIES


class DeepSeekClient(BaseClient):
    """
    HTTP transport client for the DeepSeek API.

    Handles authentication and exposes provider-level utility methods.
    Model selection, sampling parameters, and task-specific logic
    belong to the model/task layer above this class.

    The DeepSeek API is OpenAI-compatible — request/response shapes
    mirror OpenAI's Chat Completions endpoint.  The key difference is
    the ``deepseek-reasoner`` model, which returns an additional
    ``reasoning_content`` field alongside the standard ``content`` in
    assistant messages.

    Base URL: ``https://api.deepseek.com``
    Auth:     ``Authorization: Bearer <api_key>``

    Parameters
    ----------
    api_key : str
        DeepSeek API key (``DEEPSEEK_API_KEY``).
    url : str | None, optional
        Override the default base URL (e.g. for the beta endpoint
        ``https://api.deepseek.com/beta``).
    timeout : urllib3.Timeout, optional
        Network timeout.  Defaults to DEFAULT_TIMEOUT.
    retries : urllib3.Retry, optional
        Retry policy.  Defaults to DEFAULT_RETRIES.
    proxy : dict | None, optional
        Optional proxy configuration (see BaseClient).
    """

    BASE_URL = "https://api.deepseek.com"

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

        Calls ``GET /models`` (OpenAI-compatible endpoint).

        Returns
        -------
        list[str]
            Model identifier strings, e.g. ``["deepseek-chat", "deepseek-reasoner"]``.

        Raises
        ------
        APIError
            On HTTP error or network failure (401 for invalid key).
        """
        data = self._get("/models", self._auth_headers())
        return [m["id"] for m in json.loads(data)["data"]]
