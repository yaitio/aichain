import json
import urllib3

from ._base import BaseClient, APIError
from ._constants import DEFAULT_TIMEOUT, DEFAULT_RETRIES

# Perplexity does not expose a public /models endpoint.
# This list reflects the current model roster (March 2026).
_KNOWN_MODELS: list[str] = [
    "sonar",
    "sonar-pro",
    "sonar-reasoning",
    "sonar-reasoning-pro",
    "sonar-deep-research",
    "r1-1776",
]


class PerplexityClient(BaseClient):
    """
    HTTP transport client for the Perplexity AI API.

    Handles authentication and exposes provider-level utility methods.
    Model selection, sampling parameters, and task-specific logic
    (online chat completions with web search, etc.) belong to the
    model/task layer above this class.

    Perplexity exposes an OpenAI-compatible ``/chat/completions``
    endpoint but does **not** publish a ``/models`` discovery endpoint.
    ``list_models()`` therefore returns a curated static list.
    ``check_auth()`` is overridden to verify credentials via a minimal
    real API call instead.

    Base URL: ``https://api.perplexity.ai``
    Auth:     ``Authorization: Bearer <api_key>``

    Parameters
    ----------
    api_key : str
        Perplexity API key (``pplx-...``).
    url : str | None, optional
        Override the default base URL.
    timeout : urllib3.Timeout, optional
        Network timeout.  Defaults to DEFAULT_TIMEOUT.
    retries : urllib3.Retry, optional
        Retry policy.  Defaults to DEFAULT_RETRIES.
    proxy : dict | None, optional
        Optional proxy configuration (see BaseClient).
    """

    BASE_URL = "https://api.perplexity.ai"

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
        Return the known Perplexity model IDs.

        Perplexity does not publish a ``/models`` discovery endpoint, so
        this returns a curated static list of currently available models.

        Returns
        -------
        list[str]
            Known model identifier strings.
        """
        return list(_KNOWN_MODELS)

    def check_auth(self) -> bool:
        """
        Verify that the configured API key is accepted by Perplexity.

        Because Perplexity has no dedicated auth-check endpoint, this
        method makes a minimal ``/chat/completions`` request (1 token)
        and inspects the response.

        Returns ``False`` for 401/403 (bad or revoked key).
        Re-raises ``APIError`` for 5xx / network errors.

        Returns
        -------
        bool
            ``True`` if the key is valid, ``False`` if it is rejected.
        """
        body = {
            "model": _KNOWN_MODELS[0],
            "messages": [{"role": "user", "content": "1"}],
            "max_tokens": 1,
        }
        try:
            self._post("/chat/completions", body, self._auth_headers())
            return True
        except APIError as exc:
            if exc.status in (401, 403):
                return False
            raise
