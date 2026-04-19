"""
clients._qwen — QwenClient
===========================

HTTP transport client for the Alibaba DashScope / Qwen API.

The DashScope API is OpenAI-compatible on the ``/compatible-mode/v1/``
path and also exposes native endpoints for TTS and image generation on
``/api/v1/``.

Base URL varies by region:

  +-----------+-----------------------------------------------+
  | Region    | Base URL                                      |
  +===========+===============================================+
  | ap        | https://dashscope-intl.aliyuncs.com (default) |
  | us        | https://dashscope-us.aliyuncs.com             |
  | cn        | https://dashscope.aliyuncs.com                |
  | hk        | https://cn-hongkong.dashscope.aliyuncs.com    |
  +-----------+-----------------------------------------------+

The region is resolved (in priority order):

  1. The ``region`` constructor argument.
  2. The ``DASHSCOPE_REGION`` environment variable.
  3. The default: ``"ap"`` (Asia-Pacific / international endpoint).

Auth: ``Authorization: Bearer <api_key>``

Environment variables
----------------------
  DASHSCOPE_API_KEY   — API key (get one at https://dashscope.console.aliyun.com/)
  DASHSCOPE_REGION    — Optional region selector: ap | us | cn | hk
"""

from __future__ import annotations

import json
import os
import urllib3

from ._base import BaseClient
from ._constants import DEFAULT_TIMEOUT, DEFAULT_RETRIES


# ---------------------------------------------------------------------------
# Region → base URL mapping
# ---------------------------------------------------------------------------

REGION_URLS: dict[str, str] = {
    "ap": "https://dashscope-intl.aliyuncs.com",
    "us": "https://dashscope-us.aliyuncs.com",
    "cn": "https://dashscope.aliyuncs.com",
    "hk": "https://cn-hongkong.dashscope.aliyuncs.com",
}

_DEFAULT_REGION = "ap"


def resolve_qwen_base_url(region: "str | None" = None) -> str:
    """
    Return the DashScope base URL for *region*.

    Parameters
    ----------
    region : str | None
        One of ``"ap"``, ``"us"``, ``"cn"``, ``"hk"``.  When *None*,
        the ``DASHSCOPE_REGION`` env var is checked; falls back to ``"ap"``.

    Returns
    -------
    str
        Fully-qualified base URL (no trailing slash).

    Raises
    ------
    ValueError
        When an unrecognised region string is given.
    """
    r = (region or os.environ.get("DASHSCOPE_REGION") or _DEFAULT_REGION).lower().strip()
    if r not in REGION_URLS:
        raise ValueError(
            f"Unknown DashScope region {r!r}.  "
            f"Valid regions: {', '.join(sorted(REGION_URLS))}."
        )
    return REGION_URLS[r]


# ---------------------------------------------------------------------------
# QwenClient
# ---------------------------------------------------------------------------

class QwenClient(BaseClient):
    """
    HTTP transport client for the Alibaba DashScope / Qwen API.

    Parameters
    ----------
    api_key : str
        DashScope API key (``DASHSCOPE_API_KEY``).
    url : str | None, optional
        Override the region-derived base URL.  When set, *region* is ignored.
    region : str | None, optional
        One of ``"ap"`` (default), ``"us"``, ``"cn"``, ``"hk"``.
        Overrides the ``DASHSCOPE_REGION`` env var.
    timeout : urllib3.Timeout, optional
        Network timeout.  Defaults to DEFAULT_TIMEOUT.
    retries : urllib3.Retry, optional
        Retry policy.  Defaults to DEFAULT_RETRIES.
    proxy : dict | None, optional
        Optional proxy configuration (see BaseClient).
    """

    BASE_URL = REGION_URLS[_DEFAULT_REGION]

    def __init__(
        self,
        api_key: str,
        url:     "str | None"        = None,
        region:  "str | None"        = None,
        timeout: urllib3.Timeout     = DEFAULT_TIMEOUT,
        retries: urllib3.Retry       = DEFAULT_RETRIES,
        proxy:   "dict | None"       = None,
    ) -> None:
        base = url or resolve_qwen_base_url(region)
        super().__init__(api_key, base, timeout, retries, proxy)

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
        Return model IDs available on the compatible-mode endpoint.

        Calls ``GET /compatible-mode/v1/models``.

        Returns
        -------
        list[str]
            Model identifier strings.

        Raises
        ------
        APIError
            On HTTP error or network failure.
        """
        data = self._get("/compatible-mode/v1/models", self._auth_headers())
        return [m["id"] for m in json.loads(data)["data"]]
