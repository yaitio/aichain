"""
tools.imgbb — ImgbbUploadTool
==============================

Upload a base64-encoded image to `Imgbb <https://api.imgbb.com/>`_ and
receive a permanent public HTTPS URL.

Imgbb offers a free image-hosting API that requires no server, no S3 bucket,
and no OAuth flow — just a single API key.  It is the standard bridge step
whenever a pipeline produces a base64 image (e.g. from ``gpt-image-1`` or
``DALL-E``) but the next step (e.g. Late social publishing) needs a URL.

API reference
-------------
``POST https://api.imgbb.com/1/upload``

Fields (multipart/form-data)
  ``key``        — API key (required)
  ``image``      — base64-encoded image data, without the ``data:…;base64,``
                   URI prefix (required)
  ``name``       — optional filename (no extension required)
  ``expiration`` — optional TTL in seconds (60 – 15 552 000)

Response (JSON)
  ``data.url``           — direct image URL (permanent)
  ``data.display_url``   — viewer page URL
  ``data.delete_url``    — single-use deletion URL

Environment variable
--------------------
``IMGBB_API_KEY``  — your API key from https://api.imgbb.com/

Get a free key
--------------
Sign up at https://api.imgbb.com/ → "API" → copy the key.
"""

from __future__ import annotations

import json
import os
import urllib.parse

import urllib3

from ._base import Tool


# ── Constants ─────────────────────────────────────────────────────────────────

_UPLOAD_URL = "https://api.imgbb.com/1/upload"
_ENV_KEY     = "IMGBB_API_KEY"


# ── ImgbbUploadTool ───────────────────────────────────────────────────────────

class ImgbbUploadTool(Tool):
    """
    Upload a base64 image to Imgbb and return a public HTTPS URL.

    This tool is the "bridge" step in any pipeline that generates images
    (e.g. via ``gpt-image-1`` or ``DALL-E``) and then needs to attach them
    to a social media post or share them publicly.  Imgbb stores the image
    permanently and returns a stable URL.

    Parameters
    ----------
    api_key : str | None, optional
        Imgbb API key.  When omitted the ``IMGBB_API_KEY`` environment
        variable is used.

    ``run()`` options
    -----------------
    ``name`` : str
        Optional image filename (no extension required).
    ``expiration`` : int
        Image TTL in seconds (60 – 15 552 000).  Omit for permanent storage.

    Examples
    --------
    Basic upload::

        from tools import ImgbbUploadTool

        tool = ImgbbUploadTool()
        url  = tool.run(input=base64_string)
        print(url)   # https://i.ibb.co/...

    In a Chain (receiving base64 from a gpt-image-1 Skill)::

        (imgbb, "image_url", {"input": "base64"})
    """

    name        = "imgbb_upload"
    description = (
        "Upload a base64-encoded image to Imgbb and return a permanent "
        "public HTTPS URL.  Use this whenever a downstream step needs a "
        "URL rather than raw binary or base64 data."
    )
    parameters  = {
        "type": "object",
        "properties": {
            "input": {
                "type":        "string",
                "description": (
                    "Base64-encoded image data.  "
                    "A leading ``data:<mime>;base64,`` URI prefix is "
                    "stripped automatically if present."
                ),
            },
            "options": {
                "type":        "object",
                "description": "Upload options.",
                "properties": {
                    "name": {
                        "type":        "string",
                        "description": "Optional image name (no extension needed).",
                    },
                    "expiration": {
                        "type":        "integer",
                        "description": "Image TTL in seconds (60–15 552 000).",
                    },
                },
            },
        },
        "required": ["input"],
    }

    # ──────────────────────────────────────────────────────────────────────────

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or os.environ.get(_ENV_KEY)
        if not key:
            raise ValueError(
                f"No Imgbb API key found.  "
                f"Pass api_key= or set the {_ENV_KEY!r} environment variable.  "
                f"Get a free key at https://api.imgbb.com/"
            )
        self._api_key = key
        self._http    = urllib3.PoolManager()

    # ──────────────────────────────────────────────────────────────────────────

    def run(self, input: str, options: dict | None = None) -> str:
        """
        Upload *input* (base64 image) to Imgbb and return the public URL.

        Parameters
        ----------
        input : str
            Base64-encoded image string.  A ``data:<mime>;base64,`` prefix
            is stripped automatically if present.
        options : dict | None, optional
            ``name``       (str) — optional image filename.
            ``expiration`` (int) — TTL in seconds; omit for permanent storage.

        Returns
        -------
        str
            Permanent public HTTPS URL, e.g. ``"https://i.ibb.co/…/img.png"``.

        Raises
        ------
        ValueError
            When *input* is empty after prefix stripping.
        RuntimeError
            On any non-2xx HTTP response from the Imgbb API, or when the
            API reports ``success: false``.
        """
        opts = options or {}

        # ── Strip data URI prefix ─────────────────────────────────────────
        b64 = input.strip()
        if "," in b64:
            b64 = b64.split(",", 1)[1].strip()
        if not b64:
            raise ValueError("base64 image data is empty after stripping the data URI prefix.")

        # ── Build form fields ─────────────────────────────────────────────
        fields: dict[str, str] = {
            "key":   self._api_key,
            "image": b64,
        }
        if opts.get("name"):
            fields["name"] = str(opts["name"])
        if opts.get("expiration") is not None:
            fields["expiration"] = str(int(opts["expiration"]))

        # ── POST multipart/form-data ──────────────────────────────────────
        response = self._http.request(
            "POST",
            _UPLOAD_URL,
            fields = fields,
        )
        raw = response.data.decode("utf-8", errors="replace")

        if not (200 <= response.status < 300):
            raise RuntimeError(
                f"Imgbb API error [{response.status}]: {raw[:400]}"
            )

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Imgbb returned non-JSON response: {raw[:200]}"
            ) from exc

        if not data.get("success"):
            raise RuntimeError(
                f"Imgbb upload failed (success=false): {raw[:400]}"
            )

        url = data.get("data", {}).get("url", "")
        if not url:
            raise RuntimeError(
                f"Imgbb upload succeeded but returned no URL: {raw[:400]}"
            )

        return url

    def __repr__(self) -> str:
        return f"ImgbbUploadTool(name={self.name!r})"
