"""
tools.services.social_post — serviceSocialPost
================================================

Merges LatePublishTool and LateAccountsTool into a single class with two
methods: ``run()`` to publish/schedule posts and ``accounts()`` to list
connected accounts.

Endpoint  : https://getlate.dev/api
Auth      : ``Authorization: Bearer <api_key>``
Docs      : https://docs.getlate.dev
Key       : https://getlate.dev/dashboard/api-keys

Typical agent workflow
----------------------
1. ``tool.accounts()``                → discover connected accounts + collect IDs
2. ``tool.run(content, options={...})`` → post with those account IDs

Environment variable
---------------------
``LATE_API_KEY`` — Bearer API key from https://getlate.dev/dashboard/api-keys
"""

from __future__ import annotations

import json
import mimetypes
import os
from typing import Any

import urllib3

from .._base import Tool

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_URL = "https://getlate.dev/api"
_ENV_KEY  = "LATE_API_KEY"

# Platforms supported by Late
_PLATFORMS = (
    "twitter", "instagram", "linkedin", "facebook", "tiktok",
    "youtube", "threads", "reddit", "pinterest", "bluesky",
    "telegram", "googlebusiness", "snapchat", "whatsapp",
)

_MEDIA_TYPE_MAP = {
    "image/jpeg":       "image",
    "image/jpg":        "image",
    "image/png":        "image",
    "image/webp":       "image",
    "image/gif":        "gif",
    "video/mp4":        "video",
    "video/mpeg":       "video",
    "video/quicktime":  "video",
    "video/avi":        "video",
    "video/x-msvideo":  "video",
    "video/webm":       "video",
    "video/x-m4v":      "video",
    "application/pdf":  "document",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_headers(api_key: str) -> dict:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
        "Accept":        "application/json",
    }


def _request(
    http:     urllib3.PoolManager,
    method:   str,
    path:     str,
    api_key:  str,
    body:     dict | None = None,
    params:   dict | None = None,
) -> dict:
    """Make an authenticated request and return the parsed JSON body."""
    url = f"{_BASE_URL}{path}"
    if params:
        # Encode non-None params as query string
        qs = "&".join(
            f"{k}={urllib3.request.urlencode(str(v))}"
            for k, v in params.items()
            if v is not None
        )
        if qs:
            url = f"{url}?{qs}"

    kwargs: dict[str, Any] = {"headers": _make_headers(api_key)}
    if body is not None:
        kwargs["body"] = json.dumps(body).encode("utf-8")

    response = http.request(method, url, **kwargs)
    raw = response.data.decode("utf-8", errors="replace")

    if not (200 <= response.status < 300):
        raise RuntimeError(
            f"Late API error [{response.status}] {path}: {raw[:500]}"
        )

    return json.loads(raw) if raw else {}


def _guess_media_type(url: str) -> str:
    """Infer the Late media type (image/video/gif/document) from a URL."""
    mime, _ = mimetypes.guess_type(url)
    if mime:
        return _MEDIA_TYPE_MAP.get(mime, "image")
    # Fallback: inspect extension
    low = url.lower().split("?")[0]
    if any(low.endswith(ext) for ext in (".mp4", ".mov", ".avi", ".webm", ".m4v")):
        return "video"
    if low.endswith(".gif"):
        return "gif"
    if low.endswith(".pdf"):
        return "document"
    return "image"


def _format_post_result(data: dict) -> str:
    """Render a create-post API response as a plain-text string."""
    post    = data.get("post", {})
    message = data.get("message", "")

    post_id     = post.get("_id", "unknown")
    status      = post.get("status", "unknown")
    sched_for   = post.get("scheduledFor") or post.get("publishedAt") or ""
    tz          = post.get("timezone", "UTC")

    lines = [
        f"Post created  ·  status: {status}  ·  ID: {post_id}",
    ]

    if sched_for:
        label = "Published at" if status == "published" else "Scheduled for"
        lines.append(f"{label}: {sched_for} ({tz})")

    lines.append("")
    lines.append("Platforms:")

    plat_entries = post.get("platforms", [])
    for i, pe in enumerate(plat_entries, 1):
        plat    = pe.get("platform", "")
        acct    = pe.get("accountId") or {}
        uname   = ""
        if isinstance(acct, dict):
            uname = acct.get("username") or acct.get("displayName") or ""
        pstatus = pe.get("status", "")
        url     = pe.get("platformPostUrl", "")
        error   = pe.get("errorMessage", "")

        line = f"  [{i}]  {plat:<14}  {uname:<20}  {pstatus}"
        if url:
            line += f"\n        {url}"
        if error:
            line += f"\n        ERROR: {error}"
        lines.append(line)

    if message:
        lines.append(f"\n{message}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# serviceSocialPost
# ---------------------------------------------------------------------------

class serviceSocialPost(Tool):
    """
    Publish or schedule social media posts via the Late API.

    Use :meth:`run` to publish or schedule a post, and :meth:`accounts` to
    list connected social accounts (call this first to discover account IDs).

    Posts can be sent to one or more platforms simultaneously.  Supports
    plain text, images, videos, and GIFs.  Publish immediately or schedule
    for a future time.

    Parameters
    ----------
    api_key : str | None, optional
        Late API key.  When omitted the value of the ``LATE_API_KEY``
        environment variable is used.

    Examples
    --------
    Discover connected accounts first::

        from tools.services import serviceSocialPost

        tool = serviceSocialPost()
        print(tool.accounts())

    Publish immediately to Twitter and LinkedIn::

        result = tool.run(
            input="Big news — we just launched!",
            options={
                "platforms": [
                    {"platform": "twitter",  "account_id": "64e1f0..."},
                    {"platform": "linkedin", "account_id": "64e2f0..."},
                ],
                "publish_now": True,
            },
        )
        print(result)

    Schedule a post with an image::

        result = tool.run(
            input="Check out our new product drop!",
            options={
                "platforms":    [{"platform": "instagram", "account_id": "64e3f0..."}],
                "media_urls":   ["https://cdn.example.com/product.jpg"],
                "publish_now":  False,
                "scheduled_at": "2025-02-01T10:00:00Z",
                "timezone":     "America/New_York",
            },
        )
    """

    name        = "serviceSocialPost"
    description = (
        "Publish or schedule social media posts via the Late API. "
        "Use run() to publish, accounts() to list connected accounts."
    )
    parameters  = {
        "type": "object",
        "properties": {
            "input": {
                "type": "string",
                "description": "Post caption / text content.",
            },
            "options": {
                "type": "object",
                "description": "Publishing options.",
                "properties": {
                    "platforms": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["platform", "account_id"],
                            "properties": {
                                "platform":       {"type": "string"},
                                "account_id":     {"type": "string"},
                                "custom_content": {"type": "string"},
                            },
                        },
                        "description": (
                            "Platforms to post to. Each needs 'platform' and "
                            "'account_id' (from accounts())."
                        ),
                    },
                    "media_urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Public URLs of images, videos, or GIFs to attach.",
                    },
                    "publish_now": {
                        "type": "boolean",
                        "default": True,
                        "description": "Publish immediately when True (default).",
                    },
                    "scheduled_at": {
                        "type": "string",
                        "description": (
                            "ISO 8601 datetime for scheduling "
                            "(required when publish_now=false)."
                        ),
                    },
                    "timezone": {
                        "type": "string",
                        "default": "UTC",
                        "description": "IANA timezone. Default: UTC.",
                    },
                    "is_draft": {
                        "type": "boolean",
                        "default": False,
                        "description": "Save as draft without scheduling.",
                    },
                },
                "required": ["platforms"],
            },
        },
        "required": ["input"],
    }

    # ------------------------------------------------------------------

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or os.environ.get(_ENV_KEY)
        if not key:
            raise ValueError(
                f"No Late API key found.  Pass api_key= or set {_ENV_KEY}."
            )
        self._api_key = key
        self._http    = urllib3.PoolManager()

    # ------------------------------------------------------------------

    def run(
        self,
        input:   str,
        options: dict | None = None,
    ) -> str:
        """
        Create a post and return a formatted result string.

        Parameters
        ----------
        input : str
            Post caption / text content.
        options : dict | None, optional
            Publishing options:
              ``platforms``    — list of ``{"platform": ..., "account_id": ...,
                                "custom_content": ...}`` dicts.  Get IDs from
                                :meth:`accounts`.  Required.
              ``media_urls``   — public URLs of images/videos/GIFs to attach.
              ``publish_now``  — publish immediately when True (default).
              ``scheduled_at`` — ISO 8601 datetime when ``publish_now=False``.
              ``timezone``     — IANA timezone string.  Default ``"UTC"``.
              ``is_draft``     — save as draft without scheduling.

        Returns
        -------
        str
            Formatted result including post ID, status, and per-platform URLs.

        Raises
        ------
        RuntimeError
            On any non-2xx HTTP response from the Late API.
        ValueError
            When ``platforms`` is not provided or is empty.
        """
        opts         = options or {}
        platforms    = opts.get("platforms")
        media_urls   = opts.get("media_urls")
        publish_now  = opts.get("publish_now", True)
        scheduled_at = opts.get("scheduled_at")
        timezone     = opts.get("timezone", "UTC")
        is_draft     = opts.get("is_draft", False)

        if not platforms:
            raise ValueError(
                "platforms is required.  "
                "Call accounts() first to discover account IDs, then pass "
                "options={'platforms': [{'platform': ..., 'account_id': ...}]}."
            )

        content = input

        # ── Build platforms list ───────────────────────────────────────
        platform_list = []
        for p in platforms:
            entry: dict = {
                "platform":  p["platform"],
                "accountId": p["account_id"],
            }
            if p.get("custom_content"):
                entry["customContent"] = p["custom_content"]
            platform_list.append(entry)

        # ── Build request body ─────────────────────────────────────────
        body: dict = {
            "platforms":  platform_list,
            "publishNow": publish_now,
            "timezone":   timezone,
            "isDraft":    is_draft,
        }
        if content:
            body["content"] = content
        if scheduled_at and not publish_now:
            body["scheduledFor"] = scheduled_at
        if media_urls:
            body["mediaItems"] = [
                {"type": _guess_media_type(url), "url": url}
                for url in media_urls
            ]

        # ── Send request ───────────────────────────────────────────────
        data = _request(self._http, "POST", "/v1/posts", self._api_key, body=body)

        # ── Format output ──────────────────────────────────────────────
        return _format_post_result(data)

    def accounts(
        self,
        input:   None = None,
        options: dict | None = None,
    ) -> str:
        """
        List all connected Late social accounts.

        Returns a human-readable table of account IDs, platform names,
        usernames, and profile names — everything needed to call :meth:`run`.

        Parameters
        ----------
        input : None, optional
            Unused.  Present for interface symmetry.
        options : dict | None, optional
            Filtering options:
              ``profile_id`` — filter accounts by profile ID.
              ``platform``   — filter to a specific platform name.

        Returns
        -------
        str
            Numbered list of connected accounts::

                Connected accounts (3):

                [1]  twitter         @acme                 ID: 64e1f0...  |  Profile: My Brand (64f0...)
                [2]  linkedin        Acme Corp             ID: 64e2f0...  |  Profile: My Brand (64f0...)
                [3]  instagram       @acme                 ID: 64e3f0...  |  Profile: Marketing (65f0...)

                Use the ID and platform name from above in serviceSocialPost · platforms.

        Raises
        ------
        RuntimeError
            On any non-2xx HTTP response from the Late API.
        """
        opts       = options or {}
        profile_id = opts.get("profile_id")
        platform   = opts.get("platform")

        params: dict = {}
        if profile_id: params["profileId"] = profile_id
        if platform:   params["platform"]  = platform

        data     = _request(self._http, "GET", "/v1/accounts", self._api_key,
                            params=params or None)
        accounts = data.get("accounts", [])

        if not accounts:
            return "No connected accounts found."

        lines = [f"Connected accounts ({len(accounts)}):\n"]
        for i, acc in enumerate(accounts, 1):
            acct_id   = acc.get("_id", "")
            plat      = acc.get("platform", "")
            username  = acc.get("username") or acc.get("displayName") or "(no name)"
            profile   = acc.get("profileId") or {}
            if isinstance(profile, dict):
                profile_name = profile.get("name", "")
                profile_id_s = profile.get("_id", "")
            else:
                profile_name = str(profile)
                profile_id_s = ""

            profile_label = (
                f"{profile_name} ({profile_id_s})" if profile_name else profile_id_s
            )
            lines.append(
                f"[{i}]  {plat:<14}  {username:<20}  "
                f"ID: {acct_id}  |  Profile: {profile_label}"
            )

        lines.append(
            "\nUse the ID and platform name from above in serviceSocialPost · platforms."
        )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"serviceSocialPost(name={self.name!r})"
