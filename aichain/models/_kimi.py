"""
models._kimi — KimiModel
=========================

Covers the Kimi (Moonshot AI) model family:

  Chat / text  (Chat Completions — /v1/chat/completions)
  ────────────
  K2.5 series   kimi-k2.5              — multimodal, thinking toggle
  K2 series     kimi-k2-0905-preview   — text-only, 256K context
                kimi-k2-turbo-preview  — text-only, high-speed (60–100 tok/s)
  Thinking      kimi-k2-thinking       — always-on reasoning, 256K context
                kimi-k2-thinking-turbo — fast always-on reasoning

  Multimodal (kimi-k2.5 only)
  ─────────────────────────────
  Images : PNG, JPEG, WebP, GIF (max 4K resolution; body ≤ 100 MB)
  Video  : MP4, MPEG, MOV, AVI, WebM (max 2K resolution)

Thinking mode
─────────────
Kimi's thinking feature is distinct from the universal ``reasoning``
option name but maps cleanly:

  Universal     Kimi API parameter
  ──────────────────────────────────
  None          thinking param omitted (model decides its own default)
  "low"         {"thinking": {"type": "enabled"}}
  "medium"      {"thinking": {"type": "enabled"}}
  "high"        {"thinking": {"type": "enabled"}}

Kimi does not expose a budget_tokens-style granularity — thinking is
simply on or off.  The three universal levels all map to "enabled" so
that cross-provider code works without special-casing Kimi.  When only
kimi-k2-thinking or kimi-k2-thinking-turbo is used, the thinking
parameter is not needed (always on); passing it is harmless.

Temperature constraints
───────────────────────
The Kimi API enforces fixed temperatures depending on thinking mode:

  thinking enabled  → temperature must be 1.0
  thinking disabled → temperature should be 0.6

The library sets 1.0 as the default (thinking-mode safe) and overrides
to 1.0 whenever reasoning is active.  Users who need the non-thinking
temperature of 0.6 should pass ``options={"temperature": 0.6}``
explicitly.  Passing any other fixed temperature will be rejected by
the API when thinking mode is active.

Default generation parameters
──────────────────────────────
temperature   1.0    API default for thinking mode; must be 1.0 when
                     thinking is enabled (API enforces this).
max_tokens    32768  Kimi API default.
top_p         0.95   Kimi API default; valid range 0–1.
top_k         None   Not supported.
cache_control False  Not exposed by Kimi's API.
reasoning     None   Maps to thinking parameter (see above).

Environment variable
────────────────────
MOONSHOT_API_KEY
"""

import json

import urllib3

from ._base import Model
from ._openai import _part_to_openai, _parse_openai_compat_response
from clients._kimi import KimiClient
from clients._constants import DEFAULT_TIMEOUT, DEFAULT_RETRIES


# ---------------------------------------------------------------------------
# KimiModel
# ---------------------------------------------------------------------------

class KimiModel(Model):
    """Provider-specific model for the Kimi (Moonshot AI) API."""

    _ENV_KEY = "MOONSHOT_API_KEY"

    # ── generation defaults ──────────────────────────────────────────────
    _DEFAULT_TEMPERATURE:   float        = 1.0
    _DEFAULT_MAX_TOKENS:    int          = 32768
    _DEFAULT_TOP_P:         float | None = 0.95
    _DEFAULT_TOP_K:         int   | None = None   # unsupported
    _DEFAULT_CACHE_CONTROL: bool         = False
    _DEFAULT_REASONING:     str   | None = None

    # All three universal levels map to enabled — Kimi has no token-budget knob.
    _REASONING_MAP: dict = {
        "low":    "enabled",
        "medium": "enabled",
        "high":   "enabled",
    }

    # ------------------------------------------------------------------

    def _build_client(self, api_key: str, client_options: dict) -> KimiClient:
        """
        Construct a :class:`~clients.KimiClient`.

        Supported *client_options* keys: ``url``, ``timeout``,
        ``retries``, ``proxy``.
        """
        return KimiClient(
            api_key = api_key,
            url     = client_options.get("url"),
            timeout = client_options.get("timeout", DEFAULT_TIMEOUT),
            retries = client_options.get("retries", DEFAULT_RETRIES),
            proxy   = client_options.get("proxy"),
        )

    def to_request(self, messages: list, output: dict) -> "tuple[str, dict]":
        """
        Translate universal messages into a Kimi ``/v1/chat/completions`` request.

        The Kimi API is OpenAI-compatible, so message conversion re-uses
        :func:`~models._openai._part_to_openai`.

        Thinking mode
        -------------
        When ``self.reasoning`` is set (any of ``"low"``, ``"medium"``,
        ``"high"``), a ``"thinking": {"type": "enabled"}`` field is added to
        the request body and temperature is forced to 1.0 (the API requires
        this when thinking is active).

        When ``self.reasoning`` is ``None`` the ``thinking`` field is omitted
        entirely — the model uses its own default (always-on for
        ``kimi-k2-thinking`` / ``kimi-k2-thinking-turbo``; off for the others).
        """
        # ── Convert universal messages to OpenAI-compatible format ────
        openai_messages: list[dict] = []
        for msg in messages:
            role  = msg["role"]
            items = [_part_to_openai(p) for p in msg["parts"]]
            items = [it for it in items if it is not None]
            if not items:
                continue
            if len(items) == 1 and items[0]["type"] == "text":
                openai_messages.append({"role": role, "content": items[0]["text"]})
            else:
                openai_messages.append({"role": role, "content": items})

        # ── Thinking mode ─────────────────────────────────────────────
        thinking_active = (
            self.reasoning is not None
            and self._REASONING_MAP.get(self.reasoning) == "enabled"
        )

        # Temperature must be 1.0 when thinking is enabled (API constraint).
        temperature = 1.0 if thinking_active else self.temperature

        body: dict = {
            "model":       self.name,
            "messages":    openai_messages,
            "max_tokens":  self.max_tokens,
            "temperature": temperature,
        }

        if self.top_p is not None:
            body["top_p"] = self.top_p

        if thinking_active:
            body["thinking"] = {"type": "enabled"}

        # ── Output format ─────────────────────────────────────────────
        fmt   = output.get("format", {})
        ftype = fmt.get("type", "text")
        if ftype == "json":
            body["response_format"] = {"type": "json_object"}
        elif ftype == "json_schema":
            body["response_format"] = {
                "type":        "json_schema",
                "json_schema": {
                    "name":   fmt.get("name", "response"),
                    "schema": fmt["schema"],
                    "strict": fmt.get("strict", True),
                },
            }

        return "/v1/chat/completions", body

    def from_response(self, response: dict, output: dict) -> "str | dict":
        """
        Extract the clean result from a Kimi Chat Completions response.

        Uses the same OpenAI-compatible response shape:
        ``choices[0].message.content``.
        """
        return _parse_openai_compat_response(response, output)
