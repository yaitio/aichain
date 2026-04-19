"""
models._xai — XAIModel
========================

Covers the xAI Grok model family:

  grok-4-0709               Most capable (always a reasoning model)
  grok-4-fast-reasoning     Speed-optimised Grok 4 reasoning
  grok-4-1-fast-reasoning   Grok 4.1 fast reasoning
  grok-3                    Flagship non-reasoning model (131 K context)
  grok-3-fast               Speed-optimised grok-3
  grok-3-mini               Lightweight; supports reasoning_effort
  grok-3-mini-fast          Fastest grok-3 variant with reasoning support
  grok-imagine-image        Image generation
  grok-imagine-image-pro    High-quality image generation

Default generation parameters
------------------------------
temperature   1.0    OpenAI-compatible API default; valid range 0–2.
max_tokens    16384  Practical default; grok-3/4 support up to 131 072
                     (grok-3) or 256 000 (grok-4) context tokens.
top_p         1.0    OpenAI-compatible default; range 0–1.
top_k         None   Not part of the OpenAI-compatible parameter set.
cache_control False  Not supported by xAI.
reasoning     None   Universal reasoning depth — **only for grok-3-mini and
                     grok-3-mini-fast**.  All other models either always
                     reason (grok-4 family) or never reason (grok-3).

                     Accepts None | "low" | "medium" | "high"; translated
                     to ``reasoning_effort`` via _REASONING_MAP:
                       "low"            → "low"
                       "medium"/"high"  → "high"  (xAI has no "medium")

                     ⚠ When reasoning is active, the API rejects
                     ``frequency_penalty``, ``presence_penalty``, and
                     ``stop`` — do not send them for reasoning models.

Reasoning support by model
--------------------------
grok-3-mini           YES — reasoning="low"|"medium"|"high"
grok-3-mini-fast      YES — reasoning="low"|"medium"|"high"
grok-4-0709           ALWAYS reasoning — no effort parameter accepted
grok-4-fast-reasoning ALWAYS reasoning — no effort parameter accepted
grok-3                NO  — reasoning not supported (error if sent)
grok-3-fast           NO  — reasoning not supported (error if sent)

Recommended overrides by model
--------------------------------
grok-3              temperature=1.0, max_tokens=16384
grok-3-fast         temperature=1.0, max_tokens=16384
grok-3-mini         temperature=1.0, reasoning="high"
grok-4-0709         temperature=1.0, max_tokens=32768  (always reasons)

Note
----
The xAI API is OpenAI-compatible.  Request and response shapes are
identical to OpenAI's ``/v1/chat/completions`` and
``/v1/images/generations``; only the base URL and model roster differ.

Environment variable
---------------------
XAI_API_KEY
"""

import urllib3

from ._base import Model
from ._openai import (
    _build_openai_compat_request,
    _parse_openai_compat_response,
    _build_image_generations_request,
    _parse_image_generations_response,
)
from clients._xai import XAIClient
from clients._constants import DEFAULT_TIMEOUT, DEFAULT_RETRIES


def _is_xai_image_model(name: str) -> bool:
    """Return True when *name* identifies an xAI image-generation model."""
    return name.startswith("grok-imagine-")


class XAIModel(Model):
    """Provider-specific model for the xAI (Grok) API."""

    _ENV_KEY = "XAI_API_KEY"

    # ── generation defaults ──────────────────────────────────────────
    _DEFAULT_TEMPERATURE:   float        = 1.0
    _DEFAULT_MAX_TOKENS:    int          = 16384
    _DEFAULT_TOP_P:         float | None = 1.0
    _DEFAULT_TOP_K:         int   | None = None   # unsupported
    _DEFAULT_CACHE_CONTROL: bool         = False  # unsupported
    _DEFAULT_REASONING:     str   | None = None

    # xAI only supports "low" and "high"; map "medium" to "high".
    _REASONING_MAP: dict = {"low": "low", "medium": "high", "high": "high"}

    # ------------------------------------------------------------------

    def _build_client(self, api_key: str, client_options: dict) -> XAIClient:
        """
        Construct an :class:`~clients.XAIClient`.

        Supported *client_options* keys: ``url``, ``timeout``,
        ``retries``, ``proxy``.
        """
        return XAIClient(
            api_key = api_key,
            url     = client_options.get("url"),
            timeout = client_options.get("timeout", DEFAULT_TIMEOUT),
            retries = client_options.get("retries", DEFAULT_RETRIES),
            proxy   = client_options.get("proxy"),
        )

    def to_request(self, messages: list, output: dict) -> "tuple[str, dict]":
        """
        Translate universal messages into the correct xAI request format.

        Routing
        -------
        * Image models (``grok-imagine-*``) are routed to the **Images
          Generations API** (``POST /v1/images/generations``), which is
          OpenAI-compatible and shared with :func:`_build_image_generations_request`.
        * All other models use the **Chat Completions API**
          (``POST /v1/chat/completions``).

        When ``reasoning`` is set the ``reasoning_effort`` parameter is
        added to Chat Completions requests.  xAI only supports ``"low"``
        and ``"high"``; ``"medium"`` is mapped to ``"high"``.
        """
        if _is_xai_image_model(self.name):
            # xAI grok-imagine models do not accept the ``size`` or ``quality``
            # parameters — build the request manually to avoid 400 errors.
            prompt = ""
            for msg in reversed(messages):
                if msg["role"] == "user":
                    texts = [p["text"] for p in msg["parts"] if p["type"] == "text"]
                    if texts:
                        prompt = "\n".join(texts)
                        break
            body = {
                "model":           self.name,
                "prompt":          prompt,
                "n":               1,
                "response_format": "b64_json",
            }
            return "/v1/images/generations", body

        path, body = _build_openai_compat_request(
            self, messages, output, "/v1/chat/completions"
        )
        if self.reasoning:
            effort = self._REASONING_MAP.get(self.reasoning)
            if effort:
                body["reasoning_effort"] = effort
        return path, body

    def from_response(self, response: dict, output: dict) -> "str | dict":
        """
        Extract the clean result from an xAI response.

        Detects the API automatically:
        * ``"data"``    present → Images Generations response
        * ``"choices"`` present → Chat Completions response
        """
        if "data" in response:
            return _parse_image_generations_response(response)
        return _parse_openai_compat_response(response, output)
