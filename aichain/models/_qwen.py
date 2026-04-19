"""
models._qwen — QwenModel
=========================

Covers the Alibaba Qwen / DashScope model family:

  Text / chat (via ``/compatible-mode/v1/chat/completions``)
  ──────────────────────────────────────────────────────────
  qwen-max          Flagship Qwen model (128 K context)
  qwen-plus         Balanced quality / speed
  qwen-turbo        Cost-optimised, high throughput
  qwen3-235b-a22b   Qwen3 flagship (235 B MoE, 22 B active)
  qwen3-72b         Qwen3 dense 72 B
  qwen3-32b         Qwen3 dense 32 B
  qwen3-14b         Qwen3 dense 14 B
  qwen3-8b          Qwen3 dense 8 B

  Vision / multimodal (same endpoint, image_url content type)
  ────────────────────────────────────────────────────────────
  qwen-vl-max       Vision-language flagship
  qwen-vl-plus      Vision-language balanced
  qwen2.5-vl-max    Qwen 2.5 vision
  qwen2.5-vl-plus   Qwen 2.5 vision balanced

  Reasoning (always-on thinking via ``enable_thinking: true``)
  ─────────────────────────────────────────────────────────────
  QwQ-32B           Always-on chain-of-thought reasoning model

  Image generation (via ``/compatible-mode/v1/images/generations``)
  ──────────────────────────────────────────────────────────────────
  wanx-v1            Wanx image generation
  wanx2.1-t2i-turbo  Wanx 2.1 turbo
  wanx2.1-t2i-plus   Wanx 2.1 plus

Default generation parameters
------------------------------
temperature   0.7    DashScope default for chat models.
max_tokens    2048   Conservative default; qwen-max supports up to 8 192.
top_p         0.8    DashScope default.
top_k         None   Not used by default.
cache_control False  Not supported.
reasoning     None   Universal reasoning depth.

Reasoning support
-----------------
``QwQ-*`` models always have thinking enabled (``enable_thinking: true``
is added unconditionally).

``qwen3-*`` models support optional thinking mode:
  any non-None reasoning level → adds ``enable_thinking: true``.
  None → standard mode (no thinking tokens).

All other Qwen models do not support the reasoning parameter.

Region selection
----------------
The base URL is derived from the ``DASHSCOPE_REGION`` env var (or the
``region`` key in *client_options*):

  ap (default) — https://dashscope-intl.aliyuncs.com
  us           — https://dashscope-us.aliyuncs.com
  cn           — https://dashscope.aliyuncs.com
  hk           — https://cn-hongkong.dashscope.aliyuncs.com

Environment variable
---------------------
DASHSCOPE_API_KEY
"""

from __future__ import annotations

import urllib3

from ._base import Model
from ._openai import (
    _build_openai_compat_request,
    _parse_openai_compat_response,
    _build_image_generations_request,
    _parse_image_generations_response,
)
from clients._qwen import QwenClient
from clients._constants import DEFAULT_TIMEOUT, DEFAULT_RETRIES


def _is_qwen_image_model(name: str) -> bool:
    """Return True when *name* identifies a Qwen image-generation model."""
    lower = name.lower()
    return lower.startswith("wanx")


def _is_qwq_model(name: str) -> bool:
    """Return True when *name* is a QwQ always-on reasoning model."""
    return name.lower().startswith("qwq")


def _is_qwen3_model(name: str) -> bool:
    """Return True when *name* is a Qwen3 series model (optional thinking)."""
    return name.lower().startswith("qwen3")


class QwenModel(Model):
    """Provider-specific model for the Alibaba DashScope / Qwen API."""

    _ENV_KEY = "DASHSCOPE_API_KEY"

    # ── generation defaults ──────────────────────────────────────────
    _DEFAULT_TEMPERATURE:   float        = 0.7
    _DEFAULT_MAX_TOKENS:    int          = 2048
    _DEFAULT_TOP_P:         float | None = 0.8
    _DEFAULT_TOP_K:         int   | None = None
    _DEFAULT_CACHE_CONTROL: bool         = False
    _DEFAULT_REASONING:     str   | None = None

    # Reasoning map — Qwen does not have effort levels; any level enables thinking.
    _REASONING_MAP: dict = {"low": True, "medium": True, "high": True}

    # ------------------------------------------------------------------

    def _build_client(self, api_key: str, client_options: dict) -> QwenClient:
        """
        Construct a :class:`~clients.QwenClient`.

        Supported *client_options* keys: ``url``, ``region``, ``timeout``,
        ``retries``, ``proxy``.
        """
        return QwenClient(
            api_key = api_key,
            url     = client_options.get("url"),
            region  = client_options.get("region"),
            timeout = client_options.get("timeout", DEFAULT_TIMEOUT),
            retries = client_options.get("retries", DEFAULT_RETRIES),
            proxy   = client_options.get("proxy"),
        )

    def to_request(self, messages: list, output: dict) -> "tuple[str, dict]":
        """
        Translate universal messages into the correct DashScope request format.

        Routing
        -------
        * Image models (``wanx-*``) → **Images Generations API**
          (``POST /compatible-mode/v1/images/generations``).
        * All other models → **Chat Completions API**
          (``POST /compatible-mode/v1/chat/completions``).

        Thinking / reasoning
        --------------------
        * ``QwQ-*`` models: ``enable_thinking: true`` is always injected.
        * ``qwen3-*`` models: ``enable_thinking: true`` is injected when
          ``self.reasoning`` is not None.
        * All other models: no thinking parameter.
        """
        if _is_qwen_image_model(self.name):
            path, body = _build_image_generations_request(
                self, messages, output,
                "/compatible-mode/v1/images/generations",
            )
            return path, body

        path, body = _build_openai_compat_request(
            self, messages, output, "/compatible-mode/v1/chat/completions"
        )

        # Enable thinking for QwQ models (always) and Qwen3 (when reasoning set).
        if _is_qwq_model(self.name):
            body["enable_thinking"] = True
        elif _is_qwen3_model(self.name) and self.reasoning:
            body["enable_thinking"] = True

        return path, body

    def from_response(self, response: dict, output: dict) -> "str | dict":
        """
        Extract the clean result from a DashScope response.

        Detects the API automatically:
        * ``"data"``    present → Images Generations response
        * ``"choices"`` present → Chat Completions response
        """
        if "data" in response:
            return _parse_image_generations_response(response)
        return _parse_openai_compat_response(response, output)
