"""
models._anthropic — AnthropicModel
====================================

Covers the Claude model family:

  claude-opus-4-6                    Most intelligent, best for agents
  claude-sonnet-4-6                  Best speed / intelligence balance
  claude-haiku-4-5-20251001          Fastest with near-frontier intelligence

Default generation parameters
------------------------------
temperature   1.0    Anthropic API default.
                     ⚠ Valid range is 0.0–1.0 (not 0–2 like OpenAI).
                     Anthropic recommends using EITHER temperature OR
                     top_p, not both simultaneously.
max_tokens    8192   Required by the API; 8 192 is a safe default across
                     the full model family.  claude-3-7-sonnet and the
                     Claude 4 series support up to 64 000 output tokens —
                     raise max_tokens in options for long-form generation.
top_p         None   Nucleus sampling.  Disabled by default; use
                     temperature instead.  Set only when temperature=1.0
                     and you specifically need nucleus sampling.
top_k         None   Top-K sampling.  Rarely needed; disabled by default.
cache_control False  When True, the task layer marks eligible content
                     blocks with ``{"cache_control": {"type": "ephemeral",
                     "ttl": "5m"}}`` for Anthropic's prompt-caching
                     feature (TTL options: ``"5m"`` or ``"1h"``).
reasoning     None   Universal reasoning depth (claude-3-7-sonnet-20250219
                     and all Claude 4 models).  Accepts None | "low" |
                     "medium" | "high"; translated to Anthropic's native
                     extended-thinking config via _REASONING_MAP:

                       None     — thinking disabled
                       "low"    — budget_tokens=4 000
                       "medium" — budget_tokens=10 000
                       "high"   — budget_tokens=20 000

                     Temperature is automatically forced to 1.0 whenever
                     reasoning is active (Anthropic requirement).

Recommended overrides by model
--------------------------------
claude-opus-4-6          max_tokens=32000, temperature=1.0
claude-sonnet-4-6        max_tokens=16000, temperature=1.0
claude-haiku-4-5         max_tokens=8192,  temperature=1.0

Environment variable
---------------------
ANTHROPIC_API_KEY
"""

import json

import urllib3

from ._base import Model
from clients._anthropic import AnthropicClient
from clients._constants import DEFAULT_TIMEOUT, DEFAULT_RETRIES


# ---------------------------------------------------------------------------
# Part-level conversion helper
# ---------------------------------------------------------------------------

def _part_to_anthropic(part: dict) -> "dict | None":
    """
    Convert one universal part dict to an Anthropic content block.

    Returns ``None`` for unsupported types (video) so callers can filter.
    """
    ptype = part["type"]

    if ptype == "text":
        return {"type": "text", "text": part["text"]}

    if ptype == "image":
        src  = part["source"]
        kind = src["kind"]
        if kind == "url":
            return {
                "type":   "image",
                "source": {"type": "url", "url": src["url"]},
            }
        if kind in ("base64", "file"):
            return {
                "type":   "image",
                "source": {
                    "type":       "base64",
                    "media_type": src.get("mime", "image/png"),
                    "data":       src["data"],
                },
            }

    if ptype == "audio":
        src  = part["source"]
        kind = src["kind"]
        if kind == "base64":
            return {
                "type":   "document",
                "source": {
                    "type":       "base64",
                    "media_type": src.get("mime", "audio/wav"),
                    "data":       src["data"],
                },
            }
        return None

    # ptype == "video" — not supported by Anthropic
    return None


# ---------------------------------------------------------------------------
# AnthropicModel
# ---------------------------------------------------------------------------

class AnthropicModel(Model):
    """Provider-specific model for the Anthropic Claude API."""

    _ENV_KEY = "ANTHROPIC_API_KEY"

    # ── generation defaults ──────────────────────────────────────────
    _DEFAULT_TEMPERATURE:   float        = 1.0
    _DEFAULT_MAX_TOKENS:    int          = 8192
    _DEFAULT_TOP_P:         float | None = None   # use temp OR top_p, not both
    _DEFAULT_TOP_K:         int   | None = None
    _DEFAULT_CACHE_CONTROL: bool         = False
    _DEFAULT_REASONING:     str   | None = None

    # Maps universal reasoning level → Anthropic budget_tokens value.
    # Temperature must be 1.0 when thinking is active (enforced in to_request).
    _REASONING_MAP: dict = {"low": 4000, "medium": 10000, "high": 20000}

    # ------------------------------------------------------------------

    def _build_client(self, api_key: str, client_options: dict) -> AnthropicClient:
        """
        Construct an :class:`~clients.AnthropicClient`.

        Supported *client_options* keys: ``url``, ``timeout``,
        ``retries``, ``proxy``.
        """
        return AnthropicClient(
            api_key = api_key,
            url     = client_options.get("url"),
            timeout = client_options.get("timeout", DEFAULT_TIMEOUT),
            retries = client_options.get("retries", DEFAULT_RETRIES),
            proxy   = client_options.get("proxy"),
        )

    def to_request(self, messages: list, output: dict) -> "tuple[str, dict]":
        """
        Translate universal messages → Anthropic ``POST /v1/messages``
        ``(path, body)`` pair.

        System messages are lifted into the top-level ``system`` field;
        multiple system blocks are joined with a double newline.
        """
        system_parts:       list[dict] = []
        anthropic_messages: list[dict] = []

        for msg in messages:
            role   = msg["role"]
            blocks = [_part_to_anthropic(p) for p in msg["parts"]]
            blocks = [b for b in blocks if b is not None]
            if not blocks:
                continue
            if role == "system":
                system_parts.extend(blocks)
            else:
                anthropic_messages.append({"role": role, "content": blocks})

        body: dict = {
            "model":       self.name,
            "messages":    anthropic_messages,
            "max_tokens":  self.max_tokens,
            "temperature": self.temperature,
        }

        if system_parts:
            if all(b["type"] == "text" for b in system_parts):
                body["system"] = "\n\n".join(b["text"] for b in system_parts)
            else:
                body["system"] = system_parts

        if self.top_p is not None:
            body["top_p"] = self.top_p
        if self.top_k is not None:
            body["top_k"] = self.top_k

        if self.reasoning:
            budget = self._REASONING_MAP.get(self.reasoning)
            if budget is not None:
                body["thinking"]     = {"type": "enabled", "budget_tokens": budget}
                body["temperature"]  = 1.0   # required by Anthropic when thinking is active

        # ── Structured output via tool_use ────────────────────────────────
        # Anthropic has no native response_format / json_schema parameter.
        # The canonical way to get validated structured output is to define a
        # tool with the desired schema and force the model to call it exactly
        # once via tool_choice={"type":"tool","name":"..."}.
        fmt   = output.get("format", {})
        ftype = fmt.get("type", "text")
        if ftype == "json_schema":
            tool_name = fmt.get("name", "structured_output")
            body["tools"] = [{
                "name":         tool_name,
                "description":  "Return the result matching the given schema.",
                "input_schema": fmt["schema"],
            }]
            body["tool_choice"] = {"type": "tool", "name": tool_name}

        return "/v1/messages", body

    def from_response(self, response: dict, output: dict) -> "str | dict":
        """
        Extract the clean result from an Anthropic messages response.

        * ``json_schema`` — finds the ``tool_use`` block and returns its
          ``input`` dict (already a parsed Python object).
        * ``json``        — parses the first text block as JSON.
        * ``text``        — returns the first text block's content as a string.

        Returns an empty string / empty dict when the expected block is absent.
        """
        ftype = output.get("format", {}).get("type", "text")
        content = response.get("content", [])

        if ftype == "json_schema":
            for block in content:
                if block.get("type") == "tool_use":
                    return block.get("input", {})
            return {}

        text = ""
        for block in content:
            if block.get("type") == "text":
                text = block["text"]
                break
        if ftype == "json":
            return json.loads(text)
        return text
