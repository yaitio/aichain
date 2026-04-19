"""
models._google — GoogleAIModel
================================

Covers the Gemini model family:

  gemini-3.1-pro-preview       Most capable Gemini 3.1
  gemini-3.1-flash-lite-preview  Lightweight / lowest cost Gemini 3.1
  gemini-3.1-flash-image-preview Image generation (text-to-image)
  gemini-3-flash-preview         Fast multimodal Gemini 3
  gemini-3-pro-image-preview     High-quality image generation (text-to-image)
  gemini-2.5-pro                 Most capable Gemini 2.5, extended thinking
  gemini-2.5-flash               Speed-optimised Gemini 2.5

Default generation parameters
------------------------------
temperature   1.0   Gemini 2.x API default (Gemini 1.5 used 0.9).
max_tokens    8192  maxOutputTokens; applies to all current Gemini models.
                    gemini-2.5-pro supports up to 65 536 (with thinking
                    disabled); raise in options for long-form tasks.
top_p         0.95  Google's recommended default for controlled generation.
top_k         40    Google's default for Gemini 2.x Flash models.
                    Gemini 1.5 Pro used topK=64; adjust in options if needed.
cache_control False Context caching is a separate Gemini API feature; this
                    flag is reserved for future use.
reasoning     None  Universal reasoning depth for gemini-2.5-pro / flash.
                    Accepts None | "low" | "medium" | "high"; translated to
                    ``thinkingConfig.thinkingBudget`` via _REASONING_MAP:
                      "low"    → 2 048 tokens
                      "medium" → 8 192 tokens
                      "high"   → 24 576 tokens

Recommended overrides by model
--------------------------------
gemini-3.1-pro-preview        temperature=1.0, max_tokens=32768, reasoning="medium"
gemini-3.1-flash-lite-preview temperature=1.0, top_p=0.95, top_k=40, max_tokens=8192
gemini-3-flash-preview         temperature=1.0, top_p=0.95, top_k=40, max_tokens=8192
gemini-2.5-pro                 temperature=1.0, max_tokens=32768, reasoning="medium"
gemini-2.5-flash               temperature=1.0, top_p=0.95, top_k=40, max_tokens=8192

Note
----
Google authenticates via query-string ``?key=<api_key>`` rather than an
Authorization header.  The client handles this transparently.

Environment variable
---------------------
GOOGLE_AI_API_KEY
"""

import json

import urllib3

from ._base import Model
from clients._google import GoogleAIClient
from clients._constants import DEFAULT_TIMEOUT, DEFAULT_RETRIES


# ---------------------------------------------------------------------------
# Schema sanitisation for Google's responseSchema proto
# ---------------------------------------------------------------------------

def _sanitize_google_schema(schema: object) -> object:
    """
    Convert a JSON Schema dict to a form accepted by Google's ``responseSchema``
    proto field.

    Two incompatibilities are fixed recursively:

    * ``additionalProperties``          — not supported; stripped silently.
    * ``"type": ["X", "null"]``         — union types are not supported;
      converted to ``"type": "X", "nullable": true``.
    """
    if not isinstance(schema, dict):
        return schema

    result: dict = {}
    for key, value in schema.items():
        if key == "additionalProperties":
            continue                              # strip
        if key == "type" and isinstance(value, list):
            non_null = [t for t in value if t != "null"]
            result["type"] = non_null[0] if non_null else "string"
            if "null" in value:
                result["nullable"] = True
        elif key == "properties" and isinstance(value, dict):
            result[key] = {k: _sanitize_google_schema(v) for k, v in value.items()}
        elif key == "items" and isinstance(value, dict):
            result[key] = _sanitize_google_schema(value)
        else:
            result[key] = value
    return result


# ---------------------------------------------------------------------------
# Part-level conversion helper
# ---------------------------------------------------------------------------

def _part_to_google(part: dict) -> "dict | None":
    """
    Convert one universal part dict to a Google AI part object.

    Returns ``None`` for unsupported source kinds so callers can filter.
    """
    ptype = part["type"]

    if ptype == "text":
        return {"text": part["text"]}

    if ptype == "image":
        src  = part["source"]
        kind = src["kind"]
        if kind == "url":
            return {
                "fileData": {
                    "mimeType": src.get("mime", "image/png"),
                    "fileUri":  src["url"],
                },
            }
        if kind in ("base64", "file"):
            return {
                "inlineData": {
                    "mimeType": src.get("mime", "image/png"),
                    "data":     src["data"],
                },
            }

    if ptype == "video":
        src  = part["source"]
        kind = src["kind"]
        if kind == "url":
            return {
                "fileData": {
                    "mimeType": src.get("mime", "video/mp4"),
                    "fileUri":  src["url"],
                },
            }
        if kind in ("base64", "file"):
            return {
                "inlineData": {
                    "mimeType": src.get("mime", "video/mp4"),
                    "data":     src["data"],
                },
            }

    if ptype == "audio":
        src  = part["source"]
        kind = src["kind"]
        if kind == "base64":
            return {
                "inlineData": {
                    "mimeType": src.get("mime", "audio/wav"),
                    "data":     src["data"],
                },
            }
        if kind == "url":
            return {
                "fileData": {
                    "mimeType": src.get("mime", "audio/wav"),
                    "fileUri":  src["url"],
                },
            }

    return None


# ---------------------------------------------------------------------------
# GoogleAIModel
# ---------------------------------------------------------------------------

class GoogleAIModel(Model):
    """Provider-specific model for the Google AI (Gemini) API."""

    _ENV_KEY = "GOOGLE_AI_API_KEY"

    # ── generation defaults ──────────────────────────────────────────
    _DEFAULT_TEMPERATURE:   float        = 1.0
    _DEFAULT_MAX_TOKENS:    int          = 8192
    _DEFAULT_TOP_P:         float | None = 0.95  # Google's recommended default
    _DEFAULT_TOP_K:         int   | None = 40    # Gemini 2.x Flash default
    _DEFAULT_CACHE_CONTROL: bool         = False
    _DEFAULT_REASONING:     str   | None = None

    # Maps universal reasoning level → thinkingBudget token count.
    _REASONING_MAP: dict = {"low": 2048, "medium": 8192, "high": 24576}

    # ------------------------------------------------------------------

    def _build_client(self, api_key: str, client_options: dict) -> GoogleAIClient:
        """
        Construct a :class:`~clients.GoogleAIClient`.

        Supported *client_options* keys: ``url``, ``timeout``,
        ``retries``, ``proxy``.
        """
        return GoogleAIClient(
            api_key = api_key,
            url     = client_options.get("url"),
            timeout = client_options.get("timeout", DEFAULT_TIMEOUT),
            retries = client_options.get("retries", DEFAULT_RETRIES),
            proxy   = client_options.get("proxy"),
        )

    def to_request(self, messages: list, output: dict) -> "tuple[str, dict]":
        """
        Translate universal messages → Google AI ``generateContent``
        ``(path, body)`` pair.

        The API key is embedded in the path (``?key=…``) because Google
        authenticates via query string rather than a header.  System
        messages are lifted into ``system_instruction``.
        """
        system_parts:    list[dict] = []
        google_contents: list[dict] = []

        for msg in messages:
            role   = msg["role"]
            gparts = [_part_to_google(p) for p in msg["parts"]]
            gparts = [gp for gp in gparts if gp is not None]
            if not gparts:
                continue
            if role == "system":
                system_parts.extend(gparts)
            else:
                google_role = "model" if role == "assistant" else "user"
                google_contents.append({"role": google_role, "parts": gparts})

        generation_config: dict = {
            "temperature":     self.temperature,
            "maxOutputTokens": self.max_tokens,
        }
        if self.top_p is not None:
            generation_config["topP"] = self.top_p
        if self.top_k is not None:
            generation_config["topK"] = self.top_k

        if self.reasoning:
            budget = self._REASONING_MAP.get(self.reasoning)
            if budget is not None:
                generation_config["thinkingConfig"] = {"thinkingBudget": budget}

        fmt        = output.get("format", {})
        ftype      = fmt.get("type", "text")
        modalities = output.get("modalities", ["text"])

        if "image" in modalities:
            # Map universal modality names → Google's responseModalities values
            response_mods = ["IMAGE"]
            if "text" in modalities:
                response_mods.append("TEXT")
            generation_config["responseModalities"] = response_mods
        elif ftype == "json":
            generation_config["responseMimeType"] = "application/json"
        elif ftype == "json_schema":
            generation_config["responseMimeType"] = "application/json"
            generation_config["responseSchema"]   = _sanitize_google_schema(fmt["schema"])

        body: dict = {
            "contents":         google_contents,
            "generationConfig": generation_config,
        }
        if system_parts:
            body["system_instruction"] = {"parts": system_parts}

        path = f"/models/{self.name}:generateContent?key={self._api_key}"
        return path, body

    def from_response(self, response: dict, output: dict) -> "str | dict":
        """
        Extract the clean result from a Google AI ``generateContent``
        response.

        For text/json/json_schema output, returns the first text part.
        For image output, returns a dict with ``base64`` and ``mime_type`` keys.
        """
        ftype = output.get("format", {}).get("type", "text")
        candidates = response.get("candidates", [])
        if not candidates:
            if ftype == "image":
                return {"url": None, "base64": None, "mime_type": "image/png", "revised_prompt": ""}
            return ""
        parts = (
            candidates[0]
            .get("content", {})
            .get("parts", [])
        )

        if ftype == "image":
            image_part = next((p for p in parts if "inlineData" in p), None)
            if image_part:
                return {
                    "url":            None,
                    "base64":         image_part["inlineData"]["data"],
                    "mime_type":      image_part["inlineData"].get("mimeType", "image/png"),
                    "revised_prompt": "",
                }
            return {"url": None, "base64": None, "mime_type": "image/png", "revised_prompt": ""}

        # text / json / json_schema
        try:
            text = next(p["text"] for p in parts if "text" in p)
        except StopIteration:
            text = ""
        if ftype in ("json", "json_schema"):
            return json.loads(text)
        return text
