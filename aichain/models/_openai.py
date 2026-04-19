"""
models._openai — OpenAIModel
=============================

Covers the full OpenAI model family:

  Chat / reasoning  (Chat Completions API — /v1/chat/completions)
  ────────────────
  GPT-4.1 series  gpt-4.1, gpt-4.1-mini, gpt-4.1-nano
  GPT-4o series   gpt-4o, gpt-4o-mini, gpt-4o-audio-preview, …
  o-series        o1, o1-mini, o3, o3-mini, o4-mini, …

  Chat / reasoning  (Responses API — /v1/responses)
  ────────────────────────────────────────────────
  GPT-5 series    gpt-5, gpt-5-mini, gpt-5.4, gpt-5.4-…
                  These models are ONLY available via /v1/responses.
                  The library auto-detects any model whose name starts
                  with "gpt-5" and routes it to the Responses API.

  Image
  ─────
  DALL-E          dall-e-3, dall-e-2
  GPT Image       gpt-image-1, gpt-image-1.5, gpt-image-1-mini

  Embeddings
  ──────────
  text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002

  Audio
  ─────
  whisper-1, tts-1, tts-1-hd

Responses API vs Chat Completions API
--------------------------------------
The two endpoints differ in request and response shape:

  Feature              Chat Completions          Responses API
  ─────────────────────────────────────────────────────────────
  endpoint             /v1/chat/completions      /v1/responses
  messages key         messages                  input
  system message       inside messages[]         top-level instructions
  token limit param    max_completion_tokens     max_output_tokens
  structured output    response_format           text.format
  response content     choices[0].message        output[0].content[0].text

The library detects the correct path automatically; callers do not need
to change anything.

Default generation parameters
------------------------------
temperature   1.0    OpenAI API default; valid range 0–2.
                     Note: o-series (o1, o3, o4-mini) ignore this
                     parameter — only GPT models use it.
max_tokens    16384  ⚠ ``max_tokens`` is **deprecated** in the OpenAI API
                     in favour of ``max_completion_tokens``.  The task
                     layer must send ``max_completion_tokens`` for all
                     current models (GPT-4o, GPT-4.1, o-series).
                     ``max_tokens`` is incompatible with the o-series.
                     16 384 is a safe default for GPT-4o / GPT-4.1.
top_p         1.0    Nucleus sampling; range 0–1, default 1.
                     OpenAI recommends altering either temperature or
                     top_p, not both.
top_k         None   Not part of the OpenAI API — unsupported.
cache_control False  Automatic prompt caching is available for GPT-4.1
                     and GPT-4o.  When True, the task layer adds
                     ``{"cache_control": {"type": "ephemeral"}}`` to
                     eligible content blocks.
reasoning     None   Universal reasoning depth for o-series models only.
                     Accepts None | "low" | "medium" | "high"; mapped to
                     the ``reasoning_effort`` API parameter:
                       "low"    — faster, fewer reasoning tokens
                       "medium" — API default for o-series
                       "high"   — deepest reasoning, most tokens
                     GPT models ignore this field entirely.

Recommended overrides by model
--------------------------------
gpt-4o / gpt-4.1    temperature=1.0, max_tokens=16384
gpt-4o-mini         temperature=1.0, max_tokens=16384
gpt-4.1-nano        temperature=1.0, max_tokens=8192
gpt-5 / gpt-5-mini  temperature=1.0, max_tokens=16384  (Responses API)
o1                  reasoning="medium", max_tokens=32768
o3 / o4-mini        reasoning="high",   max_tokens=32768
dall-e-3            temperature / max_tokens not applicable
gpt-image-1(.5)     temperature / max_tokens not applicable

Environment variable
---------------------
OPENAI_API_KEY
"""

import copy
import json

import urllib3

from ._base import Model
from clients._openai import OpenAIClient
from clients._constants import DEFAULT_TIMEOUT, DEFAULT_RETRIES


# ---------------------------------------------------------------------------
# Part-level conversion helper (also imported by XAIModel and PerplexityModel)
# ---------------------------------------------------------------------------

def _part_to_openai(part: dict) -> "dict | None":
    """
    Convert one universal part dict to an OpenAI content item.

    Returns ``None`` for part types that OpenAI does not support (video, URL
    audio) so callers can filter them out.
    """
    ptype = part["type"]

    if ptype == "text":
        return {"type": "text", "text": part["text"]}

    if ptype == "image":
        src    = part["source"]
        detail = part.get("meta", {}).get("detail", "auto")
        kind   = src["kind"]
        if kind == "url":
            return {
                "type":      "image_url",
                "image_url": {"url": src["url"], "detail": detail},
            }
        if kind in ("base64", "file"):
            mime = src.get("mime", "image/png")
            return {
                "type":      "image_url",
                "image_url": {
                    "url":    f"data:{mime};base64,{src['data']}",
                    "detail": detail,
                },
            }

    if ptype == "audio":
        src  = part["source"]
        kind = src["kind"]
        if kind == "base64":
            mime = src.get("mime", "audio/wav")
            fmt  = mime.split("/")[-1]   # "wav", "mp3", …
            return {
                "type":        "input_audio",
                "input_audio": {"data": src["data"], "format": fmt},
            }
        # URL audio is not supported by OpenAI input_audio
        return None

    # ptype == "video" — not supported by OpenAI chat completions
    return None


def _build_openai_compat_request(
    model,
    messages: list,
    output:   dict,
    path:     str,
) -> "tuple[str, dict]":
    """
    Build an OpenAI-compatible ``(path, body)`` pair.

    Shared by :class:`OpenAIModel`, :class:`~models._xai.XAIModel`, and
    :class:`~models._perplexity.PerplexityModel`.

    Parameters
    ----------
    model    : Provider model instance (OpenAI / xAI / Perplexity).
    messages : Substituted universal messages list.
    output   : Universal output spec.
    path     : Provider-specific endpoint path.
    """
    openai_messages: list[dict] = []
    for msg in messages:
        role  = msg["role"]
        items = [_part_to_openai(p) for p in msg["parts"]]
        items = [it for it in items if it is not None]
        if not items:
            continue
        # Use a plain string when the message is a single text block
        if len(items) == 1 and items[0]["type"] == "text":
            openai_messages.append({"role": role, "content": items[0]["text"]})
        else:
            openai_messages.append({"role": role, "content": items})

    body: dict = {
        "model":                 model.name,
        "messages":              openai_messages,
        "max_completion_tokens": model.max_tokens,
        "temperature":           model.temperature,
    }

    if model.top_p is not None:
        body["top_p"] = model.top_p

    # Output format
    fmt   = output.get("format", {})
    ftype = fmt.get("type", "text")
    if ftype == "json":
        body["response_format"] = {"type": "json_object"}
    elif ftype == "json_schema":
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name":   fmt.get("name", "response"),
                "schema": fmt["schema"],
                "strict": fmt.get("strict", True),
            },
        }

    return path, body


def _parse_openai_compat_response(response: dict, output: dict) -> "str | dict":
    """
    Extract the clean result from an OpenAI-compatible chat completion
    response.

    Shared by :class:`OpenAIModel`, :class:`~models._xai.XAIModel`, and
    :class:`~models._perplexity.PerplexityModel`.
    """
    text  = response["choices"][0]["message"]["content"] or ""
    ftype = output.get("format", {}).get("type", "text")
    if ftype in ("json", "json_schema"):
        return json.loads(text)
    return text


# ---------------------------------------------------------------------------
# Image generation helpers  (for dall-e-* and gpt-image-* models)
# ---------------------------------------------------------------------------

_OPENAI_IMAGE_MODEL_PREFIXES = ("dall-e-", "gpt-image-")


def _is_openai_image_model(name: str) -> bool:
    """Return True when *name* identifies an OpenAI image-generation model."""
    return name.startswith(_OPENAI_IMAGE_MODEL_PREFIXES)


def _build_image_generations_request(
    model,
    messages: list,
    output:   dict,
    path:     str = "/v1/images/generations",
) -> "tuple[str, dict]":
    """
    Build an OpenAI ``/v1/images/generations`` ``(path, body)`` pair.

    Shared by :class:`OpenAIModel` and :class:`~models._xai.XAIModel`.

    The text prompt is extracted from the last user message.  Optional
    ``size`` and ``quality`` values are read from ``output["format"]``.

    ``response_format`` handling
    ----------------------------
    All models that accept the parameter receive ``"b64_json"`` so that
    the output is always an inline base64 string — consistent across every
    provider (OpenAI DALL-E, xAI Grok-Imagine, Google).

    ``gpt-image-*`` models are the single exception: they always return
    ``b64_json`` natively and explicitly reject the parameter, so it is
    omitted for them.

    +-----------------------+--------------------------------------+
    | Model prefix          | response_format sent?                |
    +=======================+======================================+
    | ``dall-e-*``          | Yes → ``"b64_json"``                 |
    | ``gpt-image-*``       | No  → native ``b64_json``            |
    | ``grok-imagine-*``    | Yes → ``"b64_json"``                 |
    +-----------------------+--------------------------------------+
    """
    prompt = ""
    for msg in reversed(messages):
        if msg["role"] == "user":
            texts = [p["text"] for p in msg["parts"] if p["type"] == "text"]
            if texts:
                prompt = "\n".join(texts)
                break

    fmt: dict  = output.get("format", {})
    body: dict = {"model": model.name, "prompt": prompt, "n": 1}

    if fmt.get("size"):
        body["size"] = fmt["size"]
    if fmt.get("quality"):
        body["quality"] = fmt["quality"]

    # Request base64 output on every model that accepts the parameter.
    # gpt-image-* always returns b64_json natively and rejects the param —
    # omit it only for that family.
    if not model.name.startswith("gpt-image-"):
        body["response_format"] = "b64_json"

    return path, body


def _detect_image_mime(b64_data: "str | None") -> str:
    """
    Return the MIME type of a base64-encoded image by inspecting its magic bytes.

    Handles JPEG (``FF D8 FF``), PNG (``89 50 4E 47``), WebP (``RIFF…WEBP``),
    and GIF (``GIF8``).  Falls back to ``"image/png"`` for unknown formats.
    """
    if not b64_data:
        return "image/png"
    try:
        import base64 as _b64
        # Decode just enough bytes to read the magic header (12 bytes → 16 b64 chars)
        header = _b64.b64decode(b64_data[:16] + "==")
        if header[:3] == b"\xff\xd8\xff":
            return "image/jpeg"
        if header[:4] == b"\x89PNG":
            return "image/png"
        if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
            return "image/webp"
        if header[:4] == b"GIF8":
            return "image/gif"
    except Exception:
        pass
    return "image/png"


def _parse_image_generations_response(response: dict) -> dict:
    """
    Extract the image result from an OpenAI ``/v1/images/generations``
    response.

    Shared by :class:`OpenAIModel` and :class:`~models._xai.XAIModel`.

    Returns a dict with keys ``url``, ``base64``, ``mime_type``, and
    ``revised_prompt`` — consistent with the Google image response format.

    ``mime_type`` is detected from the image's magic bytes so it is always
    accurate, regardless of which provider generated the image.
    """
    item  = response.get("data", [{}])[0]
    b64   = item.get("b64_json")
    return {
        "url":            item.get("url"),
        "base64":         b64,
        "mime_type":      _detect_image_mime(b64),
        "revised_prompt": item.get("revised_prompt", ""),
    }


# ---------------------------------------------------------------------------
# Responses API helpers  (for gpt-5 and future models on /v1/responses)
# ---------------------------------------------------------------------------

# Model name prefixes that require the Responses API instead of Chat Completions.
_RESPONSES_API_PREFIXES = ("gpt-5",)


def _should_use_responses_api(model_name: str) -> bool:
    """Return True when *model_name* must be called via /v1/responses."""
    return any(model_name.startswith(p) for p in _RESPONSES_API_PREFIXES)


def _build_responses_api_request(
    model,
    messages: list,
    output:   dict,
) -> "tuple[str, dict]":
    """
    Build an OpenAI ``/v1/responses`` ``(path, body)`` pair.

    Key differences from Chat Completions:
    - ``input``        instead of ``messages``
    - ``instructions`` for the system message (separate top-level field)
    - ``max_output_tokens`` instead of ``max_completion_tokens``
    - ``text.format``  instead of ``response_format`` for structured output
    """
    instructions   = None
    input_messages: list[dict] = []

    for msg in messages:
        role  = msg["role"]
        items = [_part_to_openai(p) for p in msg["parts"]]
        items = [it for it in items if it is not None]
        if not items:
            continue

        if role == "system":
            # Responses API takes the system prompt as a top-level field.
            text_parts   = [it["text"] for it in items if it["type"] == "text"]
            instructions = "\n".join(text_parts)
        else:
            if len(items) == 1 and items[0]["type"] == "text":
                input_messages.append({"role": role, "content": items[0]["text"]})
            else:
                input_messages.append({"role": role, "content": items})

    body: dict = {
        "model":             model.name,
        "input":             input_messages,
        "max_output_tokens": model.max_tokens,
        "temperature":       model.temperature,
    }

    if instructions:
        body["instructions"] = instructions

    if model.top_p is not None:
        body["top_p"] = model.top_p

    # Structured output — lives under ``text.format`` in the Responses API
    fmt   = output.get("format", {})
    ftype = fmt.get("type", "text")
    if ftype == "json":
        body["text"] = {"format": {"type": "json_object"}}
    elif ftype == "json_schema":
        body["text"] = {
            "format": {
                "type":   "json_schema",
                "name":   fmt.get("name", "response"),
                "schema": fmt["schema"],
                "strict": fmt.get("strict", True),
            }
        }

    return "/v1/responses", body


def _parse_responses_api_response(response: dict, output: dict) -> "str | dict":
    """
    Extract the result from an OpenAI Responses API response.

    Response structure::

        {
          "output": [
            {
              "type": "message",
              "role": "assistant",
              "content": [{"type": "output_text", "text": "…"}]
            }
          ]
        }
    """
    for item in response.get("output", []):
        if item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    text  = part["text"]
                    ftype = output.get("format", {}).get("type", "text")
                    if ftype in ("json", "json_schema"):
                        return json.loads(text)
                    return text
    return ""


# ---------------------------------------------------------------------------
# OpenAIModel
# ---------------------------------------------------------------------------

class OpenAIModel(Model):
    """Provider-specific model for the OpenAI API."""

    _ENV_KEY = "OPENAI_API_KEY"

    # ── generation defaults ──────────────────────────────────────────
    _DEFAULT_TEMPERATURE:   float        = 1.0
    _DEFAULT_MAX_TOKENS:    int          = 16384
    _DEFAULT_TOP_P:         float | None = 1.0
    _DEFAULT_TOP_K:         int   | None = None   # unsupported
    _DEFAULT_CACHE_CONTROL: bool         = False
    _DEFAULT_REASONING:     str   | None = None

    # Maps universal reasoning level → OpenAI reasoning_effort value
    _REASONING_MAP: dict = {"low": "low", "medium": "medium", "high": "high"}

    # ------------------------------------------------------------------

    def _build_client(self, api_key: str, client_options: dict) -> OpenAIClient:
        """
        Construct an :class:`~clients.OpenAIClient`.

        Supported *client_options* keys: ``url``, ``timeout``,
        ``retries``, ``proxy``.
        """
        return OpenAIClient(
            api_key = api_key,
            url     = client_options.get("url"),
            timeout = client_options.get("timeout", DEFAULT_TIMEOUT),
            retries = client_options.get("retries", DEFAULT_RETRIES),
            proxy   = client_options.get("proxy"),
        )

    def to_request(self, messages: list, output: dict) -> "tuple[str, dict]":
        """
        Translate universal messages into the correct OpenAI request format.

        Routing
        -------
        * Models whose name starts with ``"gpt-5"`` are routed to the
          **Responses API** (``POST /v1/responses``).  These models are not
          available on Chat Completions and return HTTP 403 if called there.
        * Image models (``dall-e-*``, ``gpt-image-*``) are routed to the
          **Images Generations API** (``POST /v1/images/generations``).
        * All other models use the **Chat Completions API**
          (``POST /v1/chat/completions``).

        When ``reasoning`` is set the corresponding ``reasoning_effort``
        value from ``_REASONING_MAP`` is added to Chat Completions requests.
        o-series models use this to control reasoning depth; GPT models
        ignore the parameter.
        """
        if _should_use_responses_api(self.name):
            return _build_responses_api_request(self, messages, output)

        if _is_openai_image_model(self.name):
            return _build_image_generations_request(self, messages, output)

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
        Extract the clean result from an OpenAI response.

        Detects the API automatically:
        * ``"data"``    present → Images Generations response
        * ``"choices"`` present → Chat Completions response
        * ``"output"``  present → Responses API response
        """
        if "data" in response:
            return _parse_image_generations_response(response)
        if "choices" in response:
            return _parse_openai_compat_response(response, output)
        return _parse_responses_api_response(response, output)
