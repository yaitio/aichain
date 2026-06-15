"""
clients._families._openai_compat
================================

Wire-format helpers for the OpenAI Chat Completions / Responses / Images
APIs — shared by every OpenAI-compatible provider (openai, xai, perplexity,
kimi, deepseek, qwen).  Pure functions: universal format ↔ provider JSON.
"""

from __future__ import annotations

import json


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
    max_tokens_field: str = "max_completion_tokens",
) -> "tuple[str, dict]":
    """
    Build an OpenAI-compatible ``(path, body)`` pair.

    Parameters
    ----------
    model    : Provider model instance (OpenAI / xAI / Perplexity / …).
    messages : Substituted universal messages list.
    output   : Universal output spec.
    path     : Provider-specific endpoint path.
    max_tokens_field : Request key for the output-token limit
                       (``"max_completion_tokens"`` default, ``"max_tokens"``
                       for Kimi/DeepSeek).
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
        "model":          model.name,
        "messages":       openai_messages,
        max_tokens_field: model.max_tokens,
        "temperature":    model.temperature,
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

    Shared by the openai, xai and perplexity providers (and, via their own
    request builders, the kimi/deepseek/qwen providers).

    Raises
    ------
    ValueError
        With a descriptive message when the response carries no usable
        content: empty ``choices`` (blocked/failed upstream), an explicit
        model ``refusal``, or invalid/truncated JSON in a JSON mode.
    """
    choices = response.get("choices") or []
    if not choices:
        detail = (response.get("error") or {}).get("message")
        raise ValueError(
            "Provider response contains no choices — the request was "
            "blocked or failed upstream"
            + (f": {detail}" if detail else ".")
        )

    choice  = choices[0] or {}
    message = choice.get("message") or {}

    refusal = message.get("refusal")
    if refusal:
        raise ValueError(f"Model refused to answer: {refusal}")

    text  = message.get("content") or ""
    ftype = output.get("format", {}).get("type", "text")
    if ftype in ("json", "json_schema"):
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            if choice.get("finish_reason") == "length":
                raise ValueError(
                    "Response was truncated (finish_reason='length') before "
                    "the JSON completed — increase max_tokens."
                ) from exc
            raise ValueError(
                f"Model returned invalid JSON: {exc}"
            ) from exc
    return text


# ---------------------------------------------------------------------------
# Image generation helpers  (for dall-e-* and gpt-image-* models)
# ---------------------------------------------------------------------------

_OPENAI_IMAGE_MODEL_PREFIXES = ("dall-e-", "gpt-image-", "chatgpt-image-")


def _is_openai_image_model(name: str) -> bool:
    """Return True when *name* identifies an OpenAI image-generation model."""
    return name.startswith(_OPENAI_IMAGE_MODEL_PREFIXES)


# o-series reasoning models reject the sampling parameters that ordinary
# GPT models accept.
_OSERIES_FAMILIES = ("o1", "o3", "o4")


def _is_o_series_model(name: str) -> bool:
    """True for o-series reasoning models (o1, o1-mini, o3, o4-mini, …)."""
    return any(
        name == fam or name.startswith(f"{fam}-") for fam in _OSERIES_FAMILIES
    )


def _build_image_generations_request(
    model,
    messages: list,
    output:   dict,
    path:     str = "/v1/images/generations",
) -> "tuple[str, dict]":
    """
    Build an OpenAI ``/v1/images/generations`` ``(path, body)`` pair.

    Shared by the openai and xai providers.

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
    # gpt-image-* accepts a transparent background and an explicit file format
    # (``png``/``webp`` keep the alpha channel; ``jpeg`` would flatten it).
    if fmt.get("background"):
        body["background"] = fmt["background"]
    if fmt.get("output_format"):
        body["output_format"] = fmt["output_format"]

    # Request base64 output on every model that accepts the parameter.
    # gpt-image-* / chatgpt-image-* always return b64_json natively and reject
    # the param — omit it only for those families (dall-e-* still needs it).
    if not model.name.startswith(("gpt-image-", "chatgpt-image-")):
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

    Shared by the openai and xai providers.

    Returns a dict with keys ``url``, ``base64``, ``mime_type``, and
    ``revised_prompt`` — consistent with the Google image response format.

    ``mime_type`` is detected from the image's magic bytes so it is always
    accurate, regardless of which provider generated the image.

    Raises
    ------
    ValueError
        When ``data`` is empty (e.g. the prompt was rejected by moderation).
    """
    data = response.get("data") or []
    if not data:
        detail = (response.get("error") or {}).get("message")
        raise ValueError(
            "Provider returned no image data — the prompt may have been "
            "rejected by moderation"
            + (f": {detail}" if detail else ".")
        )
    item = data[0]
    b64  = item.get("b64_json")
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

    # gpt-5-family models are reasoning models: the Responses API rejects
    # ``temperature``/``top_p`` for them ("Unsupported parameter"), so the
    # sampling knobs are deliberately omitted here.
    body: dict = {
        "model":             model.name,
        "input":             input_messages,
        "max_output_tokens": model.max_tokens,
    }

    if instructions:
        body["instructions"] = instructions

    if model.reasoning:
        effort = model._REASONING_MAP.get(model.reasoning)
        if effort:
            body["reasoning"] = {"effort": effort}

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
# openai provider
# ---------------------------------------------------------------------------

