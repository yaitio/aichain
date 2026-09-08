"""
clients._families.recraft
==========================

Recraft is image-only and rides the OpenAI Bearer transport (inherited from
``OpenAIClient``), but its endpoints differ from OpenAI's:

* text-to-image  → ``POST /v1/images/generations``  (OpenAI-shaped JSON)
* image-to-image → ``POST /v1/images/imageToImage``  (multipart/form-data)

The multipart edit goes out through the ``send()`` seam already implemented on
``OpenAIClient`` (it turns a ``{"_multipart": True, "fields": [...]}`` body into
a form POST), and the ``{"data": [...]}`` response is parsed by the shared image
parser — so this client is just the request shaping.

Verified live against the Recraft API (external.api.recraft.ai).
"""

from __future__ import annotations

import base64

from ._openai_compat import (
    _detect_image_mime,
    _image_sources,
    _messages_have_image,
    _parse_image_generations_response,
    _prompt_from_messages,
)
from .openai import OpenAIClient

# Recraft request fields (beyond prompt/size) passed straight through from
# ``output["format"]`` when present.
_GEN_PASSTHROUGH  = ("style", "style_id", "negative_prompt", "controls", "text_layout")
_EDIT_PASSTHROUGH = ("style", "style_id", "negative_prompt")

#: Default edit strength when the caller does not specify one (Recraft requires it).
_DEFAULT_STRENGTH = 0.2


def _size_from_ratio(ratio: str, short: int = 1024) -> "str | None":
    """'16:9' → '1820x1024'. The short edge is pinned and the long one
    follows, which is the shape the caller asked for at this provider's
    working resolution."""
    a, _, b = ratio.partition(":")
    if not (a.isdigit() and b.isdigit() and int(a) and int(b)):
        return None
    w, h = int(a), int(b)
    if w >= h:
        return f"{round(short * w / h)}x{short}"
    return f"{short}x{round(short * h / w)}"


def _build_recraft_generation_request(name, messages, output, path):
    """OpenAI-shaped text-to-image body (Recraft returns ``{"data": [...]}``)."""
    fmt = output.get("format", {})
    body = {
        "model":           name,
        "prompt":          _prompt_from_messages(messages),
        "n":               1,
        "response_format": "b64_json",
    }
    from ...models._adaptation import Adaptation, ADAPTED, record
    if fmt.get("size"):
        body["size"] = fmt["size"]
    elif fmt.get("aspect_ratio"):
        # This API takes pixels only — an aspect ratio is accepted and
        # ignored, measured 2026-09-09: "16:9" came back 1024x1024. So the
        # ratio is turned into a size at a 1024 short edge and the provider
        # snaps it to its own nearest (1707x1024 was returned as 1820x1024).
        size = _size_from_ratio(fmt["aspect_ratio"])
        if size:
            body["size"] = size
            record(Adaptation(
                kind=ADAPTED, option="aspect_ratio",
                asked=fmt["aspect_ratio"], sent=f"size={size}", model=name,
                why="this provider takes pixels, not a ratio; the shape was "
                    "kept at a 1024 short edge and it picks the nearest it "
                    "renders"))
    for k in _GEN_PASSTHROUGH:
        if fmt.get(k) is not None:
            body[k] = fmt[k]
    return path, body


def _build_recraft_edit_request(name, messages, output, path):
    """Multipart imageToImage body — sentinel consumed by ``OpenAIClient.send``."""
    sources = _image_sources(messages)
    if not sources:
        raise ValueError("image edit requires at least one input image part")
    src = sources[0]
    if src.get("kind") == "url":
        raise ValueError(
            "Recraft imageToImage needs binary image data, not a URL — "
            "pass a base64 or file source."
        )
    raw  = base64.b64decode(src["data"])
    mime = src.get("mime", "image/png")
    ext  = mime.split("/")[-1]

    fmt = output.get("format", {})
    fields = [
        ("model",           name),
        ("prompt",          _prompt_from_messages(messages)),
        ("strength",        str(fmt.get("strength", _DEFAULT_STRENGTH))),
        ("response_format", "b64_json"),
    ]
    for k in _EDIT_PASSTHROUGH:
        if fmt.get(k) is not None:
            fields.append((k, str(fmt[k])))
    fields.append(("image", (f"image.{ext}", raw, mime)))

    return path, {"_multipart": True, "fields": fields}


def _build_recraft_vectorize_request(messages, output, path):
    """
    Multipart vectorize body — raster → SVG trace. No model, no prompt: the input
    image is converted as-is (distinct from imageToImage, which is a content
    *variation*). Sentinel consumed by ``OpenAIClient.send`` → ``_post_form``.
    """
    sources = _image_sources(messages)
    if not sources:
        raise ValueError("vectorize requires exactly one input image")
    src = sources[0]
    if src.get("kind") == "url":
        raise ValueError(
            "Recraft vectorize needs binary image data, not a URL — "
            "pass a base64 or file source."
        )
    raw  = base64.b64decode(src["data"])
    mime = src.get("mime", "image/png")
    ext  = mime.split("/")[-1]

    fmt = output.get("format", {})
    fields = [
        ("response_format", fmt.get("response_format", "b64_json")),
        ("image", (f"image.{ext}", raw, mime)),
    ]
    return path, {"_multipart": True, "fields": fields}


def _parse_vectorize_response(response: dict) -> dict:
    """
    Parse a Recraft vectorize response into the standard
    ``{url, base64, mime_type, revised_prompt}`` shape.

    Vectorize returns a single object ``{"image": {"b64_json"|"url", ...},
    "credits": N}`` — NOT the ``{"data": [...]}`` list shape of generation
    (verified live, 2026-06-30). The result is always SVG.
    """
    img = response.get("image")
    if not isinstance(img, dict):
        raise ValueError(
            f"Recraft vectorize: unexpected response shape; keys {sorted(response)}"
        )
    b64 = img.get("b64_json")
    if b64:
        return {"url": None, "base64": b64,
                "mime_type": _detect_image_mime(b64) or "image/svg+xml",
                "revised_prompt": None}
    if img.get("url"):
        return {"url": img["url"], "base64": None,
                "mime_type": "image/svg+xml", "revised_prompt": None}
    raise ValueError("Recraft vectorize: no image data in response")


class RecraftClient(OpenAIClient):
    """Recraft image generation + editing + vectorize (Bearer transport, image-only)."""

    def __init__(self, api_key: str, *, data: dict, **client_opts) -> None:
        super().__init__(api_key, data=data, **client_opts)
        self._vectorize_path = data["provider"].get(
            "images_vectorize_path", "/v1/images/vectorize")

    def build_request(self, messages, output, params) -> "tuple[str, dict]":
        name = params["name"]
        if name == "recraft-vectorize":
            return _build_recraft_vectorize_request(messages, output, self._vectorize_path)
        if _messages_have_image(messages):
            return _build_recraft_edit_request(name, messages, output, self._images_edits_path)
        return _build_recraft_generation_request(name, messages, output, self._images_path)

    def parse_response(self, response, output) -> dict:
        # Vectorize returns {"image": {...}}; generation/edit return {"data": [...]}.
        if isinstance(response.get("image"), dict):
            return _parse_vectorize_response(response)
        return _parse_image_generations_response(response)
