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


def _build_recraft_generation_request(name, messages, output, path):
    """OpenAI-shaped text-to-image body (Recraft returns ``{"data": [...]}``)."""
    fmt = output.get("format", {})
    body = {
        "model":           name,
        "prompt":          _prompt_from_messages(messages),
        "n":               1,
        "response_format": "b64_json",
    }
    if fmt.get("size"):
        body["size"] = fmt["size"]
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


class RecraftClient(OpenAIClient):
    """Recraft image generation + editing (Bearer transport, image-only)."""

    def build_request(self, messages, output, params) -> "tuple[str, dict]":
        name = params["name"]
        if _messages_have_image(messages):
            return _build_recraft_edit_request(name, messages, output, self._images_edits_path)
        return _build_recraft_generation_request(name, messages, output, self._images_path)

    def parse_response(self, response, output) -> dict:
        return _parse_image_generations_response(response)
