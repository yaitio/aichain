"""
clients._families.reve
======================

Reve (api.reve.com) — image generation and editing.

Reve has its own wire shape (not OpenAI-compatible), so this is its own family
client. Bearer auth; the operation is chosen by the input, mapping to one of
three synchronous endpoints:

* **generate** — no input image → ``POST /v1/image/create``
  body ``{prompt, aspect_ratio?, version}``
* **edit** — exactly one input image → ``POST /v1/image/edit``
  body ``{edit_instruction, reference_image (raw base64), aspect_ratio?, version}``
* **remix** — two or more input images → ``POST /v1/image/remix``
  body ``{prompt, reference_images: [raw base64, …], aspect_ratio?, version}``
  (the prompt references images by position with ``<img>0</img>`` tokens)

``version`` defaults to ``"latest"`` (override via ``output.format.version``;
Reve also accepts ``latest-fast`` and pinned ids like ``reve-create@20250915``).
``aspect_ratio`` is forwarded when supplied (Reve options: 1:1, 16:9, 9:16, 3:2,
2:3, 4:3, 3:4). Two optional quality controls pass through from the output format:
``postprocessing`` (list: upscale / remove_background / fit_image / effect) and
``test_time_scaling`` (1–15).

The request is sent with ``Accept: application/json``, so Reve returns a JSON
object with the image base64-encoded (PNG) plus extra properties (``credits_used``,
…). Request shapes are confirmed against the official console docs (api.reve.com).
"""

from __future__ import annotations

import json as _json

from .._base import BaseClient
from ._openai_compat import (
    _image_sources,
    _prompt_from_messages,
    _detect_image_mime,
    _parse_image_generations_response,
)

_GEN_PATH    = "/v1/image/create"
_EDIT_PATH   = "/v1/image/edit"
_REMIX_PATH  = "/v1/image/remix"
_DEFAULT_VERSION = "latest"


def _reve_image_value(src: dict) -> str:
    """A base64/file source → its raw base64 string; a URL source → the URL."""
    if src.get("kind") == "url":
        return src["url"]
    return src["data"]              # Reve reference images are raw base64 (no data: prefix)


def _parse_reve_response(response: dict) -> dict:
    """
    Extract the image from a Reve response into the standard
    ``{url, base64, mime_type, revised_prompt}`` shape (matching the OpenAI/
    Google image parsers).

    In JSON mode (``Accept: application/json``) Reve returns the image
    base64-encoded under ``image``; this parser stays defensive across the other
    plausible field names too. Raises ``ValueError`` on a content violation or an
    unrecognised shape (with the keys, to make a live mismatch obvious).
    """
    if not isinstance(response, dict):
        raise ValueError(f"Reve returned a non-object response: {type(response).__name__}")

    # Moderation / safety rejection.
    if response.get("content_violation") or response.get("violation"):
        raise ValueError("Reve rejected the request (content violation).")

    # OpenAI-ish data: [{b64_json|url}] — reuse the shared parser if present.
    if isinstance(response.get("data"), list) and response["data"]:
        return _parse_image_generations_response(response)

    # Reve's native shape: a base64 string under one of these keys.
    for key in ("image", "image_base64", "b64_json", "base64"):
        val = response.get(key)
        if isinstance(val, str) and val:
            return {"url": None, "base64": val,
                    "mime_type": _detect_image_mime(val), "revised_prompt": None}

    # A list of images.
    images = response.get("images")
    if isinstance(images, list) and images:
        first = images[0]
        if isinstance(first, str):
            return {"url": None, "base64": first,
                    "mime_type": _detect_image_mime(first), "revised_prompt": None}
        if isinstance(first, dict):
            b64 = first.get("b64_json") or first.get("image") or first.get("base64")
            if b64:
                return {"url": None, "base64": b64,
                        "mime_type": _detect_image_mime(b64), "revised_prompt": None}
            if first.get("url"):
                return {"url": first["url"], "base64": None,
                        "mime_type": None, "revised_prompt": None}

    # A bare URL.
    for key in ("url", "image_url"):
        if isinstance(response.get(key), str):
            return {"url": response[key], "base64": None,
                    "mime_type": None, "revised_prompt": None}

    raise ValueError(
        f"Could not find an image in the Reve response; keys: {sorted(response)}"
    )


class ReveClient(BaseClient):
    """Reve image generation + editing (create / edit / remix)."""

    def __init__(self, api_key: str, *, data: dict, **client_opts) -> None:
        prov = data["provider"]
        super().__init__(
            api_key,
            url=client_opts.get("url") or prov.get("base_url"),
            **{k: client_opts[k] for k in ("timeout", "retries", "proxy")
               if k in client_opts},
        )
        self._data     = data
        self._provider = prov["key"]

    # ── transport ────────────────────────────────────────────────────
    def _auth_headers(self) -> dict:
        return {"Content-Type": "application/json",
                "Accept":       "application/json",
                "Authorization": f"Bearer {self.api_key}"}

    def list_models(self) -> list[str]:
        raise NotImplementedError(
            "Reve has no model-list endpoint; see the registry for known models."
        )

    # ── format: universal → Reve body ────────────────────────────────
    def build_request(self, messages, output, params) -> "tuple[str, dict]":
        fmt     = output.get("format", {})
        version = fmt.get("version", _DEFAULT_VERSION)
        prompt  = _prompt_from_messages(messages)
        sources = _image_sources(messages)

        from ...models._adaptation import (Adaptation, ADAPTED, TRANSLATED,
                                            record)
        common: dict = {"version": version}
        if fmt.get("aspect_ratio"):
            common["aspect_ratio"] = fmt["aspect_ratio"]
            record(Adaptation(
                kind=TRANSLATED, option="aspect_ratio",
                asked=fmt["aspect_ratio"], sent="aspect_ratio",
                model=params["name"], why="passed through"))
        elif fmt.get("size"):
            # This API rejects a pixel size outright — `width`/`height` come
            # back UNRECOGNIZED_PARAMETER — so the shape is kept and the
            # pixels left to it.
            from math import gcd
            w, _, h = str(fmt["size"]).partition("x")
            if w.isdigit() and h.isdigit() and int(h):
                g = gcd(int(w), int(h)) or 1
                ratio = f"{int(w)//g}:{int(h)//g}"
                common["aspect_ratio"] = ratio
                record(Adaptation(
                    kind=ADAPTED, option="size", asked=fmt["size"],
                    sent=f"aspect_ratio={ratio}", model=params["name"],
                    why="this API refuses a pixel size; the shape was kept "
                        "and the pixels left to the provider"))
        # Universal quality and background, brought to this provider's own
        # scale: a number from 1 to 15, and an entry in the postprocessing
        # list. The tables live in reve.toml, not here, so the translation can
        # be read without reading the client.
        from ...models._options import to_provider_value
        if fmt.get("quality"):
            sent, note = to_provider_value("quality", fmt["quality"], "reve")
            if isinstance(sent, int):
                common["test_time_scaling"] = sent
                record(Adaptation(
                    kind=ADAPTED, option="quality", asked=fmt["quality"],
                    sent=f"test_time_scaling={sent}", model=params["name"],
                    why=note or "this provider measures quality as compute"))
        if fmt.get("background"):
            sent, _ = to_provider_value("background", fmt["background"], "reve")
            if sent != fmt["background"]:
                common["postprocessing"] = list(
                    common.get("postprocessing", [])) + [sent]
                record(Adaptation(
                    kind=ADAPTED, option="background", asked=fmt["background"],
                    sent=f"postprocessing={sent}", model=params["name"],
                    why="this provider has no background setting; it is a "
                        "post-processing step here"))

        # Optional quality controls, forwarded only when supplied.
        if fmt.get("postprocessing"):
            common["postprocessing"] = list(
                common.get("postprocessing", [])) + list(fmt["postprocessing"])
        if fmt.get("test_time_scaling") is not None:
            common["test_time_scaling"] = fmt["test_time_scaling"]

        if not sources:
            # generate
            return _GEN_PATH, {"prompt": prompt, **common}

        if len(sources) == 1:
            # edit (single reference image)
            return _EDIT_PATH, {
                "edit_instruction": prompt,
                "reference_image":  _reve_image_value(sources[0]),
                **common,
            }

        # remix (two or more reference images; prompt uses <img>N</img> tokens)
        return _REMIX_PATH, {
            "prompt":           prompt,
            "reference_images": [_reve_image_value(s) for s in sources],
            **common,
        }

    def parse_response(self, response, output) -> dict:
        return _parse_reve_response(response)
