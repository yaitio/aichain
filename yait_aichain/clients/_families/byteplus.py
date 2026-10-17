"""BytePlus ModelArk: the Seed models through the OpenAI-compatible family, and
Seedream images through its own request — a pixel size inside the model's range,
images as data: URIs, the watermark off — priced by the size that came back.
Seedance video is video()'s, in tools/video/_jobs.py. See docs/tools/media.md.
"""

from __future__ import annotations

import json as _json
from math import sqrt

from ._shape import pixels
from ._openai_compat import _edit_sources, _image_source_to_data_uri, _last_user_text
from .openai import OpenAIClient


def _value(ratio: str) -> float:
    a, _, b = str(ratio).partition(":")
    return float(a) / float(b)


def _fit(w: float, h: float, low: int, high: int) -> tuple[int, int]:
    """A width and a height of the same shape whose product is inside [low, high],
    each a multiple of 16 — rounded inward so the range still holds."""
    scale = 1.0
    if w * h < low:
        scale = sqrt(low / (w * h))
    elif w * h > high:
        scale = sqrt(high / (w * h))
    w, h = w * scale, h * scale
    down = (max(16, int(w) // 16 * 16), max(16, int(h) // 16 * 16))
    if down[0] * down[1] >= low:
        return down
    return -(-int(w) // 16) * 16, -(-int(h) // 16) * 16


class BytePlusClient(OpenAIClient):
    """Chat as the OpenAI-compatible family describes it; Seedream as its own."""

    def _model_data(self, name: str) -> dict:
        return self._data.get("models", {}).get(name) or {}

    def build_request(self, messages, output, params, tools=None) -> tuple[str, dict]:
        name = params["name"]
        if "pixels" not in self._model_data(name):
            return super().build_request(messages, output, params, tools)
        if tools:
            raise ValueError(f"model {name!r} is an image model — it cannot take tool declarations")
        return self._images_path, self._image_body(name, messages, output)

    def _image_body(self, name: str, messages, output) -> dict:
        from ...models._adaptation import Adaptation, ADAPTED, TRANSLATED, record
        model = self._model_data(name)
        low, high = model["pixels"]
        fmt = output.get("format", {})
        body: dict = {"model": name, "prompt": _last_user_text(messages),
                      "response_format": "b64_json", "watermark": False}
        if model.get("sequential"):
            body["sequential_image_generation"] = "disabled"
        wh, ratio = pixels(fmt.get("size")) if fmt.get("size") else None, fmt.get("aspect_ratio")
        if ratio and not wh:
            try:
                r = _value(ratio)
            except (ValueError, ZeroDivisionError):
                raise ValueError(f"{name}: aspect_ratio {ratio!r} is not a ratio like '16:9'") from None
            area = model.get("default_pixels") or low
            w, h = _fit(sqrt(area * r), sqrt(area / r), low, high)
            record(Adaptation(kind=TRANSLATED, option="aspect_ratio", asked=ratio, sent=f"size={w}x{h}",
                              model=name, why="this API takes pixels; the ratio was drawn at the model's base size"))
            body["size"] = f"{w}x{h}"
        elif wh:
            w, h = _fit(wh[0], wh[1], low, high)
            if (w, h) != wh:
                record(Adaptation(kind=ADAPTED, option="size", asked=fmt["size"], sent=f"{w}x{h}", model=name,
                                  why=f"this model makes {low:,} to {high:,} pixels; the shape was kept "
                                      "and the size scaled into that range"))
            body["size"] = f"{w}x{h}"
        elif model.get("default_size"):
            body["size"] = model["default_size"]
        sources = _edit_sources(messages, self._provider, name)
        if sources:
            uris = [_image_source_to_data_uri(s) for s in sources]
            body["image"] = uris[0] if len(uris) == 1 else uris
        return body

    def send(self, path: str, body: dict, headers: dict) -> bytes:
        raw = super().send(path, body, headers)
        if path != self._images_path:
            return raw
        # Priced by the picture that came back: Seedream 5.0 pro costs twice above
        # a size, and a reference image after the first is charged.
        made = _json.loads(raw)
        model = self._model_data(body.get("model", ""))
        each = (model.get("price") or {}).get("per_image")
        if each is not None and made.get("data"):
            wh = pixels((made["data"][0] or {}).get("size")) or (0, 0)
            large = model.get("price_large") or {}
            if large and wh[0] * wh[1] > large["pixels"]:
                each = large["per_image"]
            refs = body.get("image") or []
            extra = max(0, (len(refs) if isinstance(refs, list) else 1) - 1)
            each += extra * (model.get("price_extra_input") or 0)
            made.setdefault("usage", {})["cost"] = round(each * len(made["data"]), 6)
        return _json.dumps(made).encode("utf-8")
