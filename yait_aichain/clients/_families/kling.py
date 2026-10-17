"""Kling's images, its own family: a Bearer token and an asynchronous submit →
poll → download behind the ``send()`` seam, reshaped into the standard
``{"data": [{"b64_json"}]}`` the shared image parser reads. An input image
makes an image-to-image. Video is video()'s, in tools/video/_jobs.py.
See docs/tools/media.md.
"""

from __future__ import annotations

import base64
import json as _json
import time

from ._shape import pixels
from .._base import BaseClient
from .._errors import TaskFailedError
from ._openai_compat import _edit_sources, _last_user_text, _parse_image_generations_response

_PATH = "/v1/images/generations"


def _value(ratio: str) -> float:
    a, _, b = ratio.partition(":")
    return float(a) / float(b)


class KlingClient(BaseClient):
    """Kling Image 3.0 and 2.1, as asynchronous jobs."""

    _POLL_INTERVAL = 2.0
    _POLL_TIMEOUT  = 300.0

    def __init__(self, api_key: str, *, data: dict, **client_opts) -> None:
        prov = data["provider"]
        super().__init__(
            api_key,
            url=client_opts.get("url") or prov.get("base_url"),
            **{k: client_opts[k] for k in ("timeout", "retries", "proxy") if k in client_opts},
        )
        self._data     = data
        self._provider = prov["key"]

    def list_models(self) -> list[str]:
        raise NotImplementedError("Kling's model list is its documentation; see the registry.")

    # ── format: universal → Kling body ───────────────────────────────
    def build_request(self, messages, output, params, tools=None) -> tuple[str, dict]:
        from ...models._adaptation import Adaptation, ADAPTED, record
        name  = params["name"]
        model = self._data["models"][name]
        fmt   = output.get("format", {})
        ratios = model["aspect_ratios"]
        body: dict = {"model_name": name, "prompt": _last_user_text(messages), "n": 1}
        asked = fmt.get("aspect_ratio")
        wh = pixels(fmt.get("size")) if fmt.get("size") else None
        if not asked and wh:
            asked = f"{wh[0]}:{wh[1]}"
        if asked:
            try:
                near = min(ratios, key=lambda r: abs(_value(r) - _value(asked)))
            except (ValueError, ZeroDivisionError):
                raise ValueError(f"{name}: aspect_ratio {asked!r} is not a ratio; it takes {', '.join(ratios)}") from None
            if abs(_value(near) - _value(asked)) > 0.01:
                record(Adaptation(kind=ADAPTED, option="size" if wh and not fmt.get("aspect_ratio") else "aspect_ratio",
                                  asked=fmt.get("aspect_ratio") or fmt.get("size"), sent=near, model=name,
                                  why=f"this model takes {', '.join(ratios)}; the nearest was sent"))
            body["aspect_ratio"] = near
        if wh and max(wh) > 1536:
            body["resolution"] = "2k"
        sources = _edit_sources(messages, self._provider, name, limit=1)
        if sources:
            # Base64 with no data: prefix — Kling refuses the prefixed form.
            src = sources[0]
            body["image"] = src["url"] if src.get("kind") == "url" else src["data"]
        return _PATH, body

    def parse_response(self, response, output) -> dict:
        return _parse_image_generations_response(response)

    # ── request lifecycle: submit → poll → download ──────────────────
    def send(self, path: str, body: dict, headers: dict) -> bytes:
        made = _json.loads(self._post(path, body, headers))
        task_id = (made.get("data") or {}).get("task_id")
        if made.get("code") not in (0, None) or not task_id:
            raise TaskFailedError(502, f"Kling refused the task: {made.get('message') or made}")
        url = self._poll(task_id, headers)
        blob = self._download(url)
        out: dict = {"data": [{"b64_json": base64.b64encode(blob["data"]).decode("ascii")}]}
        price = (self._data["models"].get(body.get("model_name")) or {}).get("price") or {}
        if (cost := price.get("per_image")) is not None:
            out["usage"] = {"cost": cost * (2 if body.get("image") and body.get("model_name") == "kling-v2-1" else 1)}
        return _json.dumps(out).encode("utf-8")

    def _poll(self, task_id: str, headers: dict) -> str:
        deadline = time.monotonic() + self._POLL_TIMEOUT
        while True:
            task = (_json.loads(self._get(f"{_PATH}/{task_id}", headers)).get("data") or {})
            status = task.get("task_status")
            if status == "succeed":
                images = (task.get("task_result") or {}).get("images") or []
                if not images or not images[0].get("url"):
                    raise TaskFailedError(502, f"Kling task {task_id} succeeded with no image: {task}")
                return images[0]["url"]
            if status == "failed":
                raise TaskFailedError(502, f"Kling task failed: {task.get('task_status_msg') or task}")
            if time.monotonic() >= deadline:
                raise TaskFailedError(504, f"Kling task {task_id} did not finish within "
                                           f"{self._POLL_TIMEOUT:.0f}s (last status: {status}).")
            time.sleep(self._POLL_INTERVAL)
