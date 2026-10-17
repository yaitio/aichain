"""Krea, its own family: a Bearer token, a path per model, and an asynchronous
submit → poll → download behind the ``send()`` seam that synthesises the
standard ``{"data": [{"b64_json"}]}`` shape, so the shared image parser reads
it. An input image makes an edit. See docs/tools/media.md.
"""

from __future__ import annotations

import base64
import json as _json
import time

from ._shape import pixels
from .._base import BaseClient
from .._errors import TaskFailedError
from ._openai_compat import (_edit_sources, _image_source_to_data_uri,
                             _last_user_text, _parse_image_generations_response)

#: A job's statuses that end it without an image.
_FAILED = frozenset({"failed", "cancelled"})


def _pixels_of(uri: str) -> tuple[int, int] | None:
    """A data URI's image size from its header — PNG, JPEG or WebP — or None."""
    if not isinstance(uri, str) or not uri.startswith("data:") or "," not in uri:
        return None
    try:
        raw = base64.b64decode(uri.split(",", 1)[1][:65536] + "==", validate=False)
    except ValueError:
        return None
    if raw[:8] == b"\x89PNG\r\n\x1a\n" and len(raw) >= 24:
        return int.from_bytes(raw[16:20], "big"), int.from_bytes(raw[20:24], "big")
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP" and raw[12:16] == b"VP8X" and len(raw) >= 30:
        return 1 + int.from_bytes(raw[24:27], "little"), 1 + int.from_bytes(raw[27:30], "little")
    if raw[:2] == b"\xff\xd8":
        i = 2
        while i + 9 < len(raw):
            if raw[i] != 0xFF:
                i += 1
                continue
            marker, length = raw[i + 1], int.from_bytes(raw[i + 2:i + 4], "big")
            if marker in (0xC0, 0xC1, 0xC2):
                return int.from_bytes(raw[i + 7:i + 9], "big"), int.from_bytes(raw[i + 5:i + 7], "big")
            i += 2 + length
    return None


def _value(ratio: str) -> float:
    a, _, b = ratio.partition(":")
    return float(a) / float(b)


def _nearest(target: float, ratios) -> str:
    return min(ratios, key=lambda r: abs(_value(r) - target))


class KreaClient(BaseClient):
    """Krea — Krea 2 generation and Krea 3 editing, as asynchronous jobs."""

    _POLL_INTERVAL = 2.0     # Krea's own advice is every 2 to 5 seconds
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
        raise NotImplementedError("Krea's model list is its documentation; see the registry.")

    # ── format: universal → Krea body ────────────────────────────────
    def build_request(self, messages, output, params, tools=None) -> tuple[str, dict]:
        from ...models._adaptation import Adaptation, ADAPTED, record
        name  = params["name"]
        model = self._data["models"][name]
        fmt   = output.get("format", {})
        ratios = model["aspect_ratios"]
        body: dict = {"prompt": _last_user_text(messages)}

        ratio = fmt.get("aspect_ratio")
        if ratio and ratio not in ratios:
            try:
                near = _nearest(_value(ratio), ratios)
            except (ValueError, ZeroDivisionError):
                raise ValueError(f"{name}: aspect_ratio {ratio!r} is not a ratio; it takes "
                                 f"{', '.join(ratios)}") from None
            record(Adaptation(kind=ADAPTED, option="aspect_ratio", asked=ratio, sent=near,
                              model=name, why=f"this model takes {', '.join(ratios)}; the "
                                              "nearest was sent"))
            ratio = near
        wh = pixels(fmt.get("size")) if fmt.get("size") else None
        if not ratio and wh:
            ratio = _nearest(wh[0] / wh[1], ratios)
            record(Adaptation(kind=ADAPTED, option="size", asked=fmt["size"],
                              sent=f"aspect_ratio={ratio}", model=name,
                              why="this provider takes a ratio and a resolution tier, not "
                                  "pixels; the shape was kept"))
        if ratio or model.get("ratio_required"):
            body["aspect_ratio"] = ratio or "1:1"   # Krea 2 needs one; Krea 3 frames by itself
        tiers = model["resolutions"]
        tier = tiers[0]
        if wh and len(tiers) > 1 and max(wh) > 1536:
            tier = tiers[-1]
        body["resolution"] = tier
        if fmt.get("seed") is not None:
            body["seed"] = fmt["seed"]

        if not model.get("edit_field"):
            if _edit_sources(messages, self._provider, name):
                raise ValueError(f"{name} generates and does not edit: given an image it "
                                 "redraws it rather than following the instruction. Edit with "
                                 "krea-3, or send image_url and strength through extra")
            return model["path"], body
        sources = _edit_sources(messages, self._provider, name, limit=model.get("max_input_images"))
        if sources:
            uris = [_image_source_to_data_uri(s) for s in sources]
            if model.get("edit_field") == "image_urls":
                body["image_urls"] = uris
            else:
                body["image_url"] = uris[0]
        return model["path"], body

    def parse_response(self, response, output) -> dict:
        return _parse_image_generations_response(response)

    # ── request lifecycle: submit → poll → download ──────────────────
    def _price(self, path: str, body: dict) -> float | None:
        """What Krea charges for this request, from its price list — a job
        reports no cost of its own."""
        model = next((m for m in self._data["models"].values() if m.get("path") == path), None)
        if model is None:
            return None
        tiers = model.get("tiers")
        if not tiers:
            return (model.get("price") or {}).get("per_image")
        cost = tiers["output"].get(str(body.get("resolution", "1K")).upper(), max(tiers["output"].values()))
        for uri in body.get("image_urls") or []:
            wh = _pixels_of(uri)
            cost += tiers["reference"]["1K" if wh and max(wh) <= 1536 else "2K"]
        if body.get("thinking_mode") == "high":
            cost += tiers.get("thinking_high", 0.0)
        return round(cost, 4)

    def send(self, path: str, body: dict, headers: dict) -> bytes:
        job = _json.loads(self._post(path, body, headers))
        job_id = job.get("job_id")
        if not job_id:
            raise TaskFailedError(502, f"Krea returned no job_id: {job}")
        url = self._poll(job_id, headers)
        blob = self._download(url)
        b64 = base64.b64encode(blob["data"]).decode("ascii")
        out: dict = {"data": [{"b64_json": b64}]}
        if (cost := self._price(path, body)) is not None:
            out["usage"] = {"cost": cost}
        return _json.dumps(out).encode("utf-8")

    def _poll(self, job_id: str, headers: dict) -> str:
        """Poll ``/jobs/{id}`` until it completes; the first result URL."""
        deadline = time.monotonic() + self._POLL_TIMEOUT
        while True:
            job = _json.loads(self._get(f"/jobs/{job_id}", headers))
            status = job.get("status")
            if status == "completed":
                urls = (job.get("result") or {}).get("urls") or []
                first = urls[0] if isinstance(urls, list) and urls else None
                first = first.get("url") if isinstance(first, dict) else first
                if not first:
                    raise TaskFailedError(502, f"Krea job {job_id} completed with no image: {job}")
                return first
            if status in _FAILED:
                error = job.get("error") or (job.get("result") or {}).get("error") or {}
                detail = error.get("message") or error.get("code") if isinstance(error, dict) else error
                raise TaskFailedError(502, f"Krea job {status}: {detail or job}")
            if time.monotonic() >= deadline:
                raise TaskFailedError(504, f"Krea job {job_id} did not finish within "
                                           f"{self._POLL_TIMEOUT:.0f}s (last status: {status}).")
            time.sleep(self._POLL_INTERVAL)
