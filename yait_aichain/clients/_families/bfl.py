"""Black Forest Labs (FLUX), its own family: an ``x-key`` header, the model name
as the endpoint path, and an asynchronous submit → poll → download behind the
``send()`` seam that synthesises the standard ``{"data": [{"b64_json"}]}``
shape, so the shared image parser reads it. An ``input_image`` makes a FLUX
Kontext edit. See docs/tools/media.md.
"""

from __future__ import annotations

from ._shape import pixels
import base64
import json as _json
import time

from .._base import BaseClient
from .._errors import TaskFailedError
from ._openai_compat import _edit_sources, _parse_image_generations_response, _last_user_text

# BFL polling statuses that are terminal failures (anything but Ready / pending).
_FAILED_STATUSES = frozenset(
    {"Error", "Failed", "Task not found", "Request Moderated", "Content Moderated"}
)


def _flux_image_value(src: dict) -> str:
    """A URL source → its URL; a base64/file source → the raw base64 string."""
    if src.get("kind") == "url":
        return src["url"]
    return src["data"]   # BFL accepts a raw base64 string or a URL


class BFLClient(BaseClient):
    """Black Forest Labs FLUX — async image generation + Kontext editing."""

    #: Seconds between polls, and the overall ceiling.
    _POLL_INTERVAL = 1.5
    _POLL_TIMEOUT  = 300.0

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

    def list_models(self) -> list[str]:
        raise NotImplementedError(
            "BFL has no model-list endpoint; see the registry for known models."
        )

    # ── format: universal → BFL body ─────────────────────────────────
    def build_request(self, messages, output, params, tools=None) -> tuple[str, dict]:
        name = params["name"]
        fmt  = output.get("format", {})

        body: dict = {"prompt": _last_user_text(messages)}

        # Size → width/height for generation; Kontext uses aspect_ratio instead.
        size = fmt.get("size")
        if size and "x" in size and "kontext" not in name.lower():
            wh = pixels(size)
            if wh:
                w, h = wh
                body["width"], body["height"] = w, h
                from ...models._adaptation import Adaptation, ADAPTED, record
                record(Adaptation(
                    kind=ADAPTED, option="size", asked=size,
                    sent=f"width={w}, height={h}", model=name,
                    why="this provider takes the two edges separately"))
        if fmt.get("aspect_ratio"):
            body["aspect_ratio"] = fmt["aspect_ratio"]
        if fmt.get("output_format"):
            body["output_format"] = fmt["output_format"]
        if fmt.get("seed") is not None:
            body["seed"] = fmt["seed"]

        # An input image (Kontext) switches generation → edit. Up to 4 references.
        sources = _edit_sources(messages, "bfl", name)
        if sources:
            body["input_image"] = _flux_image_value(sources[0])
            for i, extra in enumerate(sources[1:], start=2):
                body[f"input_image_{i}"] = _flux_image_value(extra)

        return f"/v1/{name}", body

    def parse_response(self, response, output) -> dict:
        return _parse_image_generations_response(response)

    # ── request lifecycle: submit → poll → download ──────────────────
    def send(self, path: str, body: dict, headers: dict) -> bytes:
        submitted   = _json.loads(self._post(path, body, headers))
        polling_url = submitted.get("polling_url")
        if not polling_url:
            raise TaskFailedError(502, f"BFL returned no polling_url: {submitted}")

        result_url = self._poll(polling_url, headers)
        blob = self._download(result_url)
        b64  = base64.b64encode(blob["data"]).decode("ascii")
        return _json.dumps({"data": [{"b64_json": b64}]}).encode("utf-8")

    def _poll(self, polling_url: str, headers: dict) -> str:
        """Poll the (absolute) polling URL until Ready; return result.sample URL."""
        deadline = time.monotonic() + self._POLL_TIMEOUT
        while True:
            payload = _json.loads(self._download(polling_url, headers,
                                                 timeout=self._poll_timeout(deadline))["data"])
            status  = payload.get("status")

            if status == "Ready":
                sample = (payload.get("result") or {}).get("sample")
                if not sample:
                    raise TaskFailedError(502, f"BFL task ready but no result.sample: {payload}")
                return sample
            if status in _FAILED_STATUSES:
                raise TaskFailedError(502, f"BFL task {status}: {payload.get('details') or payload}")

            if time.monotonic() >= deadline:
                raise TaskFailedError(
                    504,
                    f"BFL task did not finish within {self._POLL_TIMEOUT:.0f}s "
                    f"(last status: {status}).",
                )
            time.sleep(self._POLL_INTERVAL)
