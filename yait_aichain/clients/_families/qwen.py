"""
clients._families.qwen
======================

Qwen (DashScope) rides the OpenAI Chat Completions *format* for text and vision
(inherited from OpenAIClient, incl. its qwen quirk branch) but resolves its base
URL by region.

Text-to-image is the exception: DashScope does **not** serve the wanx image
models through the OpenAI-compatible endpoint.  They run only on DashScope's
native *asynchronous* task API — submit a job, poll until it finishes, then
download the rendered image.  This client implements that flow behind the
``send()`` seam and synthesises a response in the same ``{"data": [...]}`` shape
the synchronous image path returns, so ``parse_response`` is reused unchanged.

    submit  POST /api/v1/services/aigc/text2image/image-synthesis
            (header X-DashScope-Async: enable)  → { output.task_id }
    poll    GET  /api/v1/tasks/{task_id}        → until task_status terminal
    collect download each result URL → base64 → { "data": [{"b64_json": …}] }
"""

from __future__ import annotations

import base64
import json as _json
import os
import time

from .._errors import TaskFailedError
from .openai import OpenAIClient, _is_qwen_image, _last_user_text

REGION_URLS = {
    "ap": "https://dashscope-intl.aliyuncs.com",
    "us": "https://dashscope-us.aliyuncs.com",
    "cn": "https://dashscope.aliyuncs.com",
    "hk": "https://cn-hongkong.dashscope.aliyuncs.com",
}
_DEFAULT_REGION = "ap"

# DashScope native image-synthesis endpoints (relative to the regional base).
_IMAGE_SYNTHESIS_PATH = "/api/v1/services/aigc/text2image/image-synthesis"
_TASKS_PATH = "/api/v1/tasks/"


def resolve_qwen_base_url(region: "str | None" = None) -> str:
    r = (region or os.environ.get("DASHSCOPE_REGION") or _DEFAULT_REGION).lower().strip()
    if r not in REGION_URLS:
        raise ValueError(
            f"Unknown DashScope region {r!r}. "
            f"Valid regions: {', '.join(sorted(REGION_URLS))}."
        )
    return REGION_URLS[r]


def _build_qwen_image_request(model, messages: list, output: dict) -> "tuple[str, dict]":
    """
    Build the DashScope native text2image *submit* ``(path, body)`` pair.

    Size is read from ``output["format"]["size"]`` and converted from our
    ``"1280x720"`` convention to DashScope's ``"1280*720"``.  When absent,
    the parameter is omitted and DashScope applies its own default.
    """
    prompt = _last_user_text(messages)
    fmt: dict = output.get("format", {})

    parameters: dict = {"n": 1}
    size = fmt.get("size")
    if size:
        parameters["size"] = size.replace("x", "*")

    body = {
        "model": model.name,
        "input": {"prompt": prompt},
        "parameters": parameters,
    }
    return _IMAGE_SYNTHESIS_PATH, body


class QwenClient(OpenAIClient):

    #: Seconds between task-status polls, and the overall ceiling.
    _POLL_INTERVAL = 2.0
    _POLL_TIMEOUT = 300.0

    def __init__(self, api_key: str, *, data: dict, **client_opts) -> None:
        # Region-resolved base URL unless an explicit url is given.
        if not client_opts.get("url"):
            client_opts["url"] = resolve_qwen_base_url(client_opts.pop("region", None))
        super().__init__(api_key, data=data, **client_opts)

    # ── format ───────────────────────────────────────────────────────
    def build_request(self, messages, output, params) -> "tuple[str, dict]":
        m = self._wrap(params)
        if _is_qwen_image(m.name):
            return _build_qwen_image_request(m, messages, output)
        return super().build_request(messages, output, params)

    # ── request lifecycle ────────────────────────────────────────────
    def send(self, path: str, body: dict, headers: dict) -> bytes:
        # wanx text-to-image is the only async path; everything else is a
        # single POST handled by the base implementation.
        if path == _IMAGE_SYNTHESIS_PATH:
            return self._image_synthesis(body, headers)
        return self._post(path, body, headers)

    # ── async image synthesis: submit → poll → collect ───────────────
    def _image_synthesis(self, body: dict, headers: dict) -> bytes:
        submit_headers = {**headers, "X-DashScope-Async": "enable"}
        submitted = _json.loads(self._post(_IMAGE_SYNTHESIS_PATH, body, submit_headers))
        task_id = submitted.get("output", {}).get("task_id")
        if not task_id:
            raise TaskFailedError(502, f"DashScope returned no task_id: {submitted}")

        output = self._poll_task(task_id, headers)
        return self._collect(output)

    def _poll_task(self, task_id: str, headers: dict) -> dict:
        deadline = time.monotonic() + self._POLL_TIMEOUT
        while True:
            payload = _json.loads(self._get(_TASKS_PATH + task_id, headers))
            output = payload.get("output", {})
            status = output.get("task_status")

            if status == "SUCCEEDED":
                return output
            if status in ("FAILED", "CANCELED", "UNKNOWN"):
                code = output.get("code", status)
                message = output.get("message", "")
                raise TaskFailedError(502, f"DashScope task {task_id} {status}: {code} {message}".rstrip())

            if time.monotonic() >= deadline:
                raise TaskFailedError(
                    504,
                    f"DashScope task {task_id} did not finish within "
                    f"{self._POLL_TIMEOUT:.0f}s (last status: {status}).",
                )
            time.sleep(self._POLL_INTERVAL)

    def _collect(self, output: dict) -> bytes:
        """Download every result URL and shape it like an images response."""
        items: list = []
        for result in output.get("results", []):
            url = result.get("url")
            if not url:
                continue
            blob = self._download(url)
            items.append({"b64_json": base64.b64encode(blob["data"]).decode("ascii")})
        if not items:
            raise TaskFailedError(502, f"DashScope task returned no image results: {output}")
        return _json.dumps({"data": items}).encode("utf-8")
