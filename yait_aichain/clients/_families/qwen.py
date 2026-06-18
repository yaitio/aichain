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
from ._openai_compat import _image_source_to_data_uri
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

# Image *editing* rides a different, synchronous endpoint (confirmed by live
# probe): the multimodal-generation API, with a message-shaped body and the
# input image as a base64 data-URI (or URL). No polling.
_MULTIMODAL_GEN_PATH = "/api/v1/services/aigc/multimodal-generation/generation"


def _is_qwen_image_edit(name: str) -> bool:
    return "image-edit" in name.lower()   # qwen-image-edit / -plus / -max


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


def _build_qwen_image_edit_request(model, messages: list, output: dict) -> "tuple[str, dict]":
    """
    Build the DashScope multimodal-generation (image-edit) *(path, body)* pair.

    Synchronous endpoint. The input image(s) ride ``content[].image`` as a
    base64 data-URI (or a URL); the edit instruction is a ``content[].text``
    item. Size is converted from our ``"1280x720"`` to DashScope's ``"1280*720"``.
    """
    input_messages: list[dict] = []
    for msg in messages:
        content: list[dict] = []
        for part in msg["parts"]:
            if part["type"] == "text":
                content.append({"text": part["text"]})
            elif part["type"] == "image" and isinstance(part.get("source"), dict):
                content.append({"image": _image_source_to_data_uri(part["source"])})
        if content:
            input_messages.append({"role": msg["role"], "content": content})

    parameters: dict = {"n": 1}
    size = output.get("format", {}).get("size")
    if size:
        parameters["size"] = size.replace("x", "*")

    body = {"model": model.name,
            "input": {"messages": input_messages},
            "parameters": parameters}
    return _MULTIMODAL_GEN_PATH, body


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
        if _is_qwen_image_edit(m.name):
            return _build_qwen_image_edit_request(m, messages, output)
        if _is_qwen_image(m.name):
            return _build_qwen_image_request(m, messages, output)
        return super().build_request(messages, output, params)

    # ── request lifecycle ────────────────────────────────────────────
    def send(self, path: str, body: dict, headers: dict) -> bytes:
        # wanx text-to-image is async (submit → poll → download); image-edit is
        # a synchronous multimodal-generation call that returns result URLs.
        # Everything else defers to the base (JSON POST / multipart) seam.
        if path == _IMAGE_SYNTHESIS_PATH:
            return self._image_synthesis(body, headers)
        if path == _MULTIMODAL_GEN_PATH:
            return self._image_edit_sync(body, headers)
        return super().send(path, body, headers)

    # ── synchronous image edit: POST → download result URL(s) → base64 ─
    def _image_edit_sync(self, body: dict, headers: dict) -> bytes:
        """
        Run the synchronous edit and reshape its result into the
        ``{"data": [{"b64_json": …}]}`` form the image parser already consumes.

        The response carries result image **URLs** under
        ``output.choices[].message.content[].image``; download each and base64
        it (the same normalisation ``_collect`` does for the async wan path).
        An empty result falls through to the parser's descriptive error.
        """
        resp = _json.loads(self._post(_MULTIMODAL_GEN_PATH, body, headers))
        items: list = []
        for choice in resp.get("output", {}).get("choices", []):
            content = (choice.get("message") or {}).get("content") or []
            for c in content:
                url = c.get("image")
                if url:
                    blob = self._download(url)
                    items.append({"b64_json": base64.b64encode(blob["data"]).decode("ascii")})
        return _json.dumps({"data": items}).encode("utf-8")

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
