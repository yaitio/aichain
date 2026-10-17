"""
A video job, provider by provider: the request built from a plan, the submit,
the poll, the download. A job is a small dict — service, model, id — so it can
cross a process, a store or a queue the way a run document does.

Each adapter speaks through the provider's own client, for its base URL, its
auth and its retries; none of this goes through a Skill, because a job is not
one request but three kinds over minutes.
"""

from __future__ import annotations

import json

from ...clients._errors import TaskFailedError

#: What every adapter's poll answers: running, completed or failed.
RUNNING, COMPLETED, FAILED = "running", "completed", "failed"


class _Adapter:
    def __init__(self, client, model: str) -> None:
        self.client, self.model = client, model

    def _headers(self) -> dict:
        return self.client._auth_headers()

    def fetch(self, url: str) -> bytes:
        return self.client._download(url, headers=self._fetch_headers())["data"]

    def _fetch_headers(self) -> dict | None:
        return None


class Google(_Adapter):
    """Veo through the Gemini API: predictLongRunning, an operation to poll, a file
    downloaded with the key."""

    def submit(self, prompt: str, image: dict | None, plan: dict) -> str:
        instance: dict = {"prompt": prompt}
        if image:
            # The Vertex shape: the guide shows inlineData, and the endpoint answers it
            # with 400 "inlineData isn't supported by this model" (2026-10-17).
            instance["image"] = {"bytesBase64Encoded": image["base64"], "mimeType": image["mime_type"]}
        parameters = {"aspectRatio": plan["aspect_ratio"], "durationSeconds": plan["duration"],
                      "resolution": plan["resolution"]}
        if plan.get("seed") is not None:
            parameters["seed"] = plan["seed"]
        out = json.loads(self.client._post(f"/models/{self.model}:predictLongRunning",
                                           {"instances": [instance], "parameters": parameters},
                                           self._headers()))
        return out["name"]

    def poll(self, job_id: str) -> dict:
        op = json.loads(self.client._get(f"/{job_id}", self._headers()))
        if not op.get("done"):
            return {"status": RUNNING}
        if op.get("error"):
            return {"status": FAILED, "error": op["error"].get("message") or str(op["error"])}
        answer = (op.get("response") or {}).get("generateVideoResponse") or {}
        samples = answer.get("generatedSamples") or []
        if not samples:
            why = answer.get("raiMediaFilteredReasons") or ["no video in the response"]
            return {"status": FAILED, "error": "; ".join(map(str, why))}
        return {"status": COMPLETED, "url": samples[0]["video"]["uri"]}

    def _fetch_headers(self) -> dict:
        key = self._headers().get("x-goog-api-key")
        return {"x-goog-api-key": key} if key else {}


class XAI(_Adapter):
    """Grok Imagine: /v1/videos/generations, a request id to poll, a URL to download.
    The job reports its own cost, in ticks of 1e-10 USD."""

    def submit(self, prompt: str, image: dict | None, plan: dict) -> str:
        body = {"model": self.model, "prompt": prompt, "duration": plan["duration"],
                "resolution": plan["resolution"], "aspect_ratio": plan["aspect_ratio"]}
        if image:
            body["image"] = {"url": image["data_uri"]}
        return json.loads(self.client._post("/v1/videos/generations", body, self._headers()))["request_id"]

    def poll(self, job_id: str) -> dict:
        job = json.loads(self.client._get(f"/v1/videos/{job_id}", self._headers()))
        status = job.get("status")
        if status in ("failed", "expired"):
            error = job.get("error") or {}
            return {"status": FAILED, "error": error.get("message") or error.get("code") or status}
        if status != "done":
            return {"status": RUNNING, "progress": job.get("progress")}
        ticks = (job.get("usage") or {}).get("cost_in_usd_ticks")
        video = job.get("video") or {}
        return {"status": COMPLETED, "url": video.get("url"), "duration": video.get("duration"),
                "cost": ticks / 1e10 if isinstance(ticks, (int, float)) else None}


class Qwen(_Adapter):
    """Wan on Model Studio: an asynchronous task, polled, its video_url downloaded.
    Wan 2.7 reads a resolution and a ratio; Wan 2.2 reads a pixel size."""

    PATH = "/api/v1/services/aigc/video-generation/video-synthesis"

    def __init__(self, client, model: str, data: dict) -> None:
        super().__init__(client, model)
        self.data = data

    def submit(self, prompt: str, image: dict | None, plan: dict) -> str:
        inputs: dict = {"prompt": prompt}
        if image and self.data.get("frame") == "media":       # Wan 2.7: a list of typed media
            inputs["media"] = [{"type": "first_frame", "url": image["data_uri"]}]
        elif image:
            inputs["img_url"] = image["data_uri"]
        if plan.get("negative_prompt"):
            inputs["negative_prompt"] = plan["negative_prompt"]
        if self.data.get("shape") == "size":
            parameters = {"size": self.data["sizes"][plan["resolution"]][plan["aspect_ratio"]]}
        elif image and self.data.get("frame") == "media":     # the shape follows the first frame
            parameters = {"resolution": plan["resolution"].upper()}
        else:
            parameters = {"resolution": plan["resolution"].upper(), "ratio": plan["aspect_ratio"]}
        if plan.get("audio") is not None:
            parameters["audio"] = plan["audio"]
        if len(self.data.get("durations") or []) != 1:       # a model of one length takes no field
            parameters["duration"] = plan["duration"]
        if plan.get("seed") is not None:
            parameters["seed"] = plan["seed"]
        headers = {**self._headers(), "X-DashScope-Async": "enable"}
        out = json.loads(self.client._post(self.PATH, {"model": self.model, "input": inputs,
                                                       "parameters": parameters}, headers))
        return out["output"]["task_id"]

    def poll(self, job_id: str) -> dict:
        task = json.loads(self.client._get(f"/api/v1/tasks/{job_id}", self._headers()))
        out, usage = task.get("output") or {}, task.get("usage") or {}
        status = out.get("task_status")
        if status in ("FAILED", "CANCELED", "UNKNOWN"):
            return {"status": FAILED, "error": out.get("message") or out.get("code") or status}
        if status != "SUCCEEDED":
            return {"status": RUNNING}
        height = usage.get("SR")
        if height is None and usage.get("video_ratio"):              # Wan 2.2: "1920*1080"
            height = min(int(n) for n in str(usage["video_ratio"]).split("*"))
        return {"status": COMPLETED, "url": out.get("video_url"),
                "duration": usage.get("output_video_duration") or usage.get("video_duration")
                or usage.get("duration"),
                "resolution": f"{height}p" if height else None}


class MiniMax(_Adapter):
    """MiniMax H3: /v2/video_generation with a content list, a task to poll, a URL to
    download. The shape of a text-to-video comes from the ratio; with a first frame,
    from the image."""

    def submit(self, prompt: str, image: dict | None, plan: dict) -> str:
        content: list = [{"type": "text", "text": prompt}]
        if image:
            content.append({"type": "image_url", "image_url": {"url": image["data_uri"]}, "role": "first_frame"})
        body = {"model": self.model, "content": content, "resolution": plan["resolution"].upper(),
                "duration": plan["duration"], "ratio": "adaptive" if image else plan["aspect_ratio"]}
        return json.loads(self.client._post("/v2/video_generation", body, self._headers()))["task_id"]

    def poll(self, job_id: str) -> dict:
        task = json.loads(self.client._get(f"/v2/query/video_generation/{job_id}", self._headers())).get("task") or {}
        status = task.get("status")
        if status in ("failed", "cancelled"):
            error = task.get("error") or {}
            return {"status": FAILED, "error": (error.get("message") if isinstance(error, dict) else error) or status}
        if status != "succeeded":
            return {"status": RUNNING}
        return {"status": COMPLETED, "url": (task.get("content") or {}).get("url"),
                "duration": (task.get("usage") or {}).get("output_seconds")}


def adapter(service: str, client, model: str, data: dict) -> _Adapter:
    if service == "google":
        return Google(client, model)
    if service == "xai":
        return XAI(client, model)
    if service == "qwen":
        return Qwen(client, model, data)
    if service == "minimax":
        return MiniMax(client, model)
    raise TaskFailedError(400, f"no video adapter for {service}")
