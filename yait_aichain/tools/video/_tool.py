"""
``Video``, a video model as a tool. A video takes minutes, so a call is a job:
``submit`` returns it at once, ``collect`` waits for it and downloads the file,
``run`` is the two together. The job is a small dict that crosses a process.
See docs/tools/media.md.
"""

from __future__ import annotations

import base64
import hashlib
import os
import time
import warnings
from pathlib import Path

from .._base import Record, Tool
from .._security import check_input_file
from ...clients._errors import TaskFailedError
from ...models._adaptation import ADAPTED, DECLINED, Adaptation, announce, collect, record
from ...models._usage import price_per_second
from ._jobs import COMPLETED, FAILED, adapter

#: The video vocabulary — `options` keys every video provider is asked in.
OPTIONS = {
    "duration":        {"type": "integer", "description": "Seconds; the nearest length the model makes."},
    "resolution":      {"type": "string", "description": "480p, 720p, 1080p or 4k; the nearest the model has."},
    "aspect_ratio":    {"type": "string", "description": "e.g. '16:9'; the nearest the model takes."},
    "size":            {"type": "string", "description": "Pixels, e.g. '1280x720'; becomes a ratio and a resolution."},
    "seed":            {"type": "integer", "description": "For a repeatable video where the model honours it."},
    "audio":           {"type": "boolean", "description": "With sound or without, where the model lets you choose."},
    "negative_prompt": {"type": "string", "description": "What the video should not show, where the model takes it."},
}
_HEIGHT = {"480p": 480, "720p": 720, "1080p": 1080, "4k": 2160}
_STEP, _LONGEST_STEP = 5.0, 30.0         # seconds between polls: from, growing to


def _ratio(text: str) -> float:
    a, _, b = str(text).partition(":")
    return float(a) / float(b)


class Video(Tool):
    """
    A video model as a tool — from a prompt, or from an image as the first frame.
    Made by :func:`~tools.video` — ``video("google", model="veo-3.1-fast-generate-preview")``.

    ``output_dir`` is where a video is saved, and the agent is told its path;
    without it the bytes go back to the code that called, in the result.
    ``timeout`` is how long ``run`` and ``collect`` wait before raising with
    the job, which ``collect`` can be given again later.
    """

    name        = "video"
    risk        = "draft"
    priced      = True
    charges_itself = True        # it knows its price and charges the budgets in force itself
    _safe_for_untrusted_load = True
    parameters  = {
        "type": "object",
        "properties": {
            "input":   {"type": "string", "description": "What the video shows."},
            "images":  {"type": "array", "items": {"type": "string"}, "maxItems": 1,
                        "description": "One image as the first frame — a file path or a URL. None for text alone."},
            "options": {"type": "object", "description": "How to make it.", "properties": dict(OPTIONS)},
        },
        "required": ["input"],
    }

    def __init__(self, model: str, *, api_key: str | None = None, output_dir: str | None = None,
                 timeout: float = 600.0, name: str | None = None, description: str | None = None) -> None:
        from ...models import Model
        from ...models._data import PROVIDERS
        self.model = model
        self._model = Model(model, api_key=api_key)
        self.service = self._model._provider
        self._data = PROVIDERS[self.service].get("models", {}).get(model) or {}
        caps = set(self._data.get("caps") or ())
        if not caps & {"text-to-video", "image-to-video"}:
            raise ValueError(f"{model} is not a video model; registry.models(task='text-to-video') lists them")
        self._jobs = adapter(self.service, self._model.client, model, self._data)
        self.output_dir, self.timeout = output_dir, float(timeout)
        self.name = name or f"video{self.service.capitalize() if self.service != 'xai' else 'XAI'}"
        self.description = description or (
            f"Make a video with {self.service}'s {model} — from a prompt, or from an image as its "
            "first frame. Takes minutes. Returns where the video was saved, or that it went back to the code.")

    # ── the job ────────────────────────────────────────────────────────────

    def submit(self, input: str, images: list | None = None, options: dict | None = None) -> dict:
        """Start a video and return its job at once: ``{"service", "model", "id", …}``."""
        if not isinstance(input, str) or not input.strip():
            raise ValueError(f"{self.name} needs a prompt — what the video shows")
        opts = dict(options or {})
        if unknown := sorted(set(opts) - set(OPTIONS)):
            raise ValueError(f"{self.name}: unknown option(s) {unknown}; it takes {sorted(OPTIONS)}")
        if isinstance(images, str):
            images = [images]
        if images and len(images) > 1:
            raise ValueError(f"{self.name} takes one image, the first frame; got {len(images)}")
        frame = self._frame(images[0]) if images else None
        with collect() as made:
            plan = self._plan(opts, frame is not None)
        announce(made)
        notices = [_notice(a) for a in made]
        rate = price_per_second(self.model, plan["resolution"])
        estimate = round(rate * plan["duration"], 4) if rate is not None else None
        notices += self._budget(estimate)
        job_id = self._jobs.submit(input.strip(), frame, plan)
        return {"service": self.service, "model": self.model, "id": job_id, "plan": plan,
                "estimate": estimate, "notices": notices, "submitted_at": time.time()}

    def status(self, job: dict) -> dict:
        """Where the job is: ``{"status": "running" | "completed" | "failed", …}``."""
        return self._jobs.poll(job["id"])

    def collect(self, job: dict, timeout: float | None = None) -> Record:
        """Wait for *job*, download its video, price it. Raises with the job when the
        wait runs out, so it can be collected again instead of made again."""
        deadline, step = time.monotonic() + (self.timeout if timeout is None else timeout), _STEP
        while (state := self.status(job))["status"] not in (COMPLETED, FAILED):
            if time.monotonic() + step > deadline:
                error = TaskFailedError(504, f"{self.model} is still rendering job {job['id']}; "
                                             f"collect(job) again later — making it again would pay twice")
                error.job = job
                raise error
            time.sleep(step)
            step = min(step * 1.5, _LONGEST_STEP)
        if state["status"] == FAILED:
            error = TaskFailedError(502, f"{self.model} did not make the video: {state.get('error')}")
            error.job = job
            raise error
        return self._result(job, state)

    def run(self, input: str, images: list | None = None, options: dict | None = None) -> Record:  # type: ignore[override]
        return self.collect(self.submit(input, images, options))

    # ── what was asked, fitted to the model ─────────────────────────────────

    def _plan(self, opts: dict, has_frame: bool) -> dict:
        d, caps = self._data, set(self._data.get("caps") or ())
        if has_frame and "image-to-video" not in caps:
            raise ValueError(f"{self.model} makes video from text only; give no image, or use a model "
                             "with image-to-video")
        if not has_frame and "text-to-video" not in caps:
            raise ValueError(f"{self.model} makes video from an image; give one as the first frame")
        size = opts.get("size")
        wh = [int(n) for n in str(size).lower().split("x")] if size else None
        ratio = opts.get("aspect_ratio") or (f"{wh[0]}:{wh[1]}" if wh else None)
        plan = {"aspect_ratio": self._nearest_ratio(ratio, size), "seed": None, "negative_prompt": None}
        asked = str(opts.get("resolution") or (f"{min(wh)}p" if wh else "")).lower().replace("2160p", "4k")
        plan["resolution"] = self._nearest_resolution(asked, size)
        plan["duration"] = self._length(opts.get("duration"), plan["resolution"])
        for option, value in (("seed", opts.get("seed")), ("negative_prompt", opts.get("negative_prompt"))):
            if value is not None:
                if d.get(option):
                    plan[option] = value
                else:
                    self._say(DECLINED, option, value, None, "this model has no such control")
        if opts.get("audio") is not None:
            always = d.get("audio") == "always"
            if bool(opts.get("audio")) != always:
                self._say(DECLINED, "audio", opts.get("audio"), always,
                          "this model always makes sound" if always else "this model makes no sound")
        return plan

    def _nearest_ratio(self, asked, size) -> str:
        allowed = list(self._data.get("aspect_ratios") or ["16:9"])
        if not asked:
            return "16:9" if "16:9" in allowed else allowed[0]
        try:
            near = min(allowed, key=lambda r: abs(_ratio(r) - _ratio(asked)))
        except (ValueError, ZeroDivisionError):
            raise ValueError(f"{self.model}: aspect_ratio {asked!r} is not a ratio; it takes {', '.join(allowed)}") from None
        if abs(_ratio(near) - _ratio(asked)) > 0.01:                  # 1920:1080 is 16:9
            self._say(ADAPTED, "size" if size else "aspect_ratio", size or asked, near,
                      f"this model takes {', '.join(allowed)}; the nearest went")
        return near

    def _nearest_resolution(self, asked: str, size) -> str:
        allowed = list(self._data.get("resolutions") or ["720p"])
        if not asked:
            return "720p" if "720p" in allowed else allowed[0]
        if asked not in _HEIGHT and not size:
            raise ValueError(f"{self.model}: resolution {asked!r} is not one of {', '.join(_HEIGHT)}")
        height = _HEIGHT.get(asked) or int(asked.rstrip("p"))
        near = min(allowed, key=lambda r: abs(_HEIGHT[r] - height))
        if near != asked:
            self._say(ADAPTED, "size" if size else "resolution", size or asked, near,
                      f"this model makes {', '.join(allowed)}; the nearest went")
        return near

    def _length(self, asked, resolution: str) -> int:
        d = self._data
        allowed = list(d.get("durations") or range(d["duration_range"][0], d["duration_range"][1] + 1))
        if resolution in (d.get("full_length") or ()):
            if asked not in (None, max(allowed)):
                self._say(ADAPTED, "duration", asked, max(allowed),
                          f"this model makes {resolution} at {max(allowed)} seconds only")
            return max(allowed)
        want = 5 if asked is None else int(asked)
        near = min(allowed, key=lambda s: (abs(s - want), -s))      # a tie goes to the longer
        if asked is not None and near != want:
            self._say(ADAPTED, "duration", asked, near,
                      f"this model makes {_lengths(allowed)} seconds; the nearest went")
        return near

    def _say(self, kind: str, option: str, asked, sent, why: str) -> None:
        record(Adaptation(kind=kind, option=option, asked=asked, sent=sent, why=why, model=self.model))

    # ── money, the frame, the result ─────────────────────────────────────────

    def _budget(self, estimate: float | None) -> list:
        """A budget with nothing left stops the job; one with less left than the
        estimate is warned about, and the job goes ahead."""
        from ..._budget import in_force
        out = []
        for budget in in_force(None):
            budget.check(f"{self.model} for {self.name}")
            if estimate is not None and estimate > budget.remaining:
                message = (f"{self.name}: this video is estimated at ${estimate:.2f} and the budget has "
                           f"${budget.remaining:.2f} left; it goes ahead and the budget will be overspent")
                warnings.warn(message, RuntimeWarning, stacklevel=3)
                out.append(f"budget: {message}")
        return out

    def _frame(self, ref: str) -> dict:
        if not isinstance(ref, str) or not ref.strip():
            raise ValueError(f"the first frame is a file path or a URL; got {ref!r}")
        if ref.lower().startswith(("http://", "https://")):
            got = self._model.client._download(ref)
            raw, mime = got["data"], (got.get("media_type") or "image/png").split(";")[0]
        else:
            check_input_file(ref, "image")
            from ...skills._adapters import _guess_mime
            with open(ref, "rb") as fh:
                raw = fh.read()
            mime = _guess_mime(Path(ref), raw[:16])
        b64 = base64.b64encode(raw).decode("ascii")
        return {"mime_type": mime, "base64": b64, "data_uri": f"data:{mime};base64,{b64}"}

    def _result(self, job: dict, state: dict) -> Record:
        plan = job["plan"]
        data = self._jobs.fetch(state["url"])
        seconds = state.get("duration") or plan["duration"]
        resolution = state.get("resolution") or plan["resolution"]
        clip = {"path": None, "data": data, "mime_type": "video/mp4", "duration": seconds,
                "resolution": resolution}
        if self.output_dir:
            clip["path"], clip["data"] = self._save(data), None
        cost, basis = state.get("cost"), f"{self.service} reported it"
        if cost is None and (rate := price_per_second(self.model, resolution)) is not None:
            cost, basis = round(rate * seconds, 4), f"{seconds} s at ${rate}/s, the registry's {resolution} rate"
        from ..._budget import in_force
        for budget in in_force(None):
            budget.charge(cost)
        return Record({"videos": [clip], "model": self.model, "service": self.service,
                       "job": {k: job[k] for k in ("service", "model", "id")},
                       "notices": list(job.get("notices") or []), "cost": cost,
                       "cost_basis": basis if cost is not None else
                       f"not known — {self.service} reports none and the registry has no price"},
                      self.render)

    def render(self, out: dict) -> str:
        """What a model reads: what was made and where it is — never the bytes."""
        lines = []
        for v in out["videos"]:
            made = f"Made a {v['duration']}-second {v['resolution']} video with {out['model']}"
            lines.append(f"{made}, saved to {v['path']}." if v.get("path") else
                         f"{made}; it went back to the code that called, and is not shown here.")
        return "\n".join(lines + [f"Note: {n}" for n in out["notices"]])

    def _save(self, data: bytes) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        stem = f"video-{time.strftime('%Y%m%d-%H%M%S')}-{hashlib.sha256(data).hexdigest()[:8]}"
        path = os.path.join(self.output_dir, f"{stem}.mp4")
        with open(path, "wb") as fh:
            fh.write(data)
        return path

    @staticmethod
    def _untrusted_load_refusal(init_args: dict) -> str | None:
        """A folder to write into, named by the file, needs a file you trust."""
        if init_args.get("output_dir"):
            return (f"it writes into {init_args['output_dir']!r}, a folder the file chose. "
                    "Pass trusted=True to Chain.load() for a file you wrote.")
        return None

    def _serialise_init_args(self) -> dict:
        """What `Chain.save` writes: the model, the directory, the wait — never the key."""
        out = {"model": self.model, "timeout": self.timeout}
        if self.output_dir:
            out["output_dir"] = self.output_dir
        return out


def _lengths(allowed: list) -> str:
    if len(allowed) > 3 and allowed == list(range(allowed[0], allowed[-1] + 1)):
        return f"{allowed[0]} to {allowed[-1]}"
    return ", ".join(map(str, allowed[:-1])) + f" or {allowed[-1]}" if len(allowed) > 1 else str(allowed[0])


def _notice(adaptation) -> str:
    why = getattr(adaptation, "why", "") or ""
    return f"{adaptation.option}: {adaptation.kind}" + (f" — {why}" if why else "")
