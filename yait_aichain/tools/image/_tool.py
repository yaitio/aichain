"""
tools.image._tool — ``Image``, an image model as a tool.

A model is what can be an agent's mind; an image model cannot — it neither
reasons nor calls tools — so since 3.9.0 it is reached as a tool. Only the face
changed: the request is built by the same ``Skill`` and ``Model`` as before,
from the same provider data, with the same option vocabulary and the same
adaptation notices. What a call returns is a ``Record``: the image for code,
and a line for a model — never the image's bytes in a conversation.
"""

from __future__ import annotations

import base64
import hashlib
import os
import time
import warnings

from .._base import Record, Tool

#: The image vocabulary of the model layer — `output["format"]` keys every
#: image provider is asked in (models._options.UNIVERSAL_FORMAT).
OPTIONS = {
    "size":          {"type": "string", "description": "Pixels, e.g. '1024x1024'; converted to the nearest shape where a provider takes a ratio."},
    "aspect_ratio":  {"type": "string", "description": "e.g. '16:9'; converted to a size where a provider takes pixels."},
    "quality":       {"type": "string", "description": "The provider's quality tier, e.g. 'low', 'medium', 'high'."},
    "background":    {"type": "string", "description": "'transparent' where the model can, else 'opaque'."},
    "output_format": {"type": "string", "description": "png, jpeg or webp."},
    "compression":   {"type": "integer", "description": "0-100, for jpeg and webp."},
    "seed":          {"type": "integer", "description": "For a repeatable image where the provider honours it."},
    "fidelity":      {"type": "string", "description": "How closely an edit keeps its input: 'low' or 'high'."},
}

_EXT = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp", "image/svg+xml": "svg", "image/gif": "gif"}


class _FromImageTool(DeprecationWarning):
    """The Skill-with-an-image-model notice, which this tool's own Skill does not raise."""


class Image(Tool):
    """
    An image model as a tool — generation from a prompt, and editing when the
    call carries images. Made by :func:`~tools.image` — ``image("openai",
    model="gpt-image-1.5")``.

    ``output_dir`` is where an image is saved. Without it the image goes back
    to the code that called, in the result, and an agent is told only that one
    was made; with it the file is written there and the agent is told its path.
    The directory is the code's choice — a model never picks where to write.
    """

    name        = "image"
    risk        = "draft"
    parameters  = {
        "type": "object",
        "properties": {
            "input":   {"type": "string", "description": "What to draw, or what to change in the images given."},
            "images":  {"type": "array", "items": {"type": "string"},
                        "description": "Images to edit or take as reference — file paths or URLs. None to generate."},
            "options": {"type": "object", "description": "How to draw it.", "properties": dict(OPTIONS)},
        },
        "required": ["input"],
    }

    def __init__(self, model: str, *, api_key: "str | None" = None, output_dir: "str | None" = None,
                 name: "str | None" = None, description: "str | None" = None) -> None:
        from ...models import Model
        self.model = model
        self._model = Model(model, api_key=api_key)
        self.service = self._model._provider
        self.output_dir = output_dir
        self.name = name or f"image{self.service.capitalize() if self.service != 'xai' else 'XAI'}"
        self.description = description or (
            f"Make an image with {self.service}'s {model} — from a prompt, or by editing the images "
            "given. Returns where the image was saved, or that it went back to the code.")

    def run(self, input: str, images: "list | None" = None, options: "dict | None" = None) -> Record:  # type: ignore[override]
        from ...skills import Skill
        if not isinstance(input, str) or not input.strip():
            raise ValueError(f"{self.name} needs a prompt — what to draw or what to change")
        opts = dict(options or {})
        if unknown := sorted(set(opts) - set(OPTIONS)):
            raise ValueError(f"{self.name}: unknown option(s) {unknown}; it takes {sorted(OPTIONS)}")
        parts = [_image_part(i) for i in (images or [])] + [input.strip()]
        # The Skill-with-an-image-model notice fires when the Skill is made, so
        # the making goes inside the filter as well as the run.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", _FromImageTool)
            skill = Skill(self._model, input={"messages": [{"role": "user", "parts": parts}]},
                          output={"modalities": ["image"], "format": {"type": "image", **opts}})
            out = skill.run()
        picture = {"url": out.get("url"), "base64": out.get("base64"), "mime_type": out.get("mime_type"),
                   "revised_prompt": out.get("revised_prompt"), "path": None}
        if self.output_dir and picture["base64"]:
            picture["path"] = self._save(picture)
        usage = skill.last_usage
        cost = getattr(usage, "cost", None)
        # "translated" is an option passed exactly, in the provider's word; the
        # rest change what was asked — those are what a reader needs to know.
        notices = [_notice(a) for a in skill.last_adaptations if getattr(a, "kind", "") != "translated"]
        return Record({"images": [picture], "model": self.model, "service": self.service, "notices": notices,
                       "cost": cost,
                       "cost_basis": ("the provider's reported usage, at the registry's rate" if cost is not None
                                      else f"not known — {self.service} reports no usage for {self.model} "
                                           "and the registry has no price for it"),
                       "usage": _usage_dict(usage)}, self.render)

    def render(self, out: dict) -> str:
        """What a model reads: what was made and where it is — never the bytes."""
        lines = []
        for p in out["images"]:
            kind = (p.get("mime_type") or "image").split("/")[-1]
            if p.get("path"):
                lines.append(f"Made a {kind} image with {out['model']}, saved to {p['path']}.")
            elif p.get("url"):
                lines.append(f"Made a {kind} image with {out['model']}: {p['url']}")
            else:
                lines.append(f"Made a {kind} image with {out['model']}; it went back to the code that "
                             "called, and is not shown here.")
            if p.get("revised_prompt"):
                lines.append(f"The model drew from this prompt: {p['revised_prompt']}")
        lines += [f"Note: {n}" for n in out["notices"]]
        return "\n".join(lines)

    def _save(self, picture: dict) -> str:
        data = base64.b64decode(picture["base64"])
        ext = _EXT.get(picture.get("mime_type") or "", "png")
        os.makedirs(self.output_dir, exist_ok=True)
        stem = f"image-{time.strftime('%Y%m%d-%H%M%S')}-{hashlib.sha256(data).hexdigest()[:8]}"
        path = os.path.join(self.output_dir, f"{stem}.{ext}")
        with open(path, "wb") as fh:
            fh.write(data)
        return path

    def _serialise_init_args(self) -> dict:
        """What `Chain.save` writes: the model and the directory, never the key."""
        out = {"model": self.model}
        if self.output_dir:
            out["output_dir"] = self.output_dir
        return out


def _image_part(ref: str) -> dict:
    if not isinstance(ref, str) or not ref.strip():
        raise ValueError(f"an image to edit is a file path or a URL; got {ref!r}")
    if ref.lower().startswith(("http://", "https://")):
        return {"type": "image", "source": {"kind": "url", "url": ref}}
    if not os.path.isfile(ref):
        raise FileNotFoundError(f"no image at {ref!r}")
    return {"type": "image", "source": {"kind": "file", "path": ref}}


def _notice(adaptation) -> str:
    why = getattr(adaptation, "why", "") or ""
    option = getattr(adaptation, "option", "")
    kind = getattr(adaptation, "kind", "")
    return f"{option}: {kind}" + (f" — {why}" if why else "")


def _usage_dict(usage) -> dict:
    if usage is None:
        return {}
    keys = ("input_tokens", "output_tokens", "total_tokens", "image_input_tokens")
    return {k: getattr(usage, k) for k in keys if getattr(usage, k, None)}
