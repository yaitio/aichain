"""
``Image``, an image model as a tool: an image model neither reasons nor calls
tools, so it is not a model an agent can think with. The request is built by
the same ``Skill`` and ``Model``, from the same provider data. A call returns a
``Record``: the image for code, a line for a model. See docs/tools/media.md.
"""

from __future__ import annotations

from .._security import check_input_file

import base64
import hashlib
import os
import time

from .._base import Record, Tool
from ...models._usage import price_per_image

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
    priced      = True
    charges_itself = True       # its Skill charges what the usage prices; the rest below
    _safe_for_untrusted_load = True      # see Tool._safe_for_untrusted_load
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

    def __init__(self, model: str, *, api_key: str | None = None, output_dir: str | None = None,
                 name: str | None = None, description: str | None = None) -> None:
        from ...models import Model
        self.model = model
        self._model = Model(model, api_key=api_key)
        self.service = self._model._provider
        self.output_dir = output_dir
        self.name = name or f"image{self.service.capitalize() if self.service != 'xai' else 'XAI'}"
        self.description = description or (
            f"Make an image with {self.service}'s {model} — from a prompt, or by editing the images "
            "given. Returns where the image was saved, or that it went back to the code.")

    def run(self, input: str, images: list | None = None, options: dict | None = None) -> Record:  # type: ignore[override]
        from ...skills import Skill
        if not isinstance(input, str) or not input.strip():
            raise ValueError(f"{self.name} needs a prompt — what to draw or what to change")
        opts = dict(options or {})
        if unknown := sorted(set(opts) - set(OPTIONS)):
            raise ValueError(f"{self.name}: unknown option(s) {unknown}; it takes {sorted(OPTIONS)}")
        if isinstance(images, str):
            images = [images]           # one path, not a list of its characters
        parts = [_image_part(i) for i in (images or [])] + [input.strip()]
        # Through a Skill, by the seam only this tool uses: retries, the budget
        # and the record of adaptations come with it.
        skill = Skill(self._model, input={"messages": [{"role": "user", "parts": parts}]},
                      output={"modalities": ["image"], "format": {"type": "image", **opts}},
                      _image=True)
        over = self._over_budget(price_per_image(self.model))
        out = skill.run()
        picture = {"url": out.get("url"), "base64": out.get("base64"), "mime_type": out.get("mime_type"),
                   "revised_prompt": out.get("revised_prompt"), "path": None}
        if self.output_dir and picture["base64"]:
            picture["path"] = self._save(picture)
        usage = skill.last_usage
        cost = getattr(usage, "cost", None)
        basis = "the provider's reported usage, at the registry's rate"
        if cost is None and (each := price_per_image(self.model)) is not None:
            # Priced by the picture: the registry holds the base size's rate,
            # so a larger picture on a tiered model costs more than this says.
            cost, basis = each, "the registry's price per image, at the base size"
            # The Skill charged nothing it could not price; this it can.
            from ..._budget import in_force
            for budget in in_force(None):
                budget.charge(cost)
        # "translated" is an option passed exactly, in the provider's word; the
        # rest change what was asked — those are what a reader needs to know.
        notices = [_notice(a) for a in skill.last_adaptations if getattr(a, "kind", "") != "translated"] + over
        return Record({"images": [picture], "model": self.model, "service": self.service, "notices": notices,
                       "cost": cost,
                       "cost_basis": (basis if cost is not None
                                      else f"not known — {self.service} reports no usage for {self.model} "
                                           "and the registry has no price for it"),
                       "usage": _usage_dict(usage)}, self.render)

    def _over_budget(self, estimate: float | None) -> list:
        """Warn, and go ahead, when the picture's price is more than a budget in force has left."""
        from ..._budget import in_force
        out = []
        for budget in in_force(None):
            if estimate is not None and not budget.exhausted and estimate > budget.remaining:
                message = (f"{self.name}: this image is priced at ${estimate:.3f} and the budget has "
                           f"${budget.remaining:.3f} left; it goes ahead and the budget will be overspent")
                import warnings
                warnings.warn(message, RuntimeWarning, stacklevel=3)
                out.append(f"budget: {message}")
        return out

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

    @staticmethod
    def _untrusted_load_refusal(init_args: dict) -> str | None:
        """A folder to write into, named by the file, needs a file you trust."""
        if init_args.get("output_dir"):
            return (f"it writes into {init_args['output_dir']!r}, a folder the file chose. "
                    "Pass trusted=True to Chain.load() for a file you wrote.")
        return None

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
    check_input_file(ref, "image")
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
