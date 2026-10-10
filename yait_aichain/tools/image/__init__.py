"""
tools.image
===========

An image model as a tool (3.9.0). ``image(service, model=…)`` makes one —
``openai``, ``google``, ``bfl``, ``xai``, ``qwen`` or ``recraft`` —
built on the model layer's provider data: the same request, the same option
vocabulary, the same notices when a provider adapts or declines an option.

::

    picture = image("openai", model="gpt-image-1.5", output_dir="images/")
    out = picture.run("a lighthouse at dawn, watercolor", options={"aspect_ratio": "16:9"})
    out["images"][0]["path"], out["cost"]
    picture.run("make the sky stormy", images=[out["images"][0]["path"]])   # an edit
    Agent(model, tools=[picture])

Until 3.9.0 this was ``Skill(Model("gpt-image-1.5"), output={"modalities": ["image"], …})``,
which still works and warns; it raises in 4.0.
"""

from ._tool import OPTIONS, Image

#: The providers with image models — the ones `image()` takes.
SERVICES = ("openai", "google", "bfl", "xai", "qwen", "recraft")

#: Providers that were here and closed, with what to say instead of "unknown".
CLOSED = {"reve": "Reve closed its API on 2026-08-14; it was removed in 3.10.0"}


def image(service: str, *, model: str, api_key: "str | None" = None, output_dir: "str | None" = None,
          name: "str | None" = None, description: "str | None" = None) -> Image:
    """
    An image tool for *service* — one of ``SERVICES`` — and *model*.

    The model is a setting: an image's look, price and what it can do
    (transparency, vectors, edits) come with it. ``output_dir`` is where images
    are saved; without it they go back to the code in the result.
    """
    if isinstance(service, str) and service.lower().strip() in CLOSED:
        raise ValueError(f"{CLOSED[service.lower().strip()]}; one of: {', '.join(SERVICES)}")
    if not isinstance(service, str) or service.lower().strip() not in SERVICES:
        raise ValueError(f"unknown image service {service!r}; one of: {', '.join(SERVICES)}")
    tool = Image(model, api_key=api_key, output_dir=output_dir, name=name, description=description)
    if tool.service != service.lower().strip():
        raise ValueError(f"{model} is {tool.service}'s, not {service}'s — image({tool.service!r}, model={model!r})")
    return tool


__all__ = ["image", "SERVICES", "Image", "OPTIONS"]

from .._callable import callable_package as _callable_package
_callable_package(__name__, image)
