"""
A video model as a tool: ``video(service, model=…)``. A video takes minutes, so
a call is a job — ``submit``, ``collect``, or ``run`` for both — with the image
tool's option vocabulary for its shape and a notice when a provider adapts.
See docs/tools/media.md.
"""

from ._tool import OPTIONS, Video

#: The providers with video models — the ones `video()` takes.
SERVICES = ("google", "xai", "qwen", "minimax", "kling", "byteplus")

#: Providers that were here, or were asked about, and say so instead of "unknown".
CLOSED = {"openai": "OpenAI closed Sora and its Videos API on 2026-09-24"}


def video(service: str, *, model: str, api_key: "str | None" = None, output_dir: "str | None" = None,
          timeout: float = 600.0, name: "str | None" = None, description: "str | None" = None) -> Video:
    """
    A video tool for *service* — one of ``SERVICES`` — and *model*.

    ``output_dir`` is where videos are saved; without it they go back to the code
    in the result. ``timeout`` is how long a call waits for the video, in seconds.
    """
    key = service.lower().strip() if isinstance(service, str) else service
    if key in CLOSED:
        raise ValueError(f"{CLOSED[key]}; one of: {', '.join(SERVICES)}")
    if key not in SERVICES:
        raise ValueError(f"unknown video service {service!r}; one of: {', '.join(SERVICES)}")
    tool = Video(model, api_key=api_key, output_dir=output_dir, timeout=timeout, name=name,
                 description=description)
    if tool.service != key:
        raise ValueError(f"{model} is {tool.service}'s, not {service}'s — video({tool.service!r}, model={model!r})")
    return tool


__all__ = ["video", "SERVICES", "Video", "OPTIONS"]

from .._callable import callable_package as _callable_package
_callable_package(__name__, video)
