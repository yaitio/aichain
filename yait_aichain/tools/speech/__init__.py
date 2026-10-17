"""
Text read aloud into an audio file: ``speech(service)`` makes a tool by the
provider's name (``SERVICES``); the classes behind it share :class:`Speech`.
See docs/tools/media.md.
"""

from ._tts import Speech, ttsElevenLabs, ttsGoogle, ttsMiniMax, ttsOpenAI, ttsQwen, ttsXAI

#: Provider name → tool class, for :func:`speech`.
SERVICES = {"openai": ttsOpenAI, "google": ttsGoogle, "xai": ttsXAI, "qwen": ttsQwen,
            "elevenlabs": ttsElevenLabs, "minimax": ttsMiniMax}


def speech(service: str, *, model: "str | None" = None, api_key: "str | None" = None,
           region: "str | None" = None, name: "str | None" = None,
           description: "str | None" = None, output_dir: "str | None" = None) -> Speech:
    """
    A text-to-speech tool for *service* — one of ``SERVICES``.

    ``model`` defaults to the provider's (Google's has one and takes none);
    ``region`` is Qwen's alone; ``name`` and ``description`` default to the
    provider's; ``api_key`` to its environment variable. ``output_dir`` is
    where the audio files go — the system's temporary directory by default.

    ::

        speech("openai").run("Hello!", {"voice": "nova"})    # the path of the audio file
    """
    from .._callable import named, pick
    cls = pick("speech", SERVICES, service)
    if region is not None and cls is not ttsQwen:
        raise TypeError(f"region is speech('qwen')'s; speech({service!r}) takes api_key, name and description")
    extra = {"region": region} if region is not None else {}
    if model is not None:
        extra["model"] = model
    tool = cls(api_key=api_key, output_dir=output_dir, **extra)
    return named(tool, name, description)


__all__ = ["speech", "SERVICES", "Speech", "ttsOpenAI", "ttsGoogle", "ttsXAI", "ttsQwen", "ttsElevenLabs",
           "ttsMiniMax"]

from .._callable import callable_package as _callable_package
_callable_package(__name__, speech)
