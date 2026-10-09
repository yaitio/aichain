"""
tools.speech
============

Text read aloud into an audio file — one format of options, a provider per
tool. ``speech(service)`` makes one by the provider's name; until 3.5.0 these
lived in ``tools.convert.to_speech`` and were made by ``TTS(provider)``, and
both still work.

Tools
-----
ttsOpenAI   OpenAI ``gpt-4o-mini-tts`` and ``tts-1``.
ttsGoogle   Google Gemini TTS.
ttsXAI      xAI.
ttsQwen     Qwen (DashScope); takes ``region``.
"""

from ._tts import convertToSpeech, ttsGoogle, ttsOpenAI, ttsQwen, ttsXAI

#: Provider name → tool class, for :func:`speech`.
SERVICES = {"openai": ttsOpenAI, "google": ttsGoogle, "xai": ttsXAI, "qwen": ttsQwen}


def speech(service: str, *, api_key: "str | None" = None, region: "str | None" = None,
           name: "str | None" = None, description: "str | None" = None) -> convertToSpeech:
    """
    A text-to-speech tool for *service* — one of ``SERVICES``.

    ``region`` is Qwen's alone; ``name`` and ``description`` default to the
    provider's; ``api_key`` to its environment variable.

    ::

        speech("openai").run("Hello!", {"voice": "nova"})    # the path of the audio file
    """
    from .._callable import named, pick
    cls = pick("speech", SERVICES, service)
    if region is not None and cls is not ttsQwen:
        raise TypeError(f"region is speech('qwen')'s; speech({service!r}) takes api_key, name and description")
    tool = cls(api_key=api_key, **({"region": region} if region is not None else {}))
    return named(tool, name, description)


__all__ = ["speech", "SERVICES", "convertToSpeech", "ttsOpenAI", "ttsGoogle", "ttsXAI", "ttsQwen"]

from .._callable import callable_package as _callable_package
_callable_package(__name__, speech)
