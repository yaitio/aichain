"""
tools.transcribe
================

An audio file written down as text — one format of options, a provider per
tool. ``transcribe(service)`` makes one by the provider's name; the classes
behind it share the base :class:`Transcription`.

Tools
-----
sttOpenAI   OpenAI ``gpt-4o-transcribe`` and ``whisper-1``.
sttGoogle   Google Gemini.
"""

from ._stt import Transcription, sttGoogle, sttOpenAI

#: Provider name → tool class, for :func:`transcribe`.
SERVICES = {"openai": sttOpenAI, "google": sttGoogle}


def transcribe(service: str, *, model: "str | None" = None, api_key: "str | None" = None,
               name: "str | None" = None, description: "str | None" = None) -> Transcription:
    """
    A speech-to-text tool for *service* — one of ``SERVICES``.

    ``model`` defaults to the provider's and is a setting of the tool, not an
    option of a call; ``api_key`` defaults to the provider's environment
    variable.

    ::

        transcribe("openai").run("call.mp3", {"language": "ru"})    # the transcript
    """
    from .._callable import named, pick
    cls = pick("transcribe", SERVICES, service)
    return named(cls(api_key=api_key, **({"model": model} if model is not None else {})), name, description)


__all__ = ["transcribe", "SERVICES", "Transcription", "sttOpenAI", "sttGoogle"]

from .._callable import callable_package as _callable_package
_callable_package(__name__, transcribe)
