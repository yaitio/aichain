"""
tools.transcribe
================

An audio file written down as text — one format of options, a provider per
tool. ``transcribe(service)`` makes one by the provider's name; until 3.5.0
these lived in ``tools.convert.to_text`` and were made by ``STT(provider)``,
and both still work.

Tools
-----
sttOpenAI   OpenAI ``gpt-4o-transcribe`` and ``whisper-1``.
sttGoogle   Google Gemini.
sttXAI      xAI.
sttQwen     Qwen (DashScope).
"""

from ._stt import convertToText, sttGoogle, sttOpenAI, sttQwen, sttXAI

#: Provider name → tool class, for :func:`transcribe`.
SERVICES = {"openai": sttOpenAI, "google": sttGoogle, "xai": sttXAI, "qwen": sttQwen}


def transcribe(service: str, *, model: "str | None" = None, api_key: "str | None" = None,
               name: "str | None" = None, description: "str | None" = None) -> convertToText:
    """
    A speech-to-text tool for *service* — one of ``SERVICES``.

    ::

        transcribe("openai").run("call.mp3", {"language": "ru"})    # the transcript
    """
    from .._callable import named, pick
    cls = pick("transcribe", SERVICES, service)
    return named(cls(api_key=api_key, **({"model": model} if model is not None else {})), name, description)


__all__ = ["transcribe", "SERVICES", "convertToText", "sttOpenAI", "sttGoogle", "sttXAI", "sttQwen"]

from .._callable import callable_package as _callable_package
_callable_package(__name__, transcribe)
