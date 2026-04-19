"""
tools.convert
=============

Format conversion tools — all share the ``run(input, options=None)`` interface.

Tools
-----
convertToMD       Convert any file or URL to Markdown (via MarkItDown).
convertToHTML     Convert Markdown to HTML, LaTeX, or normalised Markdown.
convertToPDF      Render HTML to PDF (via WeasyPrint).
convertToSpeech   Synthesise speech from text (TTS).  Returns audio file path.
convertToText     Transcribe an audio file to text (STT).  Returns transcript.

Factory functions
-----------------
TTS(provider)   Return a ``convertToSpeech`` instance for the given provider.
STT(provider)   Return a ``convertToText``   instance for the given provider.

Supported providers: ``"openai"``, ``"google"``, ``"xai"``, ``"qwen"``

Provider classes
----------------
ttsOpenAI, ttsGoogle, ttsXAI, ttsQwen
sttOpenAI, sttGoogle, sttXAI, sttQwen

Backward-compatible aliases
---------------------------
MarkItDownTool, MistletoeTool, WeasyprintTool
"""

from .to_md     import convertToMD
from .to_html   import convertToHTML
from .to_pdf    import convertToPDF
from .to_speech import convertToSpeech, ttsOpenAI, ttsGoogle, ttsXAI, ttsQwen
from .to_text   import convertToText,   sttOpenAI, sttGoogle, sttXAI, sttQwen

# Backward-compatible aliases
MarkItDownTool = convertToMD
MistletoeTool  = convertToHTML
WeasyprintTool = convertToPDF

__all__ = [
    # ── Conversion tools ──────────────────────────────────────────────────
    "convertToMD",
    "convertToHTML",
    "convertToPDF",
    "convertToSpeech",
    "convertToText",
    # ── TTS providers ─────────────────────────────────────────────────────
    "ttsOpenAI",
    "ttsGoogle",
    "ttsXAI",
    "ttsQwen",
    # ── STT providers ─────────────────────────────────────────────────────
    "sttOpenAI",
    "sttGoogle",
    "sttXAI",
    "sttQwen",
    # ── Factory functions ─────────────────────────────────────────────────
    "TTS",
    "STT",
    # ── Backward-compatible aliases ───────────────────────────────────────
    "MarkItDownTool",
    "MistletoeTool",
    "WeasyprintTool",
]


# ── Factory functions ─────────────────────────────────────────────────────────

_TTS_MAP: dict = {
    "openai": ttsOpenAI,
    "google": ttsGoogle,
    "xai":    ttsXAI,
    "qwen":   ttsQwen,
}

_STT_MAP: dict = {
    "openai": sttOpenAI,
    "google": sttGoogle,
    "xai":    sttXAI,
    "qwen":   sttQwen,
}


def TTS(provider: str, *, api_key: "str | None" = None) -> convertToSpeech:
    """
    Return a TTS tool instance for *provider*.

    Parameters
    ----------
    provider : str
        One of ``"openai"``, ``"google"``, ``"xai"`` (case-insensitive).
    api_key : str | None, optional
        Provider API key.  Falls back to the provider's environment variable
        when omitted.

    Returns
    -------
    convertToSpeech
        A ready-to-use TTS tool instance.

    Raises
    ------
    ValueError
        When *provider* is not recognised.

    Examples
    --------
    ::

        from tools.convert import TTS

        tts  = TTS("openai")
        path = tts.run(input="Hello!", options={"voice": "nova"})

        tts  = TTS("google")
        path = tts.run(
            input="Bonjour le monde!",
            options={"language": "fr-FR", "output_path": "bonjour.mp3"},
        )
    """
    key = provider.lower().strip()
    cls = _TTS_MAP.get(key)
    if cls is None:
        raise ValueError(
            f"Unknown TTS provider {provider!r}.  "
            f"Supported: {', '.join(_TTS_MAP)}."
        )
    return cls(api_key=api_key) if api_key else cls()


def STT(provider: str, *, api_key: "str | None" = None) -> convertToText:
    """
    Return an STT tool instance for *provider*.

    Parameters
    ----------
    provider : str
        One of ``"openai"``, ``"google"``, ``"xai"`` (case-insensitive).
    api_key : str | None, optional
        Provider API key.  Falls back to the provider's environment variable
        when omitted.

    Returns
    -------
    convertToText
        A ready-to-use STT tool instance.

    Raises
    ------
    ValueError
        When *provider* is not recognised.

    Examples
    --------
    ::

        from tools.convert import STT

        stt  = STT("openai")
        text = stt.run(input="meeting.mp3", options={"language": "en"})

        stt  = STT("xai")
        text = stt.run(input="call.wav")
    """
    key = provider.lower().strip()
    cls = _STT_MAP.get(key)
    if cls is None:
        raise ValueError(
            f"Unknown STT provider {provider!r}.  "
            f"Supported: {', '.join(_STT_MAP)}."
        )
    return cls(api_key=api_key) if api_key else cls()
