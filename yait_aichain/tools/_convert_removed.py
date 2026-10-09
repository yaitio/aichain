"""
tools._convert_removed — what ``yait_aichain.tools.convert`` became in 3.8.0.

The convert tools left the library: they were one project's case, their
dependencies were heavy and undeclared (MarkItDown brings onnxruntime and numpy,
114 MB; ``convertToHTML`` and ``convertToPDF`` needed packages no extra
installed, and WeasyPrint system libraries pip cannot install), and their
example now lives in the cookbooks. A page from its URL is ``fetch("local")``.

The old path still imports, so a saved chain or old code fails with that said
rather than with ``ModuleNotFoundError``: speech and transcription, which lived
in ``tools.convert`` until 3.5.0, come from their packages; the removed names
raise ``ImportError`` with the way forward.
"""

from .speech import TTS, convertToSpeech, ttsGoogle, ttsOpenAI, ttsQwen, ttsXAI
from .transcribe import STT, convertToText, sttGoogle, sttOpenAI, sttQwen, sttXAI
from .speech import _tts as to_speech            # its submodules, as attributes too
from .transcribe import _stt as to_text

#: A package, so its old submodules (``to_speech``, ``to_text``) resolve under it.
__path__: list = []

#: name → what replaced it.
REMOVED = {
    "convertToMD":    'a page from its URL is fetch("local"); a local file, MarkItDown itself — '
                      "see the cookbook's conversion recipe",
    "convertToHTML":  "mistletoe itself — see the cookbook's conversion recipe",
    "convertToPDF":   "WeasyPrint itself — see the cookbook's conversion recipe",
}
REMOVED["MarkItDownTool"] = REMOVED["convertToMD"]
REMOVED["MistletoeTool"] = REMOVED["convertToHTML"]
REMOVED["WeasyprintTool"] = REMOVED["convertToPDF"]


def removed(name: str) -> ImportError:
    return ImportError(f"{name} was removed from yait-aichain in 3.8.0, with the other convert tools: "
                       f"{REMOVED[name]}.")


def __getattr__(name):
    if name in REMOVED:
        raise removed(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
