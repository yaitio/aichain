"""
tools._moved — the paths modules had before 3.5.0, still importable.

``Chain.save()`` writes a tool by its full class path, and code imports by
module path; both have to keep working after a module moves. Each old path is
the moved module itself — the same object, registered under its old name — so
``import``, ``from … import``, ``importlib.import_module`` (what ``Chain.load``
uses), attribute access and ``mock.patch`` all reach it, on every supported
Python. A class keeps its new ``__module__``, so a chain saved from now on
writes the new path.
"""

from __future__ import annotations

import importlib
import sys

#: old module path → new module path, both under ``yait_aichain.tools``.
MOVED = {
    "embedding":           "embed",
    "embedding._base":     "embed._base",
    "embedding._openai":   "embed._openai",
    "embedding._cohere":   "embed._cohere",
    "embedding._voyage":   "embed._voyage",
    "embedding._google":   "embed._google",
    "embedding._qwen":     "embed._qwen",
    "reranking":           "rerank",
    "reranking._base":     "rerank._base",
    "reranking._cohere":   "rerank._cohere",
    "reranking._voyage":   "rerank._voyage",
    "reranking._qwen":     "rerank._qwen",
    "convert.to_speech":   "speech._tts",
    "convert.to_text":     "transcribe._stt",
    "vectordb._chunk":     "chunk._chunk",
    "_wait":               "general._wait",
}


def register(package: str = "yait_aichain.tools") -> None:
    for old, new in MOVED.items():
        module = importlib.import_module(f"{package}.{new}")
        sys.modules[f"{package}.{old}"] = module
        parent_name, _, leaf = f"{package}.{old}".rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is not None:
            setattr(parent, leaf, module)
