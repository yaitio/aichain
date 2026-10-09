"""
yait_aichain._moved — the paths modules had before they moved, still importable.

``Chain.save()`` writes a tool by its full class path and code imports by
module path, so both have to keep working when a module moves. An old path is
the moved module itself — the same object, under its old name too — which
``import``, ``from … import``, ``import … as``, attribute access,
``mock.patch`` (3.10 and 3.11 resolve it through attributes) and
``importlib.import_module`` (what ``Chain.load`` uses) all reach. A class
keeps its new ``__module__``, so a chain saved from now on writes the new path.

The map is by prefix and the longest wins, so a package's submodules — the
vector store's ``providers._chroma`` and the rest — come with it without being
listed. Installed first thing in ``yait_aichain/__init__.py``; held by
``tests/test_tools_3_5.py`` and ``tests/test_utilities.py``.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys

#: old path → new path. 3.5.0 moved the tool families; 3.6.0 the utilities.
MOVED = {
    "yait_aichain.tools.embedding":         "yait_aichain.tools.embed",
    "yait_aichain.tools.reranking":         "yait_aichain.tools.rerank",
    "yait_aichain.tools.convert":           "yait_aichain.tools._convert_removed",
    "yait_aichain.tools.convert.to_speech": "yait_aichain.tools.speech._tts",
    "yait_aichain.tools.convert.to_text":   "yait_aichain.tools.transcribe._stt",
    "yait_aichain.tools.vectordb._chunk":   "yait_aichain.tools.chunk._chunk",
    "yait_aichain.tools._wait":             "yait_aichain.tools.general._wait",
    "yait_aichain.tools.vectordb":          "yait_aichain.utilities.vectors",
    "yait_aichain.tools.mcp":               "yait_aichain.utilities.mcp._tools",
    "yait_aichain.tools.rest_api":          "yait_aichain.utilities.rest._tool",
    "yait_aichain.tools.local":             "yait_aichain.utilities.shell",
}


#: Modules removed outright (3.8.0): importing one says what replaced it.
REMOVED = {
    "yait_aichain.tools.convert.to_md":   "convertToMD",
    "yait_aichain.tools.convert.to_html": "convertToHTML",
    "yait_aichain.tools.convert.to_pdf":  "convertToPDF",
}


def new_path(name: str) -> "str | None":
    """Where *name* lives now, or None when it has not moved."""
    for old in sorted(MOVED, key=len, reverse=True):
        if name == old or name.startswith(old + "."):
            return MOVED[old] + name[len(old):]
    return None


class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in REMOVED:
            from .tools._convert_removed import removed
            raise removed(REMOVED[fullname])
        if new_path(fullname) is None:
            return None
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec):
        # The moved module itself: its __name__, __spec__ and __file__ stay
        # its own — module_from_spec sets them only where they are missing.
        return importlib.import_module(new_path(spec.name))

    def exec_module(self, module):
        pass                                     # already executed under its new name


def install() -> None:
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
