"""
The API reference is the code's own account of itself, and covers all of it.

`docs/api/` is rendered by `.github/scripts/docs.py` from signatures and docstrings,
so an API page cannot drift from the code — but it can be silent. Held here:
every public name is on a page (or exempt, with the reason), and the number
of parameters a docstring does not describe only goes down.
"""

import importlib
import inspect
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".github" / "scripts"))
import _apidoc  # noqa: E402
import docs  # noqa: E402

MODULES = ("yait_aichain", "yait_aichain.agent", "yait_aichain.tools", "yait_aichain.utilities",
           "yait_aichain.eval", "yait_aichain.state", "yait_aichain.models")

#: Public, and deliberately without an entry of its own.
EXEMPT = {
    "Utility": "the base of the four utilities, which are documented",
    "Scorer": "a type alias for a scoring function, described on Eval",
    "registry": "a module of lookups, rendered as the model registry page",
    "UNIVERSAL_OPTIONS": "rendered as the parameters reference",
    "Decision": "base of the decision tools", "Search": "base of the search tools",
    "Fetch": "base of the fetch tools", "Embedder": "base of the embedding tools",
    "RerankBase": "base of the reranking tools", "Speech": "base of the tts* tools",
    "Transcription": "base of the stt* tools", "Image": "what image() returns, on its page",
}

#: Each is one string, named in the table of risk classes or decisions on the control page.
RISK_AND_DECISION_CONSTANTS = {"READ", "DRAFT", "WRITE", "EXTERNAL", "FINANCIAL", "DESTRUCTIVE",
                               "PRIVILEGED", "ALLOW", "APPROVE", "DENY"}

#: Parameters (of a constructor, function or method on an API page) that no
#: docstring describes. Lowered as they are written, never raised.
UNDOCUMENTED_CEILING = 52


def _covered() -> tuple:
    """The names on a page, and the objects behind them — an alias of either is covered."""
    names = {n for items in docs.API_PAGES.values() for _, n in items}
    objects = [getattr(importlib.import_module(m), n) for items in docs.API_PAGES.values() for m, n in items]
    names |= {n for leads in docs.TOOL_PAGE_LEAD.values() for n in leads}
    objects += [docs._lead(n) for leads in docs.TOOL_PAGE_LEAD.values() for n in leads]
    for _, classes in docs.TOOL_PAGES.values():
        names |= set(classes)
        objects += [docs._cls(n) for n in classes]
    return names, objects


def _public(module):
    mod = importlib.import_module(module)
    for name in getattr(mod, "__all__", None) or [n for n in dir(mod) if not n.startswith("_")]:
        try:
            obj = getattr(mod, name)
        except Exception:
            continue
        if inspect.ismodule(obj) and not callable(obj):
            continue
        yield name, obj


def _documented(doc) -> set:
    names = set()
    for title, lines in _apidoc._split_sections(doc or ""):
        if title in ("Parameters", "Attributes"):
            for entry, _, _ in _apidoc._entries(lines):
                names |= {x.strip().lstrip("*") for x in entry.split(",")}
    return names


def undocumented() -> list:
    out = []

    def check(label, fn, doc):
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            return
        have = _documented(doc)
        for p in sig.parameters.values():
            if p.name in ("self", "cls") or p.name.startswith("_") or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue
            if p.name not in have:
                out.append(f"{label}({p.name})")

    for items in docs.API_PAGES.values():
        for mod, name in items:
            obj = getattr(importlib.import_module(mod), name)
            if inspect.isclass(obj):
                # An inherited constructor is described where it is defined.
                if "__init__" in obj.__dict__:
                    check(name, obj.__init__, inspect.getdoc(obj))
                for method, member, kind in _apidoc.public_methods(obj):
                    if kind != "property":
                        check(f"{name}.{method}", member, inspect.getdoc(member))
            elif callable(obj):
                check(name, obj, inspect.getdoc(obj))
    return out


class TestTheApiReference(unittest.TestCase):

    def test_every_public_name_has_a_page(self):
        names, objects = _covered()
        for module in MODULES:
            for name, obj in _public(module):
                if name in names or name in EXEMPT or name in RISK_AND_DECISION_CONSTANTS:
                    continue
                if any(obj is o for o in objects):
                    continue
                with self.subTest(module=module, name=name):
                    self.fail(f"{module}.{name} is public and on no API page — add it to "
                              "API_PAGES in .github/scripts/docs.py, or to EXEMPT here with the reason")

    def test_undocumented_parameters_only_go_down(self):
        missing = undocumented()
        self.assertLessEqual(len(missing), UNDOCUMENTED_CEILING,
                             f"{len(missing)} parameters with no description, ceiling "
                             f"{UNDOCUMENTED_CEILING}:\n" + "\n".join(missing))
        self.assertEqual(len(missing), UNDOCUMENTED_CEILING,
                         f"only {len(missing)} left — lower UNDOCUMENTED_CEILING to it")


if __name__ == "__main__":
    unittest.main()
