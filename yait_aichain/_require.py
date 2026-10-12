"""
An optional dependency, imported where it is needed and named when it is not
there — one message for all of them, instead of thirteen copies in three
wordings.
"""

from __future__ import annotations


def require(module: str, *, package: str, who: str, extra: "str | None" = None):
    """Import *module* and return it, or raise ImportError saying what to install.

    *package* is the name on PyPI, *who* the part of the library that needs
    it, *extra* the yait-aichain extra that brings it, where there is one.
    ``__import__`` rather than ``importlib``, so a test that patches the
    built-in import still reaches it.
    """
    try:
        return __import__(module, fromlist=["__name__"])
    except ImportError:
        how = f'pip install "yait-aichain[{extra}]"' if extra else f"pip install {package}"
        raise ImportError(f"{who} needs {package}. Install it with: {how}") from None
