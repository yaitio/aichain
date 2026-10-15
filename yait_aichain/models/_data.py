"""Loads the provider data files, ``models/providers/*.toml``, once at import
into ``PROVIDERS`` (provider key → ``[provider]`` settings and ``[models]``).
On 3.10 TOML parsing needs the optional ``tomli`` package.
See docs/models/models.md.
"""

from __future__ import annotations

import os

try:
    import tomllib as _toml          # Python 3.11+
except ModuleNotFoundError:          # pragma: no cover
    try:
        import tomli as _toml        # Python 3.10
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ImportError(
            "Reading provider data needs tomllib (Python 3.11+) or 'tomli' "
            "on 3.10. Install it with: pip install tomli"
        ) from exc

_DIR = os.path.join(os.path.dirname(__file__), "providers")


def _load() -> dict:
    providers: dict = {}
    for fname in sorted(os.listdir(_DIR)):
        if not fname.endswith(".toml"):
            continue
        with open(os.path.join(_DIR, fname), "rb") as fh:
            data = _toml.load(fh)
        key = data["provider"]["key"]
        providers[key] = data
    return providers


#: provider key → {"provider": {...}, "models": {name: {...}}}
PROVIDERS: dict[str, dict] = _load()


def provider_of(model_name: str) -> str | None:
    """Return the provider key that owns *model_name*, or None."""
    for key, data in PROVIDERS.items():
        if model_name in data.get("models", {}):
            return key
    return None
