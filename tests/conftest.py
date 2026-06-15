"""
Pytest bootstrap for the legacy test layout.

The test suite was written when the library's sub-packages lived at the
repository top level (``models/``, ``skills/``, ``chain/`` ...).  After the
code moved into the ``yait_aichain`` package those imports broke.  Until the
tests are migrated to ``yait_aichain.*`` imports, alias the sub-packages —
and their submodules — so ``from models import Model
``from skills._adapters import substitute`` keep working.

Submodules must be aliased under their canonical ``yait_aichain.*`` names:
importing ``skills._adapters`` as a top-level module would re-execute it
with ``__package__ = "skills"``, breaking relative imports like
``from .._template import ...``.
"""

import importlib
import pkgutil
import sys

_SUBPACKAGES = ("models", "skills", "chain", "pool", "agent", "clients", "tools", "state")

for _name in _SUBPACKAGES:
    _pkg = importlib.import_module(f"yait_aichain.{_name}")
    sys.modules.setdefault(_name, _pkg)
    for _info in pkgutil.walk_packages(
        _pkg.__path__, prefix=f"yait_aichain.{_name}."
    ):
        try:
            _mod = importlib.import_module(_info.name)
        except Exception:
            # Optional dependency missing (markitdown, weasyprint, ...) —
            # the legacy test importing it will fail on its own terms.
            continue
        sys.modules.setdefault(_info.name[len("yait_aichain."):], _mod)
