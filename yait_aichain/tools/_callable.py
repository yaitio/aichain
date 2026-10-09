"""
tools._callable — a package that is its own factory.

``search("exa")`` calls the ``yait_aichain.tools.search`` package itself. A
function by the package's name in ``tools`` would hide the package behind it,
and every path that resolves through that attribute would fail — ``mock.patch``
on Python 3.10 and 3.11, ``import yait_aichain.tools.search.exa as m`` and
attribute access on any version. Found running 3.3.0 on 3.10; held by
``tests/test_factories_keep_their_packages.py``.
"""

from __future__ import annotations

import inspect
import sys
import types


class _CallableModule(types.ModuleType):
    def __call__(self, *args, **kwargs):
        return self.__factory__(*args, **kwargs)


def callable_package(name: str, factory) -> None:
    """Make the package *name* call *factory*, with its signature for help() and IDEs."""
    module = sys.modules[name]
    module.__class__ = _CallableModule
    module.__factory__ = factory
    module.__signature__ = inspect.signature(factory)


def named(tool, name=None, description=None):
    """A tool with *name* and *description* in place of its class's, where given."""
    if name:
        tool.name = name
    if description:
        tool.description = description
    return tool


def pick(kind: str, services: dict, service):
    """The class for *service*, or a ValueError naming the services that exist."""
    cls = services.get(service.lower().strip()) if isinstance(service, str) else None
    if cls is None:
        raise ValueError(f"unknown {kind} service {service!r}; one of: {', '.join(services)}")
    return cls
