"""
models
======

Public API for the aichain 2.0 model layer.

Typical usage
-------------
::

    from models import Model

    # Auto-detected provider — returns the correct subclass
    gpt    = Model("gpt-4o")
    sonnet = Model("claude-sonnet-4-5")
    gemini = Model("gemini-2.0-flash")
    grok   = Model("grok-3")
    sonar  = Model("sonar-pro")

    # Override generation parameters
    fast = Model("gpt-4o-mini", options={"temperature": 0.3, "max_tokens": 2048})

    # Override client settings (proxy, timeout, custom URL)
    corp = Model("gpt-4o", client_options={"proxy": {"url": "http://proxy:3128"}})

    # Explicit API key (falls back to env var when omitted)
    model = Model("claude-sonnet-4-5", api_key="sk-ant-...")

    # Query the model registry
    from models import registry
    registry.models(task="text-to-image")           # all text-to-image models
    registry.providers(task="text-to-image")        # ["google", "openai", "xai"]
    registry.tasks("gpt-4o")                        # ["image-to-text", "text-to-text"]
    registry.is_supported("gpt-image-1", "text-to-image")  # True

There is one ``Model`` class: the provider is resolved from the model name
(against the ``providers/`` data) and the matching family client handles the
wire format.  There are no per-provider Model subclasses.
"""

from types import SimpleNamespace

from ._base import (
    Model,
    models       as _q_models,
    providers    as _q_providers,
    tasks        as _q_tasks,
    is_supported as _q_is_supported,
    refresh      as _q_refresh,
    TASKS        as _TASKS,
)

#: Data-driven registry query surface.  Capabilities/prices live in the
#: ``providers/`` data; these functions read it (there is no registry module).
registry = SimpleNamespace(
    models       = _q_models,
    providers    = _q_providers,
    tasks        = _q_tasks,
    is_supported = _q_is_supported,
    refresh      = _q_refresh,
    TASKS        = _TASKS,
    PROVIDERS    = tuple(_q_providers()),
)

__all__ = [
    "Model",
    "registry",
]
