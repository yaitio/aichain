"""The model layer: ``Model``, the data-driven ``registry``, and the call types.

There is one ``Model`` class; the provider is resolved from the model name
against the ``providers/`` data and a family client speaks its wire format.
See docs/models/models.md and docs/reference/model-registry.md.
"""

from types import SimpleNamespace

from ._options import UNIVERSAL_OPTIONS, accepted_by as _q_accepts
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
    # Which universal options a provider has a control for, so a caller can
    # ask before making the call rather than reading a warning after it.
    # ``None`` means the provider declares nothing, which is not the same as
    # accepting nothing.
    accepts      = _q_accepts,
    OPTIONS      = tuple(UNIVERSAL_OPTIONS),
)

from ._schema import portable_schema, check_structure
from ..clients._families._openai_compat import SchemaNotExpressible
# What an external driver of `Agent.step()` needs, public because `step()`'s
# own error message tells the caller to use `tool_result_turn`.
from ._calls import ToolCall, ToolCallRequest, tool_result_turn, dangling_calls

__all__ = [
    "Model",
    "registry",
    "portable_schema",
    "SchemaNotExpressible",
    "check_structure",
    "UNIVERSAL_OPTIONS",
    "ToolCall",
    "ToolCallRequest",
    "tool_result_turn",
    "dangling_calls",
]
