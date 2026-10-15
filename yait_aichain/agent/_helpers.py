"""
agent._helpers — small functions the agent, its loop and its built-ins share.
"""

from __future__ import annotations

import json
import copy
import logging

from .._dispatch import MAX_ARGUMENT_DEPTH  # noqa: F401  (named in the release notes)

from ._result  import AgentResult

_logger = logging.getLogger("yait_aichain.agent")


def _interrupted(result) -> bool:
    """A run that stopped on the model rather than on its own: worth a retry."""
    return (isinstance(result, AgentResult) and not result.success
            and result.stopped_by in ("llm_error", "truncated"))


def _copy(value):
    """A copy for an event; the value itself when it cannot be copied."""
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def _action(call) -> str:
    """A call as the journal records it: name and arguments shortened for reading,
    plus a hash of the whole arguments, so calls that differ only past the cut do
    not read as one repeated."""
    import hashlib
    full = json.dumps(call.arguments or {}, ensure_ascii=False, default=str, sort_keys=True)
    digest = hashlib.sha256(full.encode()).hexdigest()[:12]
    return f"{_safe({'name': call.name, 'arguments': call.arguments})} #{digest}"


def _safe(value, limit: int = 200):
    """A short rendering for logs and the journal."""
    text = json.dumps(value, ensure_ascii=False, default=str) \
        if not isinstance(value, str) else value
    return text if len(text) <= limit else text[:limit] + "…"
