"""
yait-aichain
============

The provider-agnostic AI toolkit for Python: one interface over every provider
in the registry, for skills, chains, pools, agents and their tools.

    pip install yait-aichain

Quick start::

    from yait_aichain import Model, Skill

    skill = Skill(
        model = Model("claude-haiku-5-5"),
        input = {"messages": [{"role": "user", "parts": ["What is {topic}?"]}]},
    )
    result = skill.run(variables={"topic": "machine learning"})

Full imports::

    from yait_aichain.models import Model
    from yait_aichain.skills import Skill
    from yait_aichain.chain  import Chain
    from yait_aichain.pool   import Pool, DONE, FAILED
    from yait_aichain.agent  import Agent
    from yait_aichain.tools  import fetch, search, embed
"""

__version__ = "4.0.0"
__author__  = "YAIT"

# ── Logging ────────────────────────────────────────────────────────────────────
# The library emits through named loggers (``yait_aichain.*``) and never
# configures a sink itself — the application/environment decides where logs go.
# A NullHandler on the root package logger suppresses the "No handlers could be
# found" warning until a handler is attached.
import logging as _logging
_logging.getLogger("yait_aichain").addHandler(_logging.NullHandler())

# A module that does not exist names where its contents live
# (`yait_aichain.hooks` → `from yait_aichain import Tracer`). See _hints.py.

from ._hints import install_module_hints as _install_module_hints
_install_module_hints()

# ── Convenience re-exports ─────────────────────────────────────────────────────

from .models import Model
# Schema portability, public because every caller was writing it by hand:
# the two provider families demand opposite forms of the same schema.
from .models import portable_schema, check_structure, SchemaNotExpressible
# What the library had to change about a request to fit a provider. Reachable
# so a caller can log it, not only read the warning.
from .models._adaptation import Adaptation, UnsupportedOption
from ._budget import Budget, BudgetExceeded
from .models._usage import Usage
from .skills import Skill
from .chain  import Chain
from .pool   import Pool, PENDING, RUNNING, DONE, FAILED
from .chain  import ChainResult
from .pool   import PoolResult
from .agent  import Agent, AgentResult

# Observability events & lifecycle hooks (1.4.4).
from ._events import Event, Hook, Tracer, LoggingTracer
# Permission matrix for tool governance (1.4.4).
from .tools._permissions import (PermissionPolicy, ApprovalRequest,
                                 ApprovalDecision)

# Exception hierarchy — catch APIError for everything, or a subclass for a
# specific failure mode (rate limit, auth, server error, …).
from .clients import (
    APIError,
    NetworkError,
    RateLimitError,
    AuthenticationError,
    InsufficientCreditsError,
    InvalidRequestError,
    TruncatedResponseError,
    InvalidStructuredOutputError,
    NotFoundError,
    ServerError,
    TaskFailedError,
)

#: What `from yait_aichain import *` brings: the names above, and not the
#: subpackages their imports load along the way.
__all__ = [
    "Model", "portable_schema", "check_structure", "SchemaNotExpressible", "Adaptation", "UnsupportedOption",
    "Budget", "BudgetExceeded", "Usage",
    "Skill", "Chain", "ChainResult", "Pool", "PoolResult", "PENDING", "RUNNING", "DONE", "FAILED",
    "Agent", "AgentResult",
    "Event", "Hook", "Tracer", "LoggingTracer",
    "PermissionPolicy", "ApprovalRequest", "ApprovalDecision",
    "APIError", "NetworkError", "RateLimitError", "AuthenticationError",
    "InsufficientCreditsError", "InvalidRequestError", "TruncatedResponseError",
    "InvalidStructuredOutputError", "NotFoundError", "ServerError", "TaskFailedError",
]
