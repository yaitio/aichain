"""
yait-aichain — the provider-agnostic AI toolkit for Python: one interface over
every provider in the registry, for skills, chains, pools, agents and tools.

    from yait_aichain import Model, Skill
    Skill(Model("claude-haiku-5-5"), prompt="What is {topic}?").run(variables={"topic": "AI"})

See docs/start/quickstart.md.
"""

__version__ = "4.6.1"
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

# ── The documentation of this version ─────────────────────────────────────────

def docs_dir():
    """The documentation of the installed version, as a directory.

    A released wheel carries ``docs/`` and ``llms.txt`` inside the package, so
    an agent writing code against it reads the pages of the version it will
    run rather than what it remembers of an older one. In a checkout the
    repository's own ``docs/`` is returned. ``None`` when neither is there.
    """
    from pathlib import Path
    here = Path(__file__).resolve().parent
    for candidate in (here / "_docs", here.parent / "docs"):
        if (candidate / "index.md").is_file():
            return candidate
    return None


# ── Convenience re-exports ─────────────────────────────────────────────────────

from .models import Model
# Schema portability, public because every caller would write it by hand:
# the two provider families demand opposite forms of the same schema.
from .models import portable_schema, check_structure, SchemaNotExpressible
# What the library had to change about a request to fit a provider. Reachable
# so a caller can log it, not only read the warning.
from .models._adaptation import Adaptation, UnsupportedOption
from ._budget import Budget, BudgetExceeded
from .models._usage import Usage
from ._dispatch import Block, Interceptor
from .skills import Skill
from .chain  import Chain
from .pool   import Pool, PENDING, RUNNING, DONE, FAILED
from .chain  import ChainResult
from .pool   import PoolResult
from .agent  import Agent, AgentResult

# Observability events & lifecycle hooks.
from ._events import Event, Hook, Tracer, LoggingTracer
# Permission matrix for tool governance.
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
    ContextWindowExceededError,
    TruncatedResponseError,
    InvalidStructuredOutputError,
    NotFoundError,
    ServerError,
    TaskFailedError,
)

#: What `from yait_aichain import *` brings: the names above, and not the
#: subpackages their imports load along the way.
__all__ = [
    "docs_dir",
    "Model", "portable_schema", "check_structure", "SchemaNotExpressible", "Adaptation", "UnsupportedOption",
    "Budget", "BudgetExceeded", "Usage", "Block", "Interceptor",
    "Skill", "Chain", "ChainResult", "Pool", "PoolResult", "PENDING", "RUNNING", "DONE", "FAILED",
    "Agent", "AgentResult",
    "Event", "Hook", "Tracer", "LoggingTracer",
    "PermissionPolicy", "ApprovalRequest", "ApprovalDecision",
    "APIError", "NetworkError", "RateLimitError", "AuthenticationError",
    "InsufficientCreditsError", "InvalidRequestError", "ContextWindowExceededError", "TruncatedResponseError",
    "InvalidStructuredOutputError", "NotFoundError", "ServerError", "TaskFailedError",
]
