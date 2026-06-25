"""
yait-aichain
============

One interface. 11 providers. 81 models.
Text, image generation & editing, vision, RAG, agents, parallel pipelines.

    pip install yait-aichain

Quick start::

    from yait_aichain import Model, Skill

    skill = Skill(
        model = Model("claude-sonnet-4-6"),
        input = {"messages": [{"role": "user", "parts": ["What is {topic}?"]}]},
    )
    result = skill.run(variables={"topic": "machine learning"})

Full imports::

    from yait_aichain.models import Model
    from yait_aichain.skills import Skill
    from yait_aichain.chain  import Chain
    from yait_aichain.pool   import Pool, DONE, FAILED
    from yait_aichain.agent  import Agent
    from yait_aichain.tools  import convertToMD, searchPerplexity, Embedding
"""

__version__ = "1.4.6"
__author__  = "YAIT"

# ── Logging ────────────────────────────────────────────────────────────────────
# The library emits through named loggers (``yait_aichain.*``) and never
# configures a sink itself — the application/environment decides where logs go.
# A NullHandler on the root package logger suppresses the "No handlers could be
# found" warning until a handler is attached.
import logging as _logging
_logging.getLogger("yait_aichain").addHandler(_logging.NullHandler())

# ── Convenience re-exports ─────────────────────────────────────────────────────

from .models import Model                                          # noqa: F401
from .models._usage import Usage                                   # noqa: F401
from .skills import Skill                                          # noqa: F401
from .chain  import Chain                                          # noqa: F401
from .pool   import Pool, PENDING, RUNNING, DONE, FAILED          # noqa: F401
from .agent  import Agent                                          # noqa: F401

# Observability events & lifecycle hooks (1.4.4).
from ._events import Event, Hook, Tracer, LoggingTracer            # noqa: F401
# Permission matrix for tool governance (1.4.4).
from .tools._permissions import PermissionPolicy                   # noqa: F401

# Exception hierarchy — catch APIError for everything, or a subclass for a
# specific failure mode (rate limit, auth, server error, …).
from .clients import (                                             # noqa: F401
    APIError,
    NetworkError,
    RateLimitError,
    AuthenticationError,
    InsufficientCreditsError,
    InvalidRequestError,
    NotFoundError,
    ServerError,
    TaskFailedError,
)
