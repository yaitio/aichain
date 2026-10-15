"""
pool — one runner over many items, in parallel.

``Pool(runner, items, *, max_flows, on_error)`` runs a Skill, Tool, Chain or
Agent once per item on a thread pool and returns a ``PoolResult`` in item
order. ``PENDING``, ``RUNNING``, ``DONE``, ``FAILED`` and ``CANCELLED`` are the
statuses in ``pool.history`` and ``pool.status``. See docs/pipelines/pool.md.
"""

from ._pool import Pool, PENDING, RUNNING, DONE, FAILED, CANCELLED
from ._result import PoolResult

__all__ = [
    "Pool",
    "PoolResult",
    "PENDING",
    "RUNNING",
    "DONE",
    "FAILED",
    "CANCELLED",
]
