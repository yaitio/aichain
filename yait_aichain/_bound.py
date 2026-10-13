"""
yait_aichain._bound — a generator that runs in the context it was made in.

A generator runs in whichever context advances it. Starlette's
``iterate_in_threadpool`` advances each ``next()`` in a fresh copy of the
caller's context, so a stream lost from its second step what its first had
set — the beacon board, the tenant — and a stream made inside ``with
using(ctx)`` and read after it had no tenant at all. Two streams advanced in
turn in one thread set and reset the same variables out of order.

``bound`` captures the context when the stream is *made* and advances the
generator inside it, every step, closing included.
"""

from __future__ import annotations

import contextvars


def bound(gen, on_end=None):
    """*gen*, advanced in the context current now; *on_end* runs when it is
    exhausted."""
    ctx = contextvars.copy_context()

    def walk():
        try:
            while True:
                try:
                    item = ctx.run(next, gen)
                except StopIteration:
                    break
                yield item
        finally:
            ctx.run(gen.close)
        if on_end is not None:
            on_end()
    return walk()
